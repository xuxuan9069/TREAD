import argparse
import os
import numpy as np
import json
import pickle
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import h5py
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
import torch

import models
from models import EnsembleEvaluateModel

import loader
import util
import plots

from util import generator_from_config

EARTH_RADIUS = 6371

sns.set(font_scale=1.5)
sns.set_style('ticks')
torch.set_num_threads(10)   # 控制 PyTorch thread

def calculate_warning_times(config, model_list, data, event_metadata, batch_size, sampling_rate=100,
                            times=np.arange(0.5, 25, 0.2), alpha=(0.3, 0.4, 0.5, 0.6, 0.7), use_multiprocessing=True,
                            dataset_id=None, device='cuda'):
    training_params = config['training_params']

    if dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[dataset_id]
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    max_stations = config['model_params']['max_stations']

    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
    generator_params['upsample_high_station_events'] = None

    alpha = np.array(alpha)

    if isinstance(training_params['data_path'], list):
        if dataset_id is not None:
            training_params['data_path'] = training_params['data_path'][dataset_id]
        else:
            training_params['data_path'] = training_params['data_path'][0]

    f = h5py.File(training_params['data_path'], 'r')
    g_data = f['data']
    thresholds = f['metadata']['pga_thresholds'][()] #[0.08 0.25 0.8  1.4  2.5  4.4  8.  ]
    time_before = f['metadata']['time_before'][()] #5

    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')

    event_key = 'data_file' #台灣資料用data_file，日本資料用KiK_File

    full_predictions = []
    coord_keys = util.detect_location_keys(event_metadata.columns)

    for i, _ in tqdm(enumerate(event_metadata.iterrows()), total=len(event_metadata)): #分別進入每個事件
        event = event_metadata.iloc[i]

        event_metadata_tmp = event_metadata.iloc[i:i+1]
        data_tmp = {key: val[i:i+1] for key, val in data.items()}
        generator_params['translate'] = False
        generator = util.PreloadedEventGenerator(data=data_tmp,
                                                 event_metadata=event_metadata_tmp,
                                                 coords_target=False,
                                                 cutout=(0, 3000),
                                                 pga_targets=n_pga_targets,
                                                 max_stations=max_stations,
                                                 sampling_rate=sampling_rate,
                                                 select_first=True,
                                                 shuffle=False,
                                                 pga_mode=True,
                                                 **generator_params)

        cutout_generator = util.CutoutGenerator(generator, times, sampling_rate=sampling_rate)

        #(123, 20, 50, 3): (時間個數(0.5~25, 每0.2一個time),  測站數, 5 * 10個ensemble , 3)
        # Assume PGA output at index 2
        workers = 1
        if use_multiprocessing:
            workers = 10
            
        predictions, distances, pga_labels, distance_labels = model_list.predict_generator(cutout_generator, workers=workers, use_multiprocessing=use_multiprocessing)  # (時間, (trace(超過20測站的trace會有>1個trace), 20, 50, 3))
        
        pga_pred = predictions

        pga_pred = pga_pred.reshape((len(times), -1) + pga_pred.shape[2:])  
        #(時間個數, 10個ensemble * 測站數, 5 , 3)
        pga_pred = pga_pred[:, :len(generator.pga[0])]  # Remove padding stations

        pga_times_pre = np.zeros((pga_pred.shape[1], thresholds.shape[0], alpha.shape[0]), dtype=int)

        # for j, log_level in enumerate(thresholds / 9.81):
        # for j, log_level in enumerate(np.log10((thresholds / 9.81)*10)):
        for j, log_level in enumerate(np.log10(thresholds*10)):
            prob = np.sum(
                pga_pred[:, :, :, 0] * (1 - norm.cdf((log_level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
                axis=-1)
            prob = prob.reshape(prob.shape + (1,))
            exceedance = prob > alpha  # Shape: times, stations, 1
            exceedance = np.pad(exceedance, ((1, 0), (0, 0), (0, 0)), mode='constant')
            pga_times_pre[:, j] = np.argmax(exceedance, axis=0)

        pga_times_pre -= 1
        pga_times_pred = np.zeros_like(pga_times_pre, dtype=float)
        pga_times_pred[pga_times_pre == -1] = np.nan
        pga_times_pred[pga_times_pre > -1] = times[pga_times_pre[pga_times_pre > -1]]

        g_event = g_data[str(event[event_key])]
        #print('g_event',g_event)
        pga_times_true_pre = g_event['pga_times'][()]
        

        pga_times_true = np.zeros_like(pga_times_true_pre, dtype=float)
        pga_times_true[pga_times_true_pre == 0] = np.nan
        pga_times_true[pga_times_true_pre != 0] = (pga_times_true_pre[pga_times_true_pre != 0]) / sampling_rate - time_before  #在座資料集的時候有 + time_before

        coords = g_event['coords'][()]
        coords_event = event[coord_keys]

        dist = np.zeros(coords.shape[0])
        for j, station_coords in enumerate(coords):
            dist[j] = geodesic(station_coords[:2], coords_event[:2]).km
        dist = np.sqrt(dist ** 2 + coords_event[2] ** 2)  # 測站距離"震央"(非地底下的震源)距離 Epi- to hypocentral distance

        full_predictions += [(pga_times_pred, pga_times_true, dist)]

    return full_predictions
    

def predict_at_time(model, time, data, event_metadata, batch_size, config, 
                    sampling_rate=100,pga=False,
                    use_multiprocessing=True, no_event_token=False, dataset_id=None):
    generator = generator_from_config(config, data, event_metadata, time,  batch_size, sampling_rate, dataset_id=dataset_id)

    workers = 1
    if use_multiprocessing:
        workers = 10
    predictions, distances, pga_labels, distance_labels = model.predict_generator(generator, workers=workers, use_multiprocessing=use_multiprocessing) # (2074, 249, 5, 3)

    pga_pred = []
    pga_pred_idx = 2 - 2 * no_event_token
    for i, (start, end) in enumerate(zip(generator.reverse_index[:-1], generator.reverse_index[1:])):
        sample_pga_pred = predictions[pga_pred_idx][start:end].reshape((-1,) + predictions[pga_pred_idx].shape[-2:])
        sample_pga_pred = sample_pga_pred[:len(generator.pga[i])]
        pga_pred += [sample_pga_pred]

    return predictions, distances, pga_labels, distance_labels
    
    
def generate_true_pred_plot(pred_values, true_values, time, path, suffix=''):
    if suffix:
        suffix += '_'

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    if suffix == 'pga_':
        _, cbar = plots.true_predicted(true_values, pred_values, agg='mean', quantile=True, ax=ax)
        cax = fig.colorbar(cbar)
        cax.set_label('Quantile')
    elif suffix == 'distance_':
        _, cbar = plots.true_predicted(true_values, pred_values, agg='point', quantile=False, ax=ax)
    fig.savefig(os.path.join(path, f'truepred_{suffix}{time}.png'), bbox_inches='tight')
    plt.close(fig)


def generate_calibration_plot(pred_values, true_values, time, path, suffix=''):
    if suffix:
        suffix += '_'
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    plots.calibration_plot(pred_values, true_values, ax=ax)
    ax.set_xlabel('<-- Overestimate       Underestimate -->')
    fig.savefig(os.path.join(path, f'quantiles_{suffix}{time}.png'), bbox_inches='tight')
    plt.close(fig)

def distance_label_plot(distance_label, distance_label_log, path):
    # 繪製散佈圖
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(distance_label, distance_label_log, zorder=2)

    # 設定圖表標題與軸標籤
    plt.title('distance')
    plt.xlabel('distance_label')
    plt.ylabel('distance_label_log')
    plt.savefig(os.path.join(path, f'distance_label_1000.png'), bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--weight_file', type=str)  # If unset use latest model
    parser.add_argument('--times', type=str, default='0.5,1,2,4,8,16,25')  # Has only performance implications
    parser.add_argument('--max_stations', type=int)  # Overwrite max stations value from config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val', action='store_true')  # Evaluate on val set
    parser.add_argument('--n_pga_targets', type=int)  # Overwrite number of PGA targets
    parser.add_argument('--head_times', action='store_true')  # Evaluate warning times
    parser.add_argument('--blind_time', type=float, default=0.5)  # Time of first evaluation after first P arrival
    parser.add_argument('--alpha', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')  # 機率大於alpha以上就會發布警報 Probability thresholds alpha
    parser.add_argument('--additional_data', type=str)  # Additional data set to use for evaluation
    parser.add_argument('--dataset_id', type=int)  # ID of dataset to evaluate on, in case of joint training
    parser.add_argument('--wait_file', type=str)  # Wait for this file to exist before starting evaluation
    parser.add_argument('--ensemble_member', action='store_true')  # Task to evaluate is an ensemble member
                                                                   # (not the full ensembel)
    parser.add_argument('--loss_limit', type=float) # In ensemble model, discard members with loss above this limit
    # A combination of tensorflow multiprocessing for generators and pandas dataframes causes the code to deadlock
    # sometimes. This flag provides a workaround.
    parser.add_argument('--no_multiprocessing', action='store_true')
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    args = parser.parse_args()

    if args.wait_file is not None:
        util.wait_for_file(args.wait_file)

    times = [float(x) for x in args.times.split(',')]

    config = json.load(open(os.path.join(args.experiment_path, 'config.json'), 'r'))
    training_params = config['training_params']

    device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")


    if (args.dataset_id is None) and (isinstance(training_params['data_path'], list) and
                                      len(training_params['data_path']) > 1):
        raise ValueError('dataset_id needs to be set for experiments with multiple input data sets.')
    if (args.dataset_id is not None) and not (isinstance(training_params['data_path'], list) and
                                              len(training_params['data_path']) > 1):
        raise ValueError('dataset_id may only be set for experiments with multiple input data sets.')

    if args.dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[args.dataset_id]
        data_path = training_params['data_path'][args.dataset_id]
        n_datasets = len(training_params['data_path'])
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]
        data_path = training_params['data_path']
        n_datasets = 1
        
    batch_size = generator_params['batch_size']
    key = generator_params.get('key', 'MA')
    pos_offset = generator_params.get('pos_offset', (-21, -69))
    pga_key = generator_params.get('pga_key', 'pga')

    if args.blind_time != 0.5:
        suffix = f'_blind{args.blind_time:.1f}'
    else:
        suffix = ''

    if args.dataset_id is not None:
        suffix += f'_{args.dataset_id}'

    # 台灣、日本，要改資料集
    if args.val:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'val')
        training_params['data_path'] = training_params['val_data_path']
        test_set = False
    else:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'test')
        training_params['data_path'] = training_params['test_data_path']
        test_set = True

    if not os.path.isdir(os.path.join(args.experiment_path, f'evaluation{suffix}')):
        os.mkdir(os.path.join(args.experiment_path, f'evaluation{suffix}'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    if args.test_run:
        limit = 10
    else:
        limit = None

    shuffle_train_dev = generator_params.get('shuffle_train_dev', False)
    custom_split = generator_params.get('custom_split', None)
    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    min_mag = generator_params.get('min_mag', None)
    mag_key = generator_params.get('key', 'MA')
    event_metadata, data, metadata = loader.load_events(training_params['data_path'],
                                                        limit=limit,
                                                        shuffle_train_dev=shuffle_train_dev,
                                                        custom_split=custom_split,
                                                        min_mag=min_mag,
                                                        mag_key=mag_key,
                                                        overwrite_sampling_rate=overwrite_sampling_rate)

    if args.additional_data:
        print('Loading additional data')
        event_metadata_add, data_add, _ = loader.load_events(args.additional_data,
                                                             parts=(True, True, True),
                                                             min_mag=min_mag,
                                                             mag_key=mag_key,
                                                             overwrite_sampling_rate=overwrite_sampling_rate)
        event_metadata = pd.concat([event_metadata, event_metadata_add])
        for t_key in data.keys():
            if t_key in data_add:
                data[t_key] += data_add[t_key]

    if pga_key in data:
        pga_true = data[pga_key]
    else:
        pga_true = None

    if 'max_stations' not in config['model_params']:
        config['model_params']['max_stations'] = data['waveforms'].shape[1]
    if args.max_stations is not None:
        config['model_params']['max_stations'] = args.max_stations

    if args.n_pga_targets is not None:
        if config['model_params'].get('n_pga_targets', 0) > 0:
            print('Overwriting number of PGA targets')
            config['model_params']['n_pga_targets'] = args.n_pga_targets
        else:
            print('PGA flag is set, but model does not support PGA')

    
    ensemble = config.get('ensemble', 1)
    print(ensemble)
    if ensemble > 1 and not args.ensemble_member:
        experiment_path = args.experiment_path
        model_list = EnsembleEvaluateModel(config, experiment_path, loss_limit=args.loss_limit, batch_size=batch_size, device=device)

    else:  #沒有ensemble，一般不走這
        if 'n_datasets' in config['model_params']:
            del config['model_params']['n_datasets']
        model_list = models.build_transformer_model(**config['model_params'], trace_length=data['waveforms'][0].shape[1], n_datasets=n_datasets).to(device)

        if args.weight_file is not None:
            weight_file = os.path.join(args.experiment_path, args.weight_file)
        else:
            weight_file = sorted([x for x in os.listdir(args.experiment_path) if x[:11] == 'checkpoint-'])[-1]
            weight_file = os.path.join(args.experiment_path, weight_file)
        
        print(weight_file)
        model_list.load_state_dict(torch.load(weight_file)['model_weights'])

    mag_stats = []
    loc_stats = []
    pga_stats = []
    mag_pred_full = []
    loc_pred_full = []
    pga_pred_full = []
    
    print(f'Start to evaluate regression performance!!!')
    for time in times:
        print(f'Times: {time}s')
        pga_pred, distances , pga_labels, distance_labels = predict_at_time(model_list, time, data, event_metadata,
                               config=config,
                               batch_size=batch_size,
                               sampling_rate=metadata['sampling_rate'],
                               dataset_id=args.dataset_id)

        pga_pred_full += [pga_pred] #(10197, 15, 5, 3)
        pga_pred_reshaped = np.concatenate(pga_pred, axis=0) #(152955, 5, 3)
        pga_true_reshaped = pga_labels.reshape(-1) #(152955,)
        pga_true_reshaped = np.log10(pga_true_reshaped * 10 + 1e-8)
        mask = ~np.logical_or(np.isnan(pga_true_reshaped), np.isinf(pga_true_reshaped))  #(trace數,) 確保pred/true沒有nan或是inf的數值 (137535,)
        pga_true_reshaped = pga_true_reshaped[mask]  #有nan/inf的話直接刪除
        pga_pred_reshaped = pga_pred_reshaped[mask]  #有nan/inf的話直接刪除 #(152955, 5, 3)
        
        mask = pga_true_reshaped > -5
        pga_true_reshaped = pga_true_reshaped[mask]
        pga_pred_reshaped = pga_pred_reshaped[mask]
        plot_save_path = os.path.join(output_dir, 'pred_plot')
        if not os.path.isdir(plot_save_path):
            os.mkdir(plot_save_path)
        generate_true_pred_plot(pga_pred_reshaped, pga_true_reshaped, time, plot_save_path, suffix='pga')

        mask = distance_labels <= 1000 
        distances = distances[mask]
        distance_labels = distance_labels[mask]
        generate_true_pred_plot(distances, distance_labels, time, plot_save_path, suffix='distance')
    
    results = {'times': times,
               'pga_stats': np.array(pga_stats).tolist()}

    with open(os.path.join(output_dir, 'stats.json'), 'w') as stats_file:
        json.dump(results, stats_file, indent=4)

    if args.head_times:
        print(f'Start to evaluate alert performance!!!')
        times_pga = np.arange(args.blind_time, 25, 0.2)
        alpha = [float(x) for x in args.alpha.split(',')]
        warning_time_information = calculate_warning_times(config, model_list, data, event_metadata,
                                                           times=times_pga,
                                                           alpha=alpha,
                                                           batch_size=batch_size,
                                                           use_multiprocessing=not args.no_multiprocessing,
                                                           dataset_id=args.dataset_id, device=device)
    else:
        warning_time_information = None
        alpha = None


    with open(os.path.join(output_dir, 'head_times_predictions.pkl'), 'wb') as pred_file:
        pickle.dump((times, mag_pred_full, loc_pred_full, pga_pred_full, warning_time_information, alpha), pred_file)
        