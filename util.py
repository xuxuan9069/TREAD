import glob
import h5py
import numpy as np
import pandas as pd
import obspy
import os
from obspy import UTCDateTime
import warnings
import time
from tqdm import tqdm
from scipy.stats import norm
import torch
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt
D2KM = 111.19492664455874

def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)

#keras.utils.Sequence 批次載入input data

def detect_location_keys(columns):
    candidates = [['LAT', 'Latitude(°)', 'Latitude', 'source_latitude_deg'],  
                  ['LON', 'Longitude(°)', 'Longitude', 'source_longitude_deg'],
                  ['DEPTH', 'JMA_Depth(km)', 'Depth(km)', 'Depth/Km', 'source_depth_km']]

    coord_keys = []
    for keyset in candidates:
        for key in keyset:
            if key in columns:
                coord_keys += [key]
                break

    if len(coord_keys) != len(candidates):
        raise ValueError('Unknown location key format')

    return coord_keys

class PreloadedEventGenerator(Dataset):
    def __init__(self, data, event_metadata, all_station=False,key='MA', batch_size=32, cutout=None,
                     sliding_window=False, windowlen=3000, shuffle=True,
                     coords_target=True, oversample=1, pos_offset=(-21, -69),
                     label_smoothing=False, station_blinding=False, magnitude_resampling=3,
                     pga_targets=None, adjust_mean=True, transform_target_only=False,
                     max_stations=None, trigger_based=None, min_upsample_magnitude=2,
                     disable_station_foreshadowing=False, selection_skew=None, pga_from_inactive=False,
                     integrate=False, sampling_rate=100.,
                     select_first=False, fake_borehole=False, scale_metadata=True, pga_key='pga',
                     pga_mode=False, p_pick_limit=5000, coord_keys=None, upsample_high_station_events=None,
                     no_event_token=False, pga_selection_skew=None, **kwargs):
        if kwargs:
            #print(f'Unused parameters: {", ".join(kwargs.keys())}')
            self.all_station = all_station
            self.batch_size = batch_size 
            self.shuffle = shuffle
            self.waveforms = data['waveforms']  #list長度為n = 有n個事件數[(shape(9,3000,6)), (shape(12,3000,6)), (shape(11,3000,6)), ...... , (shape(4,3000,6)), (shape(7,3000,6))]
            self.metadata = data['coords']
            self.event_metadata = event_metadata
            
            if pga_key in data:
                self.pga = data[pga_key]
            else:
                print('Found no PGA values')
                self.pga = [np.zeros(x.shape[0]) for x in self.waveforms]
            self.key = key
            self.cutout = cutout
            self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
            self.windowlen = windowlen  # Length of window for sliding window
            self.coords_target = coords_target
            self.oversample = oversample  # 設1
            self.pos_offset = pos_offset
            self.label_smoothing = label_smoothing
            self.station_blinding = station_blinding
            self.magnitude_resampling = magnitude_resampling
            self.pga_targets = pga_targets
            self.adjust_mean = adjust_mean
            self.transform_target_only = transform_target_only
            if max_stations is None:
                max_stations = self.waveforms.shape[1]
            self.max_stations = max_stations
            self.trigger_based = trigger_based
            self.disable_station_foreshadowing = disable_station_foreshadowing
            self.selection_skew = selection_skew
            self.pga_from_inactive = pga_from_inactive
            self.pga_selection_skew = pga_selection_skew
            self.integrate = integrate
            self.sampling_rate = sampling_rate
            self.select_first = select_first
            self.fake_borehole = fake_borehole
            self.scale_metadata = scale_metadata
            self.upsample_high_station_events = upsample_high_station_events
            self.no_event_token = no_event_token

            if 'p_picks' in data:
                self.triggers = data['p_picks']  #list，內含 n個事件的測站pick數值
            else:
                print('Found no picks')
                self.triggers = [np.zeros(x.shape[0]) for x in self.waveforms]

            for i in range(len(self.triggers)):
                self.triggers[i] -= np.min(self.triggers[i]) - 500
            # Extend samples to include all pga targets in each epoch
            # PGA mode is only for evaluation, as it adds zero padding to the input/pga target!
            self.pga_mode = pga_mode
            self.p_pick_limit = p_pick_limit

            self.base_indexes = np.arange(len(self.waveforms))
            self.reverse_index = None
            if magnitude_resampling > 1:
                magnitude = self.event_metadata[key].values
                for i in np.arange(min_upsample_magnitude, 9):
                    ind = np.where(np.logical_and(i < magnitude, magnitude <= i + 1))[0]
                    self.base_indexes = np.concatenate(
                        (self.base_indexes, np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1))))

            if self.upsample_high_station_events is not None:
                new_indexes = []
                for ind in self.base_indexes:
                    n_stations = self.waveforms[ind].shape[0]
                    new_indexes += [ind for _ in range(n_stations // self.upsample_high_station_events + 1)]
                self.base_indexes = np.array(new_indexes)
            
            if pga_mode:  #evaluate.py在用的
                new_base_indexes = []
                self.reverse_index = []
                c = 0
                for idx in self.base_indexes:  #取的測站編號，每個idx都是一個batch編號  (batch=1就是指反覆所有trace數量的次數)
                    # This produces an issue if there are 0 pga targets for an event.
                    # As all input stations are always pga targets as well, this should not occur.

                    # self.pga[idx] : 該idx batch有偵測到的測站的所有最大pga  [-1.62259208 -1.59848291 ....... -1.32014115 -1.68538696]
                    num_samples = (len(self.pga[idx]) - 1) // pga_targets + 1  #有偵測到波形的測站數//20
                    new_base_indexes += [(idx, i) for i in range(num_samples)]
                    self.reverse_index += [c]
                    c += num_samples
                self.reverse_index += [c]
                self.base_indexes = new_base_indexes

            self.indexes = np.arange(len(self.waveforms))  #事件個數 

            if coord_keys is None: #回傳 event_metadata dict名稱 ['Latitude(°)', 'Longitude(°)', 'Depth(km)']
                self.coord_keys = detect_location_keys(event_metadata.columns)
            else:
                self.coord_keys = coord_keys
            
            self.indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)
            if self.shuffle:
                #randidx = torch.randperm(self.indexes.nelement())
                #self.indexes = self.indexes[randidx]
                np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index): #index=地震事件數
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        true_batch_size = len(indexes)  #一個batch的長度
        if self.pga_mode:
            # indexes = self.base_indexes
            # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), ……, (1350, 0), (1350, 1), (1350, 2), (1351, 0), (1351, 1)]
	        # (trace的編號, 某trace預測第幾輪的編號)
            pga_indexes = [x[1] for x in indexes]  #某trace預測第幾輪的編號
            indexes = [x[0] for x in indexes]  #trace的編號

        waveforms = np.zeros((true_batch_size, self.max_stations) + self.waveforms[0].shape[1:])  #(batch size, 測站數(25), 波形長度, n軸加速度(6))
        true_max_stations_in_batch = max(max([self.metadata[idx].shape[0] for idx in indexes]), self.max_stations) #int
        metadata = np.zeros((true_batch_size, true_max_stations_in_batch) + self.metadata[0].shape[1:])   #(batch size, 測站數(25), 座標維度(4)) 
        pga = np.zeros((true_batch_size, true_max_stations_in_batch))   #(batch size, 測站數(25),)
        full_p_picks = np.zeros((true_batch_size, true_max_stations_in_batch))#(batch size, 測站數(25),)
        p_picks = np.zeros((true_batch_size, self.max_stations))#(batch size, 測站數(25),)
        reverse_selections = []

        # Find list of IDs
        for i, idx in enumerate(indexes):
            
            if len(self.waveforms[idx]) <= self.max_stations:
                waveforms[i, :len(self.waveforms[idx])] = self.waveforms[idx]  #1352個
                metadata[i, :len(self.metadata[idx])] = self.metadata[idx]
                pga[i, :len(self.pga[idx])] = self.pga[idx]
                p_picks[i, :len(self.triggers[idx])] = self.triggers[idx]
                reverse_selections += [[]]
            else:
                #============================================================================沒設定此參數
                if self.selection_skew is None:  #設1000 所以不會進這行
                    selection = np.arange(0, len(self.waveforms[idx]))
                    #randidx = torch.randperm(selection.nelement())
                    #selection = selection[randidx]
                    np.random.shuffle(selection)
                #============================================================================
                else:
                    tmp_p_picks = self.triggers[idx].copy()
                    mask = np.logical_or(tmp_p_picks <= 0, tmp_p_picks > self.p_pick_limit) #logical_and->logical_or
                    tmp_p_picks[mask] = min(np.max(tmp_p_picks), self.p_pick_limit)
                    coeffs = np.exp(-tmp_p_picks / self.selection_skew)
                    coeffs *= np.random.random(coeffs.shape)
                    coeffs[self.triggers[idx] == 0] = 0
                    coeffs[self.triggers[idx] > self.waveforms[0].shape[1]] = 0
                    selection = np.argsort(-coeffs)  #隨機排序，但p_pick越快的越有機會被選中

                #============================================================================沒設定此參數
                if self.select_first:
                    selection = np.argsort(self.triggers[idx])
                #============================================================================

                metadata[i, :len(selection)] = self.metadata[idx][selection] #self.metadata[idx]看n次事件裡第idx次事件的所有測站座標，self.metadata[idx][selection]看該事件某25個測站的座標
                pga[i, :len(selection)] = self.pga[idx][selection]
                full_p_picks[i, :len(selection)] = self.triggers[idx][selection]

                tmp_reverse_selection = [0 for _ in selection]
                for j, s in enumerate(selection):
                    tmp_reverse_selection[s] = j
                reverse_selections += [tmp_reverse_selection]  #把前25個測站的順序顛倒

                selection = selection[:self.max_stations]
                waveforms[i] = self.waveforms[idx][selection]
                p_picks[i] = self.triggers[idx][selection]

        magnitude = self.event_metadata.iloc[indexes][self.key].values.copy()

        target = None
        if self.coords_target:
            target = self.event_metadata.iloc[indexes][self.coord_keys].values
        org_waveform_length = waveforms.shape[2]
        if self.cutout:
            if self.sliding_window:
                windowlen = self.windowlen
                window_end = np.random.randint(max(windowlen, self.cutout[0]), min(waveforms.shape[2], self.cutout[1]) + 1)
                waveforms = waveforms[:, :, window_end - windowlen: window_end]

                cutout = window_end
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms, axis=2, keepdims=True)
            else:
                cutout = np.random.randint(*self.cutout)  #cutout = 400, 3000
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms[:, :, :cutout+1], axis=2, keepdims=True) #把隨機生成的時間點以前的波形再做一次校正 (應該是不會差到太多)
                waveforms[:, :, cutout:] = 0 #隨機生成一個400~3000的數字，該數字以後的波形全部砍掉
        else:
            cutout = waveforms.shape[2]

        # plt.plot(waveforms[0,0,:,0])
        # plt.savefig(f'./img/{cutout}')
        # plt.clf()

        if self.trigger_based:
            # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
            p_picks[p_picks <= 0] = org_waveform_length  # <=0的pick全部變成3000，因為等一下要由小排到大
            waveforms[cutout < p_picks, :, :] = 0


        if self.integrate:
            waveforms = np.cumsum(waveforms, axis=2) / self.sampling_rate

        # Reshape magnitude to match dimension number of MDN output
        magnitude = np.expand_dims(np.expand_dims(magnitude, axis=-1), axis=-1)
        # Center location on mean of locations
        
        metadata_origin = metadata.copy()
        # if self.coords_target:
        #     metadata, target = self.location_transformation(metadata, target)
        # else:
        #     metadata = self.location_transformation(metadata)

        if self.label_smoothing:
            magnitude += (magnitude > 4) * np.random.randn(magnitude.shape[0]).reshape(magnitude.shape) * (
                    magnitude - 4) * 0.05

        if not self.pga_from_inactive and not self.pga_mode:
            metadata = metadata[:, :self.max_stations]
            pga = pga[:, :self.max_stations]

        if self.pga_targets:
            pga_values = np.zeros(
                (true_batch_size, self.pga_targets))
            pga_targets = np.zeros((true_batch_size, self.pga_targets, 3))
            pga_targets_origin = np.zeros((true_batch_size, self.pga_targets, 3))
            #1352個 trace 中的第 "pga_indexes" 個trace
            if self.pga_mode:
                for i in range(waveforms.shape[0]):   #for迴圈所有batch
                    #該pga_index trace中的第i個測站
                    pga_index = pga_indexes[i]
                    if len(reverse_selections[i]) > 0:
                        sorted_pga = pga[i, reverse_selections[i]]
                        sorted_metadata = metadata[i, reverse_selections[i]]
                        sorted_metadata_origin = metadata_origin[i, reverse_selections[i]]
                    else:
                        #第i個測站的數值
                        sorted_pga = pga[i]  
                        sorted_metadata = metadata[i]
                        sorted_metadata_origin = metadata_origin[i]
                    #
                    pga_values_pre = sorted_pga[pga_index * self.pga_targets:(pga_index + 1) * self.pga_targets]  #第pga_index個trace的所有測站數值
                    pga_values[i, :len(pga_values_pre)] = pga_values_pre 
                    pga_targets_pre = sorted_metadata[pga_index * self.pga_targets:(pga_index + 1) * self.pga_targets, :]
                    pga_targets_pre_origin = sorted_metadata_origin[pga_index * self.pga_targets:(pga_index + 1) * self.pga_targets, :]
                    if pga_targets_pre.shape[-1] == 4:
                        pga_targets_pre = pga_targets_pre[:, (0, 1, 3)]
                        pga_targets_pre_origin = pga_targets_pre_origin[:, (0, 1, 3)]
                    pga_targets[i, :len(pga_targets_pre), :] = pga_targets_pre #還是挑前20個測站數值(????)
                    pga_targets_origin[i, :len(pga_targets_pre_origin), :] = pga_targets_pre_origin #還是挑前20個測站數值(????)
            else:
                pga[np.logical_or(np.isnan(pga), np.isinf(pga))] = 0  # Ensure only legal PGA values are selected
                for i in range(waveforms.shape[0]):   #for迴圈所有batch
                    active = np.where(pga[i] != 0)[0]
                    if len(active) == 0:
                        raise ValueError(f'Found event without PGA idx={indexes[i]}')
                    while len(active) < self.pga_targets:
                        active = np.repeat(active, 2)
                    if self.pga_selection_skew is not None:
                        active_p_picks = full_p_picks[i, active]
                        mask = np.logical_or(active_p_picks <= 0, active_p_picks > self.p_pick_limit) #logical_and->logical_or
                        active_p_picks[mask] = min(np.max(active_p_picks), self.p_pick_limit)
                        coeffs = np.exp(-active_p_picks / self.pga_selection_skew)
                        coeffs *= np.random.random(coeffs.shape)
                        active = active[np.argsort(-coeffs)]
                    else:
                        #randidx = torch.randperm(active.nelement())
                        #active = active[randidx]
                        np.random.shuffle(active)

                    samples = active[:self.pga_targets]
                    if metadata.shape[-1] == 3:
                        pga_targets[i] = metadata[i, samples, :]
                    else:
                        full_targets = metadata[i, samples]
                        pga_targets[i] = full_targets[:, (0, 1, 3)]
                    pga_values[i] = pga[i, samples]
            # Last two dimensions to match shape for keras loss
            pga_values = pga_values.reshape((true_batch_size, self.pga_targets, 1, 1))

        metadata = metadata[:, :self.max_stations]

        if self.station_blinding:
            mask = np.zeros(waveforms.shape[:2], dtype=bool)

            for i in range(waveforms.shape[0]):
                active = np.where((waveforms[i] != 0).any(axis=(1, 2)))[0]
                if len(active) == 0:
                    active = np.zeros(1, dtype=int)
                blind_length = np.random.randint(0, len(active))
                #randidx = torch.randperm(active.nelement())
                #active = active[randidx]
                np.random.shuffle(active)
                blind = active[:blind_length]
                mask[i, blind] = True

            waveforms[mask] = 0
            metadata[mask] = 0

        # To avoid that stations without a trigger are masked, we can set a value in the waveforms to non-zero.
        # Thereby we keep the information that the station did not trigger yet.
        # On the other hand this might leak information that a station is still going to trigger
        stations_without_trigger = (metadata != 0).any(axis=2) & (waveforms == 0).all(axis=(2, 3))
        if self.disable_station_foreshadowing:
            metadata[stations_without_trigger] = 0
        else:
            waveforms[stations_without_trigger, 0, 0] += 1e-9

        # Avoid completely zero events, leading to NaN values in energy loss
        mask = np.logical_and((metadata == 0).all(axis=(1, 2)), (waveforms == 0).all(axis=(1, 2, 3)))
        waveforms[mask, 0, 0, 0] = 1e-9
        metadata[mask, 0, 0] = 1e-9

        if self.fake_borehole and waveforms.shape[3] == 3:
            waveforms = np.concatenate([np.zeros_like(waveforms), waveforms], axis=3)
            metadata_new = np.zeros(metadata.shape[:-1] + (4,))
            metadata_new[:, :, 0] = metadata[:, :, 0]
            metadata_new[:, :, 1] = metadata[:, :, 1]
            metadata_new[:, :, 3] = metadata[:, :, 2]
            metadata = metadata_new
        
        waveforms = torch.from_numpy(waveforms.astype('float32'))
        metadata = torch.from_numpy(metadata.astype('float32'))
        inputs = [waveforms, metadata]
        outputs = []

        #=================================================== TEAM-LM 不一定需要
        if not self.no_event_token:
            outputs += [magnitude]

            if self.coords_target:
                target = np.expand_dims(target, axis=-1)
                target = torch.from_numpy(target.astype('float32'))
                outputs += [target]
        #===================================================
        if self.all_station: #evaluate_250.py 沒進此條件
            pga_targets_by_position = np.zeros(pga_targets.shape)
            #沒有被記錄到的測站也要一起算分數
            stations_table = json.load(open('./stations.json', 'r'))
            
            for station_key in stations_table.keys():
                station = station_key.split(',')
                position = stations_table[station_key] 
                pga_targets_by_position[0][position] = np.array([[[float(station[0]),float(station[1]),float(station[2])]]])
                
            pga_targets = pga_targets_by_position #因為註解掉location_transformation的部分，所以要把pga_targets設定好
            
            # if self.coords_target:
            #     pga_targets, _ = self.location_transformation(pga_targets_by_position, target)
            # else:
            #     pga_targets = self.location_transformation(pga_targets_by_position)
                    
                    
        if self.pga_targets:
            inputs = inputs + [torch.from_numpy(pga_targets.astype('float32'))]
            outputs = [torch.from_numpy(pga_values.astype('float32'))]
        return inputs, outputs

    # def location_transformation(self, metadata, target=None):
    #     transform_target_only = self.transform_target_only
    #     metadata = metadata.copy()

    #     metadata_old = metadata
    #     metadata = metadata.copy()
    #     mask = (metadata == 0).all(axis=2)
    #     if target is not None:
    #         target[:, 0] -= self.pos_offset[0]
    #         target[:, 1] -= self.pos_offset[1]
    #     metadata[:, :, 0] -= self.pos_offset[0]
    #     metadata[:, :, 1] -= self.pos_offset[1]

    #     # Coordinates to kilometers (assuming a flat earth, which is okay close to equator)
    #     if self.scale_metadata:
    #         metadata[:, :, :2] *= D2KM
    #     if target is not None:
    #         target[:, :2] *= D2KM

    #     metadata[mask] = 0

    #     if self.scale_metadata:
    #         metadata /= 100
    #     if target is not None:
    #         target /= 100

    #     if transform_target_only:
    #         metadata = metadata_old

    #     if target is None:
    #         return metadata
    #     else:
    #         return metadata, target


def generator_from_config(config, data, event_metadata, time, batch_size=64, sampling_rate=100, dataset_id=None):
    training_params = config['training_params']

    if dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[dataset_id]
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    noise_seconds = generator_params.get('noise_seconds', 5)
    cutout = int(sampling_rate * (noise_seconds + time))
    cutout = (cutout, cutout + 1)

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    max_stations = config['model_params']['max_stations']
    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
    generator_params['upsample_high_station_events'] = None
    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')
    generator_params['translate'] = False
    '''
    print(len(data['waveforms']))
    total_trace_count = 0
    for i in data['waveforms']:
        total_trace_count += len(i)
    print('total_trace_count: ', total_trace_count)
    '''

    generator = PreloadedEventGenerator(data=data,
                                        event_metadata=event_metadata,
                                        coords_target=True,
                                        cutout=cutout,
                                        pga_targets=n_pga_targets,
                                        max_stations=max_stations,
                                        sampling_rate=sampling_rate,
                                        select_first=True,
                                        shuffle=False,
                                        pga_mode=True,
                                        **generator_params)
    #if dataset_id is not None and config['model_params'].get('dataset_bias', False):
    #    generator = JointGenerator([generator], shuffle=False, dataset_id=True, fake_id=dataset_id)
    
    return generator


class CutoutGenerator(Dataset):
    def __init__(self, generator, times, sampling_rate):
        self.generator = generator
        self.times = times
        self.sampling_rate = sampling_rate
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return len(self.generator) * len(self.times)

    def __getitem__(self, index):
        time, batch_id = self.indexes[index]
        cutout = int(self.sampling_rate * (time + 5))
        self.generator.cutout = (cutout, cutout + 1)
        return self.generator[batch_id]

    def on_epoch_end(self):
        self.indexes = []
        for time in self.times:
            self.indexes += [(time, i) for i in range(len(self.generator))]


'''
class JointGenerator(Dataset):  #沒有用到
        def __init__(self, generators=(), shuffle=True, dataset_id=False, fake_id=None):
            assert len(generators)
            self.generators = generators
            self.indexes = None
            self.shuffle = shuffle
            self.dataset_id = dataset_id
            self.fake_id = fake_id

        def __len__(self):
            return sum(len(generator) for generator in self.generators)

        def __getitem__(self, index):
            generator_id, batch_id = self.indexes[index]
            batch_inp, batch_out = self.generators[generator_id][batch_id]
            if self.dataset_id:
                if self.fake_id is None:
                    dataset_id = np.ones((batch_inp[0].shape[0], 1)) * generator_id
                else:
                    dataset_id = np.ones((batch_inp[0].shape[0], 1)) * self.fake_id
                batch_inp += [dataset_id]
            return batch_inp, batch_out

        def on_epoch_end(self):
            self.indexes = []
            for i, generator in enumerate(self.generators):
                self.indexes += [(i, j) for j in range(len(generator))]
            if self.shuffle:
                np.random.shuffle(self.indexes)
'''

