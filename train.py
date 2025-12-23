import numpy as np
import h5py
import torch
import torch.nn as nn
from tqdm import tqdm 
import os
import pickle
import argparse
import json
import time
from scipy.stats import norm

import util
import loader

import models

import logging
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from torch.backends import cudnn
cudnn.benchmark = True # fast  training

def seed_np_pt(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed) #CPU seed
    torch.cuda.manual_seed(seed) #GPU seed
    #torch.cuda.manual_seed_all(seed) #多張GPU seed


def transfer_weights(model, weights_path, ensemble_load=False, wait_for_load=False, ens_id=None, sleeptime=600):
    
    print("weights_path")
    if ensemble_load:
        weights_path = os.path.join(weights_path, f'{ens_id}')

    # If weight file does not exists, wait until it exists. Intended for ensembles. Warning: Can deadlock program.
    if wait_for_load:
        if os.path.isfile(weights_path):
            target_object = weights_path
        else:
            target_object = os.path.join(weights_path, 'train.log')

        while not os.path.exists(target_object):
            print(f'File {target_object} for weight transfer missing. Sleeping for {sleeptime} seconds.')
            time.sleep(sleeptime)

    if os.path.isdir(weights_path):
        last_weight = sorted([x for x in os.listdir(weights_path) if x[:11] == 'checkpoint_'])[-1] 
        weights_path = os.path.join(weights_path, last_weight)
        
    print(weights_path)
    own_state = model.state_dict()
    state_dict = torch.load(weights_path)['model_weights']
    
    for name, param in state_dict.items():
        if name not in own_state.keys():
            print(f"{name} is not load weight")
            continue
        else:
            own_state[name].copy_(param)
            
    full_model.load_state_dict(own_state)
    return full_model
    # td = None
    # conv1d = None
    # td_name = None
    # conv1d_name = None
    # for layer in model.layers:
    #     if layer.name.find('time_distributed') != -1:
    #         td = layer.layer
    #         td_name = layer.name
    #         break
    # for layer in td.layers:
    #     if layer.name.find('conv1d') != -1:
    #         conv1d = layer
    #         conv1d_name = layer.name
    #         break
    # model_borehole = conv1d.get_weights()[0].shape[1] == 64

    # if ensemble_load:
    #     weights_path = os.path.join(weights_path, f'{ens_id}')

    # # If weight file does not exists, wait until it exists. Intended for ensembles. Warning: Can deadlock program.
    # if wait_for_load:
    #     if os.path.isfile(weights_path):
    #         target_object = weights_path
    #     else:
    #         target_object = os.path.join(weights_path, 'hist.pkl')

    #     while not os.path.exists(target_object):
    #         print(f'File {target_object} for weight transfer missing. Sleeping for {sleeptime} seconds.')
    #         time.sleep(sleeptime)

    # if os.path.isdir(weights_path):
    #     last_weight = sorted([x for x in os.listdir(weights_path) if x[:6] == 'event-'])[-1]
    #     weights_path = os.path.join(weights_path, last_weight)

    # with h5py.File(weights_path, 'r') as weights:
    #     weights_borehole = weights[td_name][conv1d_name]['kernel:0'].shape[1] == 64
    #     weights_dict = generate_weights_dict(weights)
    # del_list = []
    # for weight in weights_dict:
    #     if weight[:9] == 'embedding':
    #         del_list += [weight]
    # for del_element in del_list:
    #     del weights_dict[del_element]
    # if model_borehole and not weights_borehole:
    #     # Take same weights for borehole as for top sensor and rescale
    #     combine_weights = np.concatenate([weights_dict[f'{conv1d_name}/kernel:0'], weights_dict[f'{conv1d_name}/kernel:0']], axis=1)
    #     combine_weights /= 2
    #     weights_dict[f'{conv1d_name}/kernel:0'] = combine_weights
    # if not model_borehole and weights_borehole:
    #     # Only take weights for the surface sensor and rescale
    #     combine_weights = weights_dict[f'{conv1d_name}/kernel:0'][:, :32, :]
    #     combine_weights *= 2
    #     weights_dict[f'{conv1d_name}/kernel:0'] = combine_weights
    # new_weights = []
    # transferred = 0
    # for i, weight in enumerate(model.weights):
    #     name = weight.name
    #     if name in weights_dict:
    #         new_weights += [weights_dict[name]]
    #         transferred += 1
    #     else:
    #         new_weights += [model.get_weights()[i]]
    # print(f'Transferred {transferred} of {len(model.weights)} weights')
    # model.set_weights(new_weights)



def gaussian_confusion_matrix(type, status, confusion_matrix, targets_pga=None, pred=None, thresholds=None, loop=None, pga_loss=None, distance_loss=None, total_loss=None, optimizer=None):
    if status == 'accumulate':
        pred_matrix = np.empty((pred.shape[0], len(thresholds)))
        targets_pga = np.reshape(targets_pga,(targets_pga.shape[0],1))
        targets_pga = (targets_pga >= thresholds).astype(int)
        targets_pga = np.sum(targets_pga, axis=1)
        for j, level in enumerate(thresholds):
            prob = np.sum(
                pred[:, :, 0] * (1 - norm.cdf((level - pred[:, :, 1]) / pred[:, :, 2])),
                axis=-1) #得到測站是log_level的機率
            exceedance = prob >= 0.4
            pred_matrix[:,j] = exceedance
        pred_matrix = pred_matrix.astype(int)
        pred_matrix = np.sum(pred_matrix, axis=1)
        for idx in range(len(pred_matrix)):
            confusion_matrix[pred_matrix[idx]][targets_pga[idx]] += 1
            
    elif status == 'write_txt':
        print(confusion_matrix)
        with open(os.path.join(training_params['weight_path'], '{}_confusion_matrix.txt'.format(type)), 'a+', encoding='utf-8') as f:
            f.write(str(epoch)+'\n')
            f.write(str(confusion_matrix)+'\n\n')
            f.write('pga_loss: {},  distance_loss: {},  total_loss:{},  lr: {}'.format((pga_loss / len(loop)), (distance_loss / len(loop)), (total_loss / len(loop)), optimizer.param_groups[0]['lr'])+'\n\n')


def distance_loss(pred_distance, station_coords, target_coords):
    """
    station_coords: (batch_size, num_stations, 3) - 測站的三維座標
    target_coords: (batch_size, num_targets, 3) - 目標的三維座標
    """
    R = 6371  # 地球半徑，單位：公里

    # 1. 將經緯度從度數轉為弧度
    stations_rad = torch.deg2rad(station_coords[..., :2])  # (batch_size, num_stations, 2)
    targets_rad = torch.deg2rad(target_coords[..., :2])    # (batch_size, num_targets, 2)

    # 2. 計算緯度與經度的差
    dlat = targets_rad.unsqueeze(1)[..., 0] - stations_rad.unsqueeze(2)[..., 0]  # Δφ
    dlon = targets_rad.unsqueeze(1)[..., 1] - stations_rad.unsqueeze(2)[..., 1]  # Δλ

    # 3. 使用 Haversine 公式計算水平距離
    a = torch.sin(dlat / 2) ** 2 + \
        torch.cos(stations_rad.unsqueeze(2)[..., 0]) * \
        torch.cos(targets_rad.unsqueeze(1)[..., 0]) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(a))
    horizontal_dist = R * c  # 單位：公里

    # 4. 計算垂直距離（深度差）
    depth_diff = target_coords.unsqueeze(1)[..., 2] - station_coords.unsqueeze(2)[..., 2]  # Δdepth
    depth_diff = depth_diff.abs()  # 取絕對值

    # 5. 合併水平距離與垂直距離
    real_dist_matrix = torch.sqrt(horizontal_dist ** 2 + depth_diff ** 2)  # shape: (batch_size, num_stations, num_targets)

    # 6. 將距離矩陣展平為 (batch_size, num_stations * num_targets)
    dist_vector = real_dist_matrix.view(real_dist_matrix.size(0), -1)
    
    # 建立 mask，標記 dist_vector 中 <= 1000 的值
    mask = dist_vector <= 1000
    # 只保留 mask 中為 True 的位置的值
    filtered_dist_vector = dist_vector[mask] ### models ###
    # filtered_dist_vector = torch.nan_to_num(torch.log10(dist_vector[mask]), nan=0.0, posinf=3.0, neginf=-3.0) #torch.log10(0)=-inf -> 0.0 ### models_v2 ###
    filtered_pred_distance = pred_distance[mask]
    
    mse_loss = nn.MSELoss()
    if filtered_dist_vector.numel() == 0:
        # print("所有距離都大於 1000，跳過此批次損失計算")
        distances_loss = torch.tensor(0.0, requires_grad=True)  # 或設定為 0
    else:
        distances_loss = mse_loss(filtered_pred_distance, filtered_dist_vector)
        # min_value = torch.min(filtered_dist_vector)
        # print(f'min_value = {min_value}')

    return distances_loss


# 載入 oversea_stations.json 中的經緯度數據
def load_oversea_stations(json_path):
    with open(json_path, 'r') as f:
        oversea_stations = set(map(lambda x: tuple(map(float, x.split(","))), json.load(f).keys()))
    return oversea_stations

def distance_loss_with_filter(pred_distance, station_coords, target_coords, epoch, batch_idx, plots_path):
    R = 6371  # 地球半徑，單位：公里

    # 定義台灣本島的經緯度範圍
    lat_min, lat_max = 21.5, 25.5  # 緯度範圍
    lon_min, lon_max = 119.5, 122.5  # 經度範圍

    # 載入海外測站資料
    oversea_stations = load_oversea_stations("/mnt/disk1/minxuan/TEAM_objective/oversea_stations.json")

    # 篩選出位於台灣本島範圍內的測站和目標
    lat_filter_stations = (station_coords[..., 0] >= lat_min) & (station_coords[..., 0] <= lat_max)
    lon_filter_stations = (station_coords[..., 1] >= lon_min) & (station_coords[..., 1] <= lon_max)
    taiwan_filter_stations = lat_filter_stations & lon_filter_stations

    lat_filter_targets = (target_coords[..., 0] >= lat_min) & (target_coords[..., 0] <= lat_max)
    lon_filter_targets = (target_coords[..., 1] >= lon_min) & (target_coords[..., 1] <= lon_max)
    taiwan_filter_targets = lat_filter_targets & lon_filter_targets

    # 過濾掉海外測站和目標
    station_latlon = torch.round(station_coords[..., :2] * 10000) / 10000  # 保留四位小數的經緯度
    # station_latlon_tuples = [tuple(coord.tolist()) for coord in station_latlon.view(-1, 2)]
    station_latlon_tuples = [tuple(float(f"{x:.4f}") for x in coord.tolist()) for coord in station_latlon.view(-1, 2)]
    oversea_mask_stations = torch.tensor([tuple(latlon) not in oversea_stations for latlon in station_latlon_tuples])
    oversea_mask_stations = oversea_mask_stations.view(station_coords.size(0), station_coords.size(1)).to(station_coords.device)

    target_latlon = torch.round(target_coords[..., :2] * 10000) / 10000  # 保留四位小數的經緯度
    # target_latlon_tuples = [tuple(coord.tolist()) for coord in target_latlon.view(-1, 2)]
    target_latlon_tuples = [tuple(float(f"{x:.4f}") for x in coord.tolist()) for coord in target_latlon.view(-1, 2)]
    oversea_mask_targets = torch.tensor([tuple(latlon) not in oversea_stations for latlon in target_latlon_tuples])
    oversea_mask_targets = oversea_mask_targets.view(target_coords.size(0), target_coords.size(1)).to(target_coords.device)

    valid_station_filter = taiwan_filter_stations & oversea_mask_stations
    valid_target_filter = taiwan_filter_targets & oversea_mask_targets

    # 確保過濾後的數據保持批次結構一致
    filtered_station_coords = station_coords.clone()
    filtered_target_coords = target_coords.clone()

    filtered_station_coords[~valid_station_filter] = float('nan')  # 將無效測站設置為 NaN
    filtered_target_coords[~valid_target_filter] = float('nan')    # 將無效目標設置為 NaN

    # 轉換為弧度，並過濾有效數據
    stations_rad = torch.deg2rad(filtered_station_coords[..., :2])
    targets_rad = torch.deg2rad(filtered_target_coords[..., :2])

    # 計算 Haversine 距離和垂直距離（跳過 NaN）
    dlat = targets_rad.unsqueeze(2)[..., 0] - stations_rad.unsqueeze(1)[..., 0]
    dlon = targets_rad.unsqueeze(2)[..., 1] - stations_rad.unsqueeze(1)[..., 1]

    # Haversine 公式
    a = torch.sin(dlat / 2) ** 2 + \
        torch.cos(stations_rad.unsqueeze(1)[..., 0]) * torch.cos(targets_rad.unsqueeze(2)[..., 0]) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(a))
    horizontal_dist = R * c  # 單位：公里

    # 計算垂直距離
    depth_diff = (filtered_target_coords.unsqueeze(2)[..., 2] - filtered_station_coords.unsqueeze(1)[..., 2]).abs()

    # 結合水平與垂直距離
    real_dist_matrix = torch.sqrt(horizontal_dist ** 2 + depth_diff ** 2)

    # 移除包含 NaN 的數據
    valid_mask = ~torch.isnan(real_dist_matrix)
    real_dist_matrix[~valid_mask] = 0

    dist_vector = real_dist_matrix.view(real_dist_matrix.size(0), -1)

    # 損失計算
    mse_loss = nn.MSELoss()
    distances_loss = mse_loss(pred_distance, dist_vector)

    # 在第一個 epoch 保存測站分布圖
    # if epoch == 0:
    #     save_plot(station_coords, target_coords, valid_station_filter, valid_target_filter, plots_path, batch_idx)

    return distances_loss

def save_plot(station_coords, target_coords, station_filter, target_filter, plots_path, batch_idx):
    """
    保存測站與目標分布圖，使用台灣地圖背景，並區分過濾條件。
    """
    # 創建保存目錄
    os.makedirs(plots_path, exist_ok=True)

    # 確保 filtered_stations 的形狀是 [num_stations, 3]，並且過濾掉深度值
    filtered_stations = station_coords[station_filter].cpu().numpy()[:, :2]
    excluded_stations = station_coords[~station_filter].cpu().numpy()[:, :2]  # 取前兩個維度（緯度、經度）
    # print(f'filtered_stations.shape = {filtered_stations.shape}')
    # print(f'excluded_stations.shape = {excluded_stations.shape}')
    # 台灣範圍
    lat_min, lat_max = 21.5, 25.5
    lon_min, lon_max = 119.5, 122.5
    
    # 建立台灣地圖背景
    fig, ax = plt.subplots(figsize=(10, 8))
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max, llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='i')
    m.drawcountries()
    m.drawcoastlines()
    m.drawmapboundary()

    # 轉換為經緯度對
    x_filtered, y_filtered = m(filtered_stations[:, 1], filtered_stations[:, 0])
    x_excluded, y_excluded = m(filtered_stations[:, 1], filtered_stations[:, 0])

    # 繪製測站分布
    m.scatter(x_excluded, y_excluded, c='gray', label='Excluded Stations', alpha=0.7)
    m.scatter(x_filtered, y_filtered, c='red', label='Valid Stations', zorder=5)

    # 標記測站座標
    for (x, y), coord in zip(zip(x_filtered, y_filtered), filtered_stations):
        plt.text(x, y, f"({coord[0]:.4f}, {coord[1]:.4f})", fontsize=5, color='red')

    # for (x, y), coord in zip(zip(x_excluded, y_excluded), excluded_stations):
    #     ax.text(x, y, f"({coord[0]:.2f}, {coord[1]:.2f})", fontsize=8, color='gray')

    # 繪製目標分布
    filtered_targets = target_coords[target_filter].cpu().numpy()[:, :2]
    x_targets, y_targets = m(filtered_targets[:, 1], filtered_targets[:, 0])

    m.scatter(x_targets, y_targets, c='blue', label='Targets')

    # 標記目標座標
    for (x, y), coord in zip(zip(x_targets, y_targets), filtered_targets):
        plt.text(x, y, f"({coord[0]:.4f}, {coord[1]:.4f})", fontsize=5, color='blue')

    # 添加標題和圖例
    plt.title("Taiwan Station Map")
    plt.legend()

    # 儲存圖像
    save_path = os.path.join(plots_path, f'stations_{batch_idx}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()


def training(model, optimizer, loader, epoch,epochs, device,training_params ,pga_loss,train_loss_record, loss_ratio, plots_path):

    # train_loop = tqdm(loader)
    train_loop = tqdm(enumerate(loader), total=len(loader))
    model.train()
    total_train_loss = 0.0
    total_pga_loss = 0.0
    total_distance_loss = 0.0

    # thresholds = np.array([0.02, 0.05, 0.1, 0.2]) #日本資料thresholds
    thresholds = np.log10(np.array([0.25, 0.8, 1.4, 2.5, 4.4, 8.0])*10) #台灣資料thresholds
    confusion_matrix = np.zeros((len(thresholds)+1, len(thresholds)+1)).astype(np.int32)
    # for x,y in train_loop:
    for batch_idx, (x, y) in train_loop:
        inputs_waveforms, inputs_coords, targets_coords, targets_pga = x[0].to(device), x[1].to(device),x[2].to(device), y[0].to(device)
        # print(f'inputs_waveforms: {inputs_waveforms.shape}')
        # print(f'inputs_coords: {inputs_coords.shape}')
        # print(f'targets_coords: {targets_coords.shape}')
        # targets_pga = targets_pga/9.8
        targets_0 = targets_pga==0
        targets_pga[targets_0] = 1e-6
        targets_pga[~targets_0] = torch.log10(targets_pga[~targets_0]*10) #日本資料註解掉
        # torch.save({'inputs_waveforms': inputs_waveforms,
        #             'inputs_coords': inputs_coords,
        #             'targets_coords': targets_coords}, 'data.pth')
        # input('please ...')
        pred, distance = model(inputs_waveforms, inputs_coords, targets_coords)     # Forward Pass
        # print(f'True: {targets_pga.shape}')
        # print(f'Pred: {pred.shape}')
        pgas_loss = pga_loss(targets_pga, pred)
        distances_loss = distance_loss(distance, inputs_coords, targets_coords)
        # distances_loss = distance_loss_with_filter(distance, inputs_coords, targets_coords, epoch, batch_idx, plots_path)
        train_loss = pgas_loss + loss_ratio * distances_loss  # Find the Loss 自訂損失比
        # train_loss = (pgas_loss * (distances_loss / (pgas_loss + distances_loss))) + (distances_loss * (pgas_loss / (pgas_loss + distances_loss))) #自動調節損失比
        
        total_train_loss = train_loss.item() + total_train_loss
        total_pga_loss = pgas_loss.item() + total_pga_loss
        total_distance_loss = distances_loss.item() + total_distance_loss
        
        train_loss.backward()      # Calculate gradients 
        
        clip_grad_norm_(model.parameters(), training_params['clipnorm'])
        optimizer.step()     # Update Weights     
        optimizer.zero_grad()     # Clear the gradients

        train_loop.set_description(f"[Train Epoch {epoch+1}/{epochs}]")
        train_loop.set_postfix(pga_loss=pgas_loss.detach().cpu().item(),distance_loss=distances_loss.detach().cpu().item(),loss=train_loss.detach().cpu().item())

        ######################### confusion matrix #########################
        targets_pga = targets_pga.contiguous().view(-1, (pred.shape[-1] - 1) // 2).cpu().numpy()   #(batch, 20, 1, 1) -> (batch*20, 1, 1)
        pred = pred.contiguous().view(-1, pred.shape[-2], pred.shape[-1]).detach().cpu().numpy() #(batch, 20, 5, 3) -> (batch*20, 5, 3)
        
        gaussian_confusion_matrix('train', 'accumulate', confusion_matrix, targets_pga=targets_pga, pred=pred, thresholds=thresholds)
    gaussian_confusion_matrix('train', 'write_txt', confusion_matrix, loop=train_loop, pga_loss=total_pga_loss, distance_loss=total_distance_loss, total_loss=total_train_loss, optimizer=optimizer)
    

    train_loss_record.append(total_train_loss / len(train_loop))  
    
    logging.info('[Train] epoch: %d -> loss: %.4f' %(epoch, total_train_loss / len(train_loop)))
    return model, optimizer,train_loss_record

def validating(model, optimizer, loader, epoch,epochs, device,pga_loss,scheduler,val_loss_record, loss_ratio, plots_path):
    
    # valid_loop = tqdm(loader)
    valid_loop = tqdm(enumerate(loader), total=len(loader))
    model.eval()
    total_val_loss = 0.0
    total_pga_loss = 0.0
    total_distance_loss = 0.0
    
    # thresholds = np.array([0.02, 0.05, 0.1, 0.2]) #日本資料thresholds
    thresholds = np.log10(np.array([0.25, 0.8, 1.4, 2.5, 4.4, 8.0])*10) #台灣資料thresholds
    confusion_matrix = np.zeros((len(thresholds)+1, len(thresholds)+1)).astype(np.int32)
    # for x,y in valid_loop:
    for batch_idx, (x, y) in valid_loop:
        with torch.no_grad():
            inputs_waveforms, inputs_coords, targets_coords, targets_pga = x[0].to(device), x[1].to(device),x[2].to(device), y[0].to(device)

            # targets_pga = targets_pga/9.8
            targets_0 = targets_pga==0
            targets_pga[targets_0] = 1e-6
            targets_pga[~targets_0] = torch.log10(targets_pga[~targets_0]*10) #日本資料註解掉
            
            pred, distance = model(inputs_waveforms, inputs_coords, targets_coords)
            
            pgas_loss = pga_loss(targets_pga, pred)
            distances_loss = distance_loss(distance, inputs_coords, targets_coords)
            # distances_loss = distance_loss_with_filter(distance, inputs_coords, targets_coords, epoch, batch_idx, plots_path)
            val_loss = pgas_loss + loss_ratio * distances_loss 
            # val_loss = (pgas_loss * (distances_loss / (pgas_loss + distances_loss))) + (distances_loss * (pgas_loss / (pgas_loss + distances_loss))) #自動調節損失比
            
            total_val_loss = val_loss.item() + total_val_loss
            total_pga_loss = pgas_loss.item() + total_pga_loss
            total_distance_loss = distances_loss.item() + total_distance_loss
        
            valid_loop.set_description(f"[Eval Epoch {epoch+1}/{epochs}]")
            valid_loop.set_postfix(pga_loss=pgas_loss.detach().cpu().item(), distance_loss=distances_loss.detach().cpu().item(), loss=val_loss.detach().cpu().item())
        
            ######################### confusion matrix #########################
            targets_pga = targets_pga.contiguous().view(-1, (pred.shape[-1] - 1) // 2).cpu().numpy()   #(batch, 20, 1, 1) -> (batch*20, 1, 1)
            pred = pred.contiguous().view(-1, pred.shape[-2], pred.shape[-1]).detach().cpu().numpy() #(batch, 20, 5, 3) -> (batch*20, 5, 3)

            gaussian_confusion_matrix('val', 'accumulate', confusion_matrix, targets_pga=targets_pga, pred=pred, thresholds=thresholds)
    gaussian_confusion_matrix('val', 'write_txt', confusion_matrix, loop=valid_loop, pga_loss=total_pga_loss, distance_loss=total_distance_loss, total_loss=total_val_loss, optimizer=optimizer)
        

    val_loss_record.append(total_val_loss / len(valid_loop))
    scheduler.step(total_val_loss/len(valid_loop))
    
    logging.info('[Eval] epoch: %d -> loss: %.4f' %(epoch, total_val_loss/len(valid_loop)))
    logging.info('======================================================')
    return val_loss_record

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    parser.add_argument('--continue_ensemble', action='store_true')  # Continues a stopped ensemble training
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))

    seed_np_pt(config.get('seed', 42))
    stations_table = json.load(open('./stations.json', 'r'))
    
    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])

    device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")
    loss_ratio = training_params['loss_ratio']
    
    if not os.path.isdir(training_params['weight_path']):
        os.makedirs(training_params['weight_path'])
    listdir = os.listdir(training_params['weight_path'])
    # if not args.test_run:
    #     if not args.continue_ensemble and listdir:
    #         if len(listdir) != 1 or listdir[0] != 'config.json':
    #             raise ValueError(f'Weight path needs to be empty. ({training_params["weight_path"]})')

    with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print('Loading data')
    if args.test_run:
        limit = 10
    else:
        limit = None

    # if not isinstance(training_params['data_path'], list):
    #     training_params['data_path'] = [training_params['data_path']]

    # assert len(generator_params) == len(training_params['data_path'])

    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    #台灣資料loader,py檔中的event_key=data_file、日本資料loader.py檔中的event_key=KiK_File #要改資料集
    full_data_train = [
        loader.load_events(
            training_params['train_data_path'],
            limit=limit,
            shuffle_train_dev=generator.get('shuffle_train_dev', False),
            custom_split=generator.get('custom_split', None),
            min_mag=generator.get('min_mag', None),
            mag_key=generator.get('key', 'MA'),
            overwrite_sampling_rate=overwrite_sampling_rate,
            decimate_events=generator.get('decimate_events', None)
        )
        for generator in generator_params
    ]
    full_data_dev = [
        loader.load_events(
            training_params['val_data_path'], limit=limit,
            shuffle_train_dev=generator.get('shuffle_train_dev', False),
            custom_split=generator.get('custom_split', None),
            min_mag=generator.get('min_mag', None),
            mag_key=generator.get('key', 'MA'),
            overwrite_sampling_rate=overwrite_sampling_rate,
            decimate_events=generator.get('decimate_events', None)
        )
        for generator in generator_params]
    
    event_metadata_train = [d[0] for d in full_data_train]
    data_train = [d[1] for d in full_data_train]
    metadata_train = [d[2] for d in full_data_train]
    event_metadata_dev = [d[0] for d in full_data_dev]
    data_dev = [d[1] for d in full_data_dev]
    metadata_dev = [d[2] for d in full_data_dev]

    sampling_rate = metadata_train[0]['sampling_rate']
    assert all(m['sampling_rate'] == sampling_rate for m in metadata_train + metadata_dev)
    waveforms = data_train[0]['waveforms']

    max_stations = config['model_params']['max_stations']
    ensemble = config.get('ensemble', 1)

    super_config = config.copy()
    super_training_params = training_params.copy()
    super_model_params = config['model_params'].copy()

    # for ens_id in range(ensemble):
    for ens_id in [0]:
        print('==============================================================')
        print('===============     第 {}/{} 輪Ensemble開始     ==============='.format(ens_id + 1, ensemble))
        print('==============================================================')
        print(' ')
        if ensemble > 1:
            seed_np_pt(ens_id)

            config = super_config.copy()
            config['ens_id'] = ens_id
            training_params = super_training_params.copy()
            training_params['weight_path'] = os.path.join(training_params['weight_path'], f'{ens_id}')
            config['training_params'] = training_params
            config['model_params'] = super_model_params.copy()

            if training_params.get('ensemble_rotation', False):
                # Rotated by angles between 0 and pi/4
                config['model_params']['rotation'] = np.pi / 4 * ens_id / (ensemble - 1)
            
            if args.continue_ensemble and os.path.isdir(training_params['weight_path']):
                hist_path = os.path.join(training_params['weight_path'], 'hist.pkl')
                if os.path.isfile(hist_path):
                    continue
                else:
                    raise ValueError(f'Can not continue unclean ensemble. Checking for {hist_path} failed.')

            if not os.path.isdir(training_params['weight_path']):
                os.mkdir(training_params['weight_path'])

            with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)


        #print('Building model')
        full_model = models.build_transformer_model(**config['model_params'], device=device, trace_length=data_train[0]['waveforms'][0].shape[1]).to(device)

        key = generator_params[0]['key']
    
        # x_train = np.concatenate(data_train[0]['waveforms'], axis=0)
        # x_dev = np.concatenate(data_dev[0]['waveforms'], axis=0)
        # y_train = np.concatenate([np.full(x.shape[0], mag) for x, mag in
        #                         zip(data_train[0]['waveforms'], event_metadata_train[0][key])])
        # y_dev = np.concatenate([np.full(x.shape[0], mag) for x, mag in
        #                         zip(data_dev[0]['waveforms'], event_metadata_dev[0][key])])

        # train_mask = (x_train != 0).any(axis=(1, 2))
        # dev_mask = (x_dev != 0).any(axis=(1, 2))
        # x_train = x_train[train_mask]
        # y_train = y_train[train_mask]
        # x_dev = x_dev[dev_mask]
        # y_dev = y_dev[dev_mask]
        
        noise_seconds = generator_params[0].get('noise_seconds', 5)
        cutout = (
            sampling_rate * (noise_seconds + generator_params[0]['cutout_start']), sampling_rate * (noise_seconds + generator_params[0]['cutout_end']))
        sliding_window = generator_params[0].get('sliding_window', False)
        n_pga_targets = config['model_params'].get('n_pga_targets', 0)
        
        if 'load_model_path' in training_params:
            print('Loading full model')
            full_model.load_weights(training_params['load_model_path'])

        if 'transfer_model_path' in training_params:
            print('Transfering model weights')
            ensemble_load = training_params.get('ensemble_load', False)
            wait_for_load = training_params.get('wait_for_load', False)
            full_model = transfer_weights(full_model, training_params['transfer_model_path'],
                            ensemble_load=ensemble_load, wait_for_load=wait_for_load, ens_id=ens_id)
        
        # saved_state_dict = torch.load('/mnt/disk1/minxuan/TEAM_objective/senior_model_weight_2/0/checkpoint_06.pth')
        # full_model.load_state_dict(saved_state_dict, strict=False)  # 加載模型參數
        
        train_datas = []
        val_datas = []               
        
        for i, generator_param_set in enumerate(generator_params): #如果有>1個訓練參數，可以加在training_params裡面，會在這裡遞迴前處理
            noise_seconds = generator_param_set.get('noise_seconds', 5)
            cutout = (sampling_rate * (noise_seconds + generator_param_set['cutout_start']), sampling_rate * (noise_seconds + generator_param_set['cutout_end']))

            generator_param_set['transform_target_only'] = generator_param_set.get('transform_target_only', True)
            #台灣資料config.json檔中的key=source_magnitude、日本資料config.json檔中的key=M_J
            train_datas += [util.PreloadedEventGenerator(data=data_train[i],
                                                        event_metadata=event_metadata_train[i],
                                                        coords_target=True,
                                                        label_smoothing=True,
                                                        station_blinding=True,
                                                        cutout=cutout,
                                                        pga_targets=n_pga_targets,
                                                        max_stations=max_stations,
                                                        sampling_rate=sampling_rate,
                                                        **generator_param_set)]
            old_oversample = generator_param_set.get('oversample', 1)
            val_datas += [util.PreloadedEventGenerator(data=data_dev[i],
                                                            event_metadata=event_metadata_dev[i],
                                                            coords_target=True,
                                                            station_blinding=True,
                                                            cutout=cutout,
                                                            pga_targets=n_pga_targets,
                                                            max_stations=max_stations,
                                                            sampling_rate=sampling_rate,
                                                            **generator_param_set)]
            generator_param_set['oversample'] = old_oversample
            
        filepath = os.path.join(training_params['weight_path'], 'event-{epoch:02d}.hdf5')
        workers = training_params.get('workers', 10)
        
        val_generators = DataLoader(val_datas[0], shuffle=False, batch_size=None, collate_fn=models.my_collate)
        
        optimizer = torch.optim.AdamW(full_model.parameters(), lr=training_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4)
        
        if n_pga_targets:
            def pga_loss(y_true, y_pred):
                return models.time_distributed_loss(y_true, y_pred, models.mixture_density_loss, device, mean=True, kwloss={'mean': False})
        
        losses = {}
        losses['pga'] = pga_loss
            
            
        num_epochs = training_params['epochs_full_model']
        metrics_record = {}
        train_loss_record = []
        val_loss_record = []
        lr_record = []
        log_path = training_params['weight_path']+'/train.log'
        logging.basicConfig(filename=os.path.join(os.getcwd(), log_path), filemode='a', level=logging.INFO)
        logging.info('start training')
        train_generators = DataLoader(train_datas[0], shuffle=True, batch_size=None, collate_fn=models.my_collate, pin_memory=True, num_workers=workers)
        val_generators = DataLoader(val_datas[0], shuffle=False, batch_size=None, collate_fn=models.my_collate, pin_memory=True, num_workers=workers)
        
        no_improve_epochs = 0
        best_val_loss = float('inf')
        early_stop_patience = 10  # 連續幾次沒改善就停止
        
        for epoch in range(num_epochs):
            
            
            full_model, optimizer,train_loss_record = training(full_model, optimizer, train_generators, epoch,num_epochs, device,training_params ,pga_loss,train_loss_record, loss_ratio, os.path.join(training_params['weight_path'], 'train_stations_plots'))
            val_loss_record = validating(full_model, optimizer, val_generators, epoch,num_epochs, device,pga_loss,scheduler,val_loss_record, loss_ratio, os.path.join(training_params['weight_path'], 'valid_stations_plots'))
            lr_record.append(scheduler.optimizer.param_groups[0]['lr'])
            
            #save model    
            # if epoch>5:
            #     if val_loss_record[-1] < min(val_loss_record[5:-1]): #從第五個epoch到前一個epoch中的最小loss
            #         metrics_record['train_loss'] = train_loss_record
            #         metrics_record['val_loss'] = val_loss_record
            #         metrics_record['lr_record'] = lr_record
            #         with open (os.path.join(training_params['weight_path'], 'metrics.txt'), 'w', encoding='utf-8') as f:
            #             f.write(str(metrics_record))

            #         print("-----Saving checkpoint-----")
            #         torch.save({
            #             'model_weights' : full_model.state_dict(), 
            #             'optimizer' : optimizer.state_dict(),
            #             'scheduler' : scheduler.state_dict(),
            #                     }, 
            #         os.path.join(training_params['weight_path'], f'checkpoint_{epoch:02d}.pth'))
            
            if val_loss_record[-1] < best_val_loss:
                best_val_loss = val_loss_record[-1]
                no_improve_epochs = 0  # reset counter
                
                metrics_record['train_loss'] = train_loss_record
                metrics_record['val_loss'] = val_loss_record
                metrics_record['lr_record'] = lr_record
                with open (os.path.join(training_params['weight_path'], 'metrics.txt'), 'w', encoding='utf-8') as f:
                    f.write(str(metrics_record))

                print("-----Saving checkpoint-----")
                torch.save({
                    'model_weights' : full_model.state_dict(), 
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                            }, 
                os.path.join(training_params['weight_path'], f'checkpoint_{epoch:02d}.pth'))
            else:
                no_improve_epochs += 1
                print(f"Validation loss not improve in {no_improve_epochs} epoches!")
            if no_improve_epochs >= early_stop_patience:
                print(f"early stop in epoch{epoch}!!!!!")
                break
                
# train 2018 evalution 2013  test 2014
                