### leading time 算預測早報和晚報的情況 ###
import argparse
import pandas as pd
import numpy as np 
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score

def get_val_threshold(val_path):
    # 讀取 pkl 檔案
    data = pd.read_pickle(val_path)
    warning_time_information = data[4]
    
    TP = np.zeros((7, 9)) #(pga_level, threshold)
    FP = np.zeros((7, 9))
    FN = np.zeros((7, 9))
    TN = np.zeros((7, 9))

    # 新增 leading_time 陣列，累加所有預測情況的 leading time
    leading_times = np.zeros((7, 9))
    leading_times_count = np.zeros((7, 9))  # 記錄所有 leading time 的數量

    for row in tqdm(warning_time_information, total=len(warning_time_information)):
        pga_times_pred = np.array(row[0])
        pga_times_true = np.repeat(np.array(row[1]), pga_times_pred.shape[2], axis=1)
        pga_times_true = pga_times_true.reshape(pga_times_pred.shape)

        # pred有值且label有值
        have_value_array = np.logical_and(np.logical_not(np.isnan(pga_times_pred)), np.logical_not(np.isnan(pga_times_true)))
        # pred沒值且label有值
        null_have_value_array = np.logical_and(np.isnan(pga_times_pred), np.logical_not(np.isnan(pga_times_true)))
        # pred有值且label沒值
        have_value_null_array = np.logical_and(np.logical_not(np.isnan(pga_times_pred)), np.isnan(pga_times_true))
        # pred沒值且label沒值
        null_null_array = np.logical_and(np.isnan(pga_times_pred), np.isnan(pga_times_true))

        # FN(錯過警告): pred和label都有值，但是pred>=label時間。或pred沒值且label有值。
        fn_array = np.logical_and(have_value_array, pga_times_pred >= pga_times_true)

        have_value_array = have_value_array.astype(int)
        null_have_value_array = null_have_value_array.astype(int)
        have_value_null_array = have_value_null_array.astype(int)
        null_null_array = null_null_array.astype(int)
        fn_array = fn_array.astype(int)

        for i in range(7):
            for j in range(9):
                tp = sum(have_value_array[:, i, j].tolist()) - sum(fn_array[:, i, j].tolist())
                fn = sum(null_have_value_array[:, i, j].tolist()) + sum(fn_array[:, i, j].tolist())
                fp = sum(have_value_null_array[:, i, j].tolist())
                tn = sum(null_null_array[:, i, j].tolist())

                TP[i][j] += tp
                FP[i][j] += fp
                FN[i][j] += fn
                TN[i][j] += tn

                # 計算 TP 的 leading time，累加所有預測情況
                for k in range(pga_times_pred.shape[0]):  # 假設每個row有多個時間步
                    for l in range(pga_times_pred.shape[1]):
                        if have_value_array[k, i, l] :
                            leading_time = pga_times_true[k, i, l] - pga_times_pred[k, i, l]
                            leading_times[i][j] += leading_time
                            leading_times_count[i][j] += 1
    
    # 計算 Precision, Recall, F1 score
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_leading_times = np.divide(leading_times, leading_times_count, out=np.zeros_like(leading_times, dtype=float), where=(leading_times_count) != 0)
        Precision = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
        Recall = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
        F1_score = np.divide(2 * Precision * Recall, (Precision + Recall), out=np.zeros_like(TP, dtype=float), where=(Precision + Recall) != 0)

    # 計算 F1 score 索引
    f1_index = np.argmax(F1_score, axis=1)
    
    return f1_index, Precision, Recall, F1_score, avg_leading_times, TP, FP, FN, TN


def write_results(result_path, f1_index, Precision, Recall, F1_score, avg_leading_times, TP, FP, FN, TN, title):
    # 儲存結果到 txt 檔案
    with open(result_path, 'a') as f:
        f.write(f"{title}, threshold:{f1_index}\n")
        f.write("橫軸: Precision, Recall, F1_score, Avg Leading Time, TP, FP, FN, TN\n")
        f.write("縱軸: 3級, 4級, 5弱, 5強, 6弱, 6強, 7級\n")
        
        # 標題行
        f.write(f"{'Precision':>12} {'Recall':>12} {'F1_score':>12} {'Avg Lead Time':>16} {'TP':>8} {'FP':>8} {'FN':>8} {'TN':>8}\n")
        
        for pga_index in range(7):
            # 每行格式化輸出，確保欄位對齊
            f.write(f"{Precision[pga_index][f1_index[pga_index]]:12.5f} "
                    f"{Recall[pga_index][f1_index[pga_index]]:12.5f} "
                    f"{F1_score[pga_index][f1_index[pga_index]]:12.5f} "
                    f"{avg_leading_times[pga_index][f1_index[pga_index]]:16.5f} "
                    f"{int(TP[pga_index][f1_index[pga_index]]):8d} "
                    f"{int(FP[pga_index][f1_index[pga_index]]):8d} "
                    f"{int(FN[pga_index][f1_index[pga_index]]):8d} "
                    f"{int(TN[pga_index][f1_index[pga_index]]):8d}\n")
        
        f.write("\n")



if __name__ == '__main__':
    # 設定命令行參數解析
    parser = argparse.ArgumentParser(description="Evaluate model and save results.")
    # parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    
    args = parser.parse_args()
    
    # val 路徑
    path_parts = args.path.rsplit("test", 1)  # 按最後一個 "test" 分割
    val_path = "val".join(path_parts)  # 替換為 "val"
    result_path = os.path.splitext(args.path)[0] + '_results.txt'
    # val_path = args.path.replace("test", "val")
    if os.path.exists(val_path):
        threshold, Precision, Recall, F1_score, avg_leading_times, TP, FP, FN, TN = get_val_threshold(val_path)
        
        write_results(result_path, threshold, Precision, Recall, F1_score, avg_leading_times, TP, FP, FN, TN, 'val(best)')
    
        f1_index, Precision, Recall, F1_score, avg_leading_times, TP, FP, FN, TN = get_val_threshold(args.path)
        write_results(result_path, f1_index, Precision, Recall, F1_score, avg_leading_times, TP, FP, FN, TN, 'test(best)')
        write_results(result_path, threshold, Precision, Recall, F1_score, avg_leading_times, TP, FP, FN, TN, 'final results(use val best threshold in test)')
    else:
        print(f"路徑不存在：{val_path}")
        f1_index, Precision, Recall, F1_score, avg_leading_times, TP, FP, FN, TN = get_val_threshold(args.path)
        write_results(result_path, f1_index, Precision, Recall, F1_score, avg_leading_times, TP, FP, FN, TN, 'test(best)')
        
    
    
