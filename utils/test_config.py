import nibabel as nib
import numpy as np
import os
import pandas as pd
from scipy import stats
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from medpy.metric.binary import assd, hd, hd95
from sklearn.metrics import rand_score, adjusted_rand_score

# Calculation and evaluation metrics
def metrics_cal(prediction_file, target_file, metrics_list, class_num):
    empty_value = -1.0 # define a null value
    smooth = 1e-10
    metrics = {metric: np.full(class_num, empty_value, dtype=np.float32) for metric in metrics_list} # Creates an array initialized to -1 for each indicator specified.

    # Load prediction and target files and convert to numpy arrays
    prediction = np.asanyarray(nib.load(prediction_file).dataobj)  
    target = np.asanyarray(nib.load(target_file).dataobj)            
    
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
  
        # Get the mask of target and predicted category i      
        prediction_per_class = (prediction == i).astype(np.float32)
        target_per_class = (target == i).astype(np.float32)        
        
        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        tn = np.sum((1 - prediction_per_class) * (1 - target_per_class))
        
        # base metrics
        if 'pre' in metrics_list:
            metrics['pre'][i] = (tp + smooth) / (tp + fp + smooth)
        if 'recall' in metrics_list:
            metrics['recall'][i] = (tp + smooth) / (tp + fn + smooth)
        if 'spec' in metrics_list:
            metrics['spec'][i] = (tn + smooth) / (tn + fp + smooth)
        if 'acc' in metrics_list:
            metrics['acc'][i] = (tp + tn + smooth) / (tp + fp + tn + fn + smooth)
        
        # extra metrics
        if 'iou' in metrics_list:
            metrics['iou'][i] = (tp + smooth) / (tp + fp + fn + smooth)
        if 'dice' in metrics_list:
            metrics['dice'][i] = (2. * tp + smooth) / (np.sum(target_per_class + prediction_per_class) + smooth)
        if 'assd' in metrics_list:
            metrics['assd'][i] = assd(target_per_class, prediction_per_class)
        if 'hd' in metrics_list:
            metrics['hd'][i] = hd(target_per_class, prediction_per_class)
        if 'hd95' in metrics_list:
            metrics['hd95'][i] = hd95(target_per_class, prediction_per_class)
        if 'voe' in metrics_list:
            metrics['voe'][i] = 1 - (tp + smooth) / (tp + fp + fn + smooth)
                        
        target_flat = target_per_class.flatten().astype(bool)
        prediction_flat = prediction_per_class.flatten().astype(bool)

        # cluster metrics
        if 'rand' in metrics_list:
            metrics['rand'][i] = rand_score(target_flat, prediction_flat)
        if 'adj_rand' in metrics_list:
            metrics['adj_rand'][i] = adjusted_rand_score(target_flat, prediction_flat)
                    
    for key in metrics:
        metrics[key] = np.where(metrics[key] == -1.0, np.nan, metrics[key]) # Replace -1 with NaN
        
    return metrics

# Compute statistics for a set of values
def statistics_cal(values):
    return [
        np.mean(values),      
        np.max(values),       
        np.min(values),        
        np.var(values),         
        np.std(values),         
        np.median(values),    
        np.ptp(values),        
        stats.skew(values),     
        stats.kurtosis(values)  
    ]

# Evaluate and save metrics to excel file
def evaluate(prediction_nii_files, target_nii_files, metrics_list, class_num, res_path, file_names):
    all_metrics = {key: [[] for _ in range(class_num + 1)] for key in metrics_list} # Initializes a list of all metrics, one for each category, plus an 'ALL' to generate a dictionary

    with ProcessPoolExecutor() as executor: # use multiple processes
        results = list(tqdm(executor.map(partial(metrics_cal, metrics_list=metrics_list, class_num=class_num), prediction_nii_files, target_nii_files), total=len(prediction_nii_files)))

    for metrics in results:
        for i in range(class_num):
            for key in metrics:
                all_metrics[key][i].append(metrics[key][i])

        # Calculate the average for non-background categories
        for key in all_metrics:
            non_background_metrics = [metrics[key][i] for i in range(1, class_num)]
            mean_value = np.nanmean(non_background_metrics)
            all_metrics[key][class_num].append(mean_value)
                
    # Create and save to Excel
    with pd.ExcelWriter(res_path) as writer:
        for i in range(class_num + 1):
            data = {
                '文件名': file_names + ['平均值', '最大值', '最小值', '方差', '标准差', '中位数', '极差', '偏度', '峰度'],
            }
            for key in all_metrics:
                data[key] = all_metrics[key][i] + statistics_cal(all_metrics[key][i])
            
            df = pd.DataFrame(data)
            sheet_name = f'类别{i}' if i != class_num else 'All'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Process the file path and call the evaluation function
def jisuan_excel(pre_path, tar_path, res_path, metrics_list, class_num):
    prediction_files = []
    target_files = []
    file_names = []
    for filename in os.listdir(pre_path):
        if filename.endswith(".nii.gz"):
            prediction_files.append(os.path.join(pre_path, filename))
            target_files.append(os.path.join(tar_path, filename))
            file_names.append(filename)

    evaluate(prediction_files, target_files, metrics_list, class_num, res_path, file_names)