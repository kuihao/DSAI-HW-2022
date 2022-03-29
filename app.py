import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
# activate DSAI-py364-tf-gpu
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (10, 5)
mpl.rcParams['axes.grid'] = False

def safely_mkdir(dirPath:str):
    '''安全地生成目錄，若目錄已存在則無動作'''
    if os.path.exists(dirPath):
        print("Directory is exist.")
    os.makedirs(dirPath, exist_ok=True) 
    return dirPath

def printlog(msg:str, dirPath:str, mode='a', sep=' ', end='\n', flush=False):
    '''print to std.out and 
    write to log file at the same time'''
    print(msg, sep=sep, end=end, flush=flush)
    with open(dirPath, mode=mode) as f:
        print(msg, file=f, sep=sep, end=end, flush=flush)
    
    # Below is almost the same, but no newline mark at end of line.
    #with open(dirPath, mode=mode, encoding=encoding, newline=newline) as f:
    #    f.write(msg)
    
def onehot_preprocess(df_dataset):
    '''對星期的資訊進行 one-hot 編碼'''
    # 資料集轉 numpy
    dataset_np = df_dataset.to_numpy()

    # 星期做成 onehot
    onehots = np.eye(8)[dataset_np[:,1].astype(int).tolist()]
    onehots = np.delete(onehots, [0], axis=1) 

    # onehot 加入 df_training
    df_dataset['MOM'] = onehots[:,0]
    df_dataset['TUE'] = onehots[:,1]
    df_dataset['WED'] = onehots[:,2]
    df_dataset['THU'] = onehots[:,3]
    df_dataset['FRI'] = onehots[:,4]
    df_dataset['SAT'] = onehots[:,5]
    df_dataset['SUN'] = onehots[:,6]
    
    return df_dataset

def mark_lowpower_preprocess(df_dataset):
    '''對一周之中用電量較低的日子進行標記'''
    mark = []
    for i in range(len(df_dataset)):
        if df_dataset['Week'].iloc[i] == 7 or df_dataset['Week'].iloc[i] == 1:
            mark.append(1)
        else:
            mark.append(0)
    df_dataset['Low Consumption'] = np.array(mark)
    # 丟去不重要資訊
    df_dataset = df_dataset[['Date Time', 'Net Peaking Capability (MW)', 'Peak Load (MW)', 'Low Consumption','MOM', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']]
    return df_dataset

def dataset_split(df_dataset):
    '''切出用來 inference 的 data'''
    inference_data_7day_before = df_dataset.iloc[:7].to_numpy().astype('float32')
    inference_data_thisday_weekOnehot = df_training[['MOM', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']].iloc[:].to_numpy().astype('float32')
    inference_data_thisday_LowConsumption = df_training['Low Consumption'].iloc[:].to_numpy().astype('float32')
    return inference_data_7day_before, inference_data_thisday_weekOnehot, inference_data_thisday_LowConsumption

# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    
    #df_submission = pd.read_csv('submission.csv')
    df_submission = pd.read_csv(args.output)
    
    #df_training = pd.read_csv('training_data.csv')
    df_training = pd.read_csv(args.training)
    df_training = onehot_preprocess(df_training)
    df_training = mark_lowpower_preprocess(df_training)
    df_training
    inference_data_7day_before, inference_data_thisday_weekOnehot, inference_data_thisday_LowConsumption = dataset_split(df_training)
    inference_data_7day_before = inference_data_7day_before[np.newaxis,:]
    inference_data_thisday_LowConsumption = inference_data_thisday_LowConsumption[np.newaxis,:]
    
    # model
    model_op_predictor = tf.keras.models.load_model('./model/2GRU_DNN_v2_onehot.h5')
    model_net_predictor = tf.keras.models.load_model('./model/2GRU_DNN_v21_NetPeakingCapability.h5')
    model_pl_predictor = tf.keras.models.load_model('./model/2GRU_DNN_v21_PeakLoad.h5')
    
    # Do while
    # op
    submission_np = np.array([])
    op = model_op_predictor.predict([inference_data_7day_before[:,:,(1,3,4,5,6,7,8,9,10)], 
                            inference_data_7day_before[:,:,(2,3,4,5,6,7,8,9,10)], 
                            inference_data_thisday_LowConsumption[:,7]])
    submission_np = np.append(submission_np, op)
    op, submission_np
    
    # net
    net_np = df_training['Net Peaking Capability (MW)'].iloc[0:7]
    net = model_net_predictor.predict([inference_data_7day_before[:,:,(1,3,4,5,6,7,8,9,10)], 
                                   inference_data_thisday_LowConsumption[:,7]])
    net_np = np.append(net_np, net)
    net, net_np
    
    # pl
    pl_np = df_training['Peak Load (MW)'].iloc[0:7]
    pl = model_pl_predictor.predict([inference_data_7day_before[:,:,(2,3,4,5,6,7,8,9,10)], 
                                   inference_data_thisday_LowConsumption[:,7]])
    pl_np = np.append(pl_np, pl)
    pl, pl_np
    
    # loop
    for i in range(1,16):
        tmp_np = np.append(net_np[np.newaxis,i:i+7, np.newaxis], inference_data_thisday_LowConsumption[:,i:i+7, np.newaxis], axis=2) 
        tmp_np = np.append(tmp_np, inference_data_thisday_weekOnehot[np.newaxis, i:i+7,:], axis=2) 
        tmp_np.shape

        tmp_np2 = np.append(pl_np[np.newaxis,i:i+7, np.newaxis], inference_data_thisday_LowConsumption[:,i:i+7, np.newaxis], axis=2) 
        tmp_np2 = np.append(tmp_np2, inference_data_thisday_weekOnehot[np.newaxis, i:i+7,:], axis=2) 
        tmp_np2.shape


        op = model_op_predictor.predict([tmp_np, 
                                        tmp_np2, 
                                        inference_data_thisday_LowConsumption[:,7+i]])
        print(op)

        submission_np = np.append(submission_np, op)

        net = model_net_predictor.predict([tmp_np, 
                                           inference_data_thisday_LowConsumption[:,7+i]])
        net_np = np.append(net_np, net)

        pl = model_pl_predictor.predict([tmp_np2, 
                                        inference_data_thisday_LowConsumption[:,7+i]])
        pl_np = np.append(pl_np, pl)
        
    df_submission['operating_reserve(MW)'] = submission_np[1:]
    df_submission
    # save
    df_submission.to_csv('submission.csv', index=0)

# python app.py --training training_data.csv --output submission.csv