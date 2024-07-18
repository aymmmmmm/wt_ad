# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:20:07 2018

@author: TZ-1
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
import pickle
import os
import os.path

turbineName = 'turbineName'
ts = 'time' # 时间戳
ws = 'wind_speed' # 风速变量名
gp = 'power' # 有功功率变量名
wtstatus = 'main_status'  # 机组状态变量名
wtstatus_n = 14  # 正常发电状态代码
cabin_temp= 'cabin_temp' # 机舱内温度变量名
outside_temp = 'environment_temp' # 舱外温度变量名
gs = 'generator_rotating_speed' # 发动机转速变量名
cv = 'nacelle_acc_X' # 塔筒振动x轴方向变量名
b1pitch ='pitch_1'
ratedgp =3200
columns = [ts, wtstatus, cv, ws, gp, cabin_temp, outside_temp, gs, b1pitch]

def data_preprocessing(data):
    """
    将输入数据与点表变量名进行匹配，转换时间戳、排序等一些预处理
    :data: 输入历史2小时一台风机的数据
    :return: 返回处理后的该台风机正常发电时的数据
    """
    data_tmp = data.copy()
    # data_tmp = data_tmp[[ts, ws, gp, outside_temp, cabin_temp, gs, cv, wtid, b1pitch]]
    data_tmp = data_tmp[[ts, turbineName, ws, gp, gs, b1pitch]]

    #风向没有清洗
    data_tmp = data_tmp[(data_tmp[ws] > 3) & (data_tmp[gp] > 10) & (data_tmp[gs] > 5)
                        & ~ ((data_tmp[gp] < ratedgp * 0.95) & (data_tmp[b1pitch] > 3))
                        # & ~ ((data_tmp[gp] < 1500 ) & (data_tmp[ws] > 10))
                            ]

    # data_tmp = data_tmp[(data_tmp[ws] > 3) & (data_tmp[gp] > 10) & (data_tmp[gs] > 5)]
    # data_tmp = data_tmp[[ts, ws, gp, outside_temp, cabin_temp, gs, cv]]
    data_tmp[ts] = pd.to_datetime(data_tmp[ts])
    data_tmp[ts] = data_tmp[ts].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    data_tmp = data_tmp.set_index(ts, drop = True).sort_index()

    # plt.scatter(data_tmp[ws], data_tmp[gp], s=1)
    # plt.show()
    return data_tmp

def data_preprocessing_realData(data):
    #实测数据补全
    data_tmp = data.copy()
    data_tmp = data_tmp[[ts, ws, gp, outside_temp, cabin_temp, gs, cv, b1pitch]]
    data_tmp[ts] = pd.to_datetime(data_tmp[ts])
    data_tmp[ts] = data_tmp[ts].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    data_tmp = data_tmp.set_index(ts, drop = True).sort_index()

    data_tmp.dropna(subset=[ ws, gp], inplace=True)


    for i in data_tmp.columns:
        if i == cabin_temp or i == outside_temp or i == gs or i == cv or i == b1pitch :
            data_tmp[i] = data_tmp[i].interpolate(method='linear')
            #如果还有空的，就用后一个填充-20190916
            data_tmp[i]=data_tmp[i].fillna(method='backfill')
            data_tmp[i]=data_tmp[i].fillna(method='ffill')

    return data_tmp


def ws_power_fit(data):
    """
    根据历史长时期正常发电时风速 - 功率关系使用GradientBoostingRegressor模型对风功率进行拟合
    :data: 输入预处理后的历史2小时一台风机的数据
    :return: 训练好的该台风机的风功率拟合模型, 模型拟合精度R方
    """
    x = data[ws].values.reshape(-1, 1)
    y = data[gp]
    
    params = {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.01, 'random_state': 0, 'loss': 'ls'}
    gbreg = GradientBoostingRegressor(** params)
    
    gbreg.fit(x, y)
    pickle.dump(gbreg, 'ws_power_model.pkl') # 储存模型文件
    return gbreg, gbreg.score(x, y)

def import_data_check(data, columns):
    status_code='000'

    # None
    if data is None: # 判断数据中某列是否全部为空值
        # raise Exception("Input Data_Raw is None")
        status_code = '200'
    # pd.DataFrame()
    elif data.shape[0] == 0:
        # raise Exception('Input Data_Raw is Empty Data_Raw Frame')
        status_code = '200'
    # NAN
    elif data.isnull().all().any():
        for col in columns:
            if len(data[data[col].notna()]) == 0:
                status_code = '301'


                # raise Exception(col + '  is NAN')
                return status_code
        # raise Exception('At Least One Column of Input Data_Raw is NAN')
    # lack of variable
    if not set(columns).issubset(set(data.columns)):
        # raise Exception('Some Variables are Missing')
        status_code = '301'
    # data type
    if (data[columns[3:]].dtypes != 'float').any():
        # raise Exception('the Dtype of Input Data_Raw is not Float')
        status_code = '302'
    # duplicate values
    for i in columns[3:]:
        if sum(data[i].duplicated()) == len(data):
            # raise Exception('Input Data_Raw Contains Too Many Duplicates')
            status_code = '102'
    # negative value
    if any(data[gs] < 0):
        # raise Exception('Rotating Speed Contains Negative Values')

        status_code = '302'
    #状态码如果有缺失按照最靠近的一个做填充-20210916
    data['main_status']=data['main_status'].fillna(method='ffill')

    if sum(data['main_status'] == wtstatus_n ) == 0: # 检查风机正常运行状态数量是否为0
        status_code = '300'


    elif sum(data['main_status'] == wtstatus_n) < 0.5 * len(data): # 检查风机正常运行状态数量是否超过一半
        status_code = '100'
    # shortage of data
    if len(data) < 180:
        status_code = '101'
    return status_code



if __name__ == '__main__':
    data_path = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\华润风机数据\\数据\\offline_analysis\\1min\\3.2MW\\'

    fileNames = []
    wind_turbine_IDs = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            wind_turbine_IDs.append(name.split('.')[0])
            fileNames.append(os.path.join(root, name))

    for i in range(len(wind_turbine_IDs)):
        wt_id = wind_turbine_IDs[i]
        path_10min = data_path + wt_id + '.csv'
        all_data_10min = pd.read_csv(path_10min, encoding='gbk')
        # all_data_10min[turbineName] = wt_id
        # all_data_10min[rs] = all_data_10min[gs] / 106.87
        all_data_10min[ts] = pd.to_datetime(all_data_10min[ts])
        all_data_10min = all_data_10min.drop_duplicates([ts])
        vars_10min = all_data_10min.columns.to_list()

        train = all_data_10min[:int(len(all_data_10min) * 0.4)]

        try:
            train = data_preprocessing(train)

            x = train[ws].values.reshape(-1, 1)
            y = train[gp]
            # plt.scatter(x, y, s=0.2)
            # plt.show()

            params = {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.01, 'random_state': 0, 'loss': 'ls'}
            gbreg = GradientBoostingRegressor(** params)

            gbreg.fit(x, y)

            save_path = '../Resource/blade_icing/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            model_save_path  = save_path + 'ws_power_model_' + wt_id + '.pkl'


            with open(model_save_path, 'wb') as f:
               pickle.dump(gbreg,f)

            with open(model_save_path, 'rb') as f:
               ws_power_model = pickle.load(f)  # 读取该风机对应的理论风速-功率模型, 该模型由过去长时期正常运行数据拟合 #####################

            power_d = ws_power_model.predict(pd.DataFrame(np.arange(1,20,0.5)))

            plt.scatter(x, y, s=0.2)
            plt.plot(np.arange(1,20,0.5),power_d , color ='red')

            plt.savefig(save_path + wt_id + '.jpg')
            # plt.show()
            plt.close()
        except:
            pass


        
        