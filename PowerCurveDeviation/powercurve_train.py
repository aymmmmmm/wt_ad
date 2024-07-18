# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 11:38
# @File    : powercurve_train.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decimal import *
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from PublicUtils.basic_powercurvefit import power_curve_fit, power_curve_predict


ba = "pitch_1"  # 叶片角度变量名

maxgp = 3200  # 额定功率
rs_lower = 3
wsin = 3.0  # 切入风速
ba_limit = 3  # 叶片停机待机桨叶角度限值
gp_percent_limit = 0.95  # 限功率剔除的功率曲线限值百分比

ws = "wind_speed"  # 风速变量名
gp = "power"  # 有功功率变量名
gs = "generator_rotating_speed"  # 转速变量名
ts = "time"  # 时间变量名
gs= 'rotor_speed'
turbineName = 'turbineName'

reso = 600  # 时间分辨率
bw = 0.5  # 拟合bin大小

airdensity_calibrate = 'False'  # 是否生成使用空气密度校正的功率曲线
airdensity = 1.225  # 现场空气密度
cut_off_mode = 'False'  # 使用截断功率曲线方法拟合


ba_upper = 90
gp_lower = 10

bin_num_limit = 50  # 功率曲线拟合bin最小数量
has_wtstatus = 'False'
wtstatus = 'main_status'
wtstatus_n = 1




def power_curve_fit(data, binwidth, windspeed, power, mode='dataframe'):
    """
    常规风功率曲线拟合数据筛选
    :param data: 原始数据
    :param binwidth: 步长
    :param windspeed: ws列名
    :param power: power列名
    :param mode: 存储数据的格式，默认dataframe
    :return: 用于拟合的风速和有功功率数据
    """
    if mode == 'dataframe':
        power_curve_data = pd.DataFrame(columns=['wsbin', 'power'])
    elif mode == 'dict':
        power_curve_data = dict()
        wsbin_ = []
        power_ = []
    else:
        power_curve_data = pd.DataFrame(columns=['wsbin', 'power'])
    # 生成指定数目的等间隔数组，包含30, 30不能改
    wsbin = np.linspace(0, 30, int(30 / binwidth) + 1) - 0.25
    for i in range(len(wsbin) - 2):
        # 选取风速在等间隔数组两个值之内的功率数据
        slice_data = data.loc[  (wsbin[i] <= data[windspeed].astype('float')) & (data[windspeed].astype('float') < wsbin[i + 1]), power]
        if len(slice_data) > 0:
            if len(slice_data) > 1:
                # 取中位数
                # 改成【25%，75%】之间的数取均值  jimmy -20221020
                gp_lower = np.nanpercentile(slice_data.astype('float'), 25)
                gp_upper = np.nanpercentile(slice_data.astype('float'), 75)
                slice_data = pd.DataFrame(slice_data)
                slice_data = slice_data.loc[(slice_data[power] >= float(gp_lower)) & (slice_data[power] <= float(gp_upper)), :]
                slice_gp = float(np.mean(slice_data.astype('float')))
            else:
                slice_gp = slice_data.astype('float').values[0]
            WSbin = float(np.mean([wsbin[i], wsbin[i + 1]]))
            # WSbin = wsbin[i]
            # 将ws和power加入dataframe
            if mode == 'dataframe':
                temp = {'wsbin': WSbin, 'power': slice_gp}
                temp = pd.DataFrame(temp, index=['0'])
                power_curve_data = power_curve_data.append(temp, ignore_index=True)
            else:
                wsbin_.append(WSbin)
                power_.append(slice_gp)
    if mode == 'dict':
        power_curve_data['wsbin'] = wsbin_
        power_curve_data['power'] = power_
    return power_curve_data


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
        all_data_10min[turbineName] = wt_id
        # all_data_10min[rs] = all_data_10min[gs] / 106.87
        all_data_10min[ts] = pd.to_datetime(all_data_10min[ts])
        all_data_10min = all_data_10min.drop_duplicates([ts])
        vars_10min = all_data_10min.columns.to_list()

        train = all_data_10min[:int(len(all_data_10min)*0.4)]

        # plt.subplot(2, 3, 1)
        # plt.scatter(train[ws], train[gp], s=1)
        # plt.subplot(2, 3, 2)
        # plt.scatter(train[ws], train[rs], s=1)
        # plt.subplot(2, 3, 3)
        # plt.scatter(train[ws], train[ba], s=1)
        # plt.subplot(2, 3, 4)
        # plt.scatter(train[rs], train[gp], s=1)
        # plt.subplot(2, 3, 5)
        # plt.scatter(train[gp], train[ba], s=1)
        # plt.show()

        train = train.set_index(ts, drop=True).sort_index()
        train.dropna(subset=[ws, gp], inplace=True)
        for i in train.columns:
            if i == gs or i == ba:
                train[i] = train[i].interpolate(method='linear')
                train[i] = train[i].fillna(method='backfill')
                train[i] = train[i].fillna(method='ffill')

        train['label'] = 0
        train.loc[(train[gs] > rs_lower)
                  & (train[ws] > wsin)
                  & (train[gp] > gp_lower)
                  & (((train[gp] < maxgp * gp_percent_limit) & (train[ba] < ba_limit))
                     | ((train[gp] >= maxgp * gp_percent_limit) & (train[ba] <= ba_upper)))
                        , 'label'] = 1


        print ('All Train Data_Raw = ', len(train), '   useful Data_Raw=', len(train[train['label'] == 1]))


        train['torque'] = train[gp] / train[ws]
        plot_test = train[train['label'] == 1]
        plot_test_2 = train[train['label'] == 0]


        # plt.subplot(2,3,1)
        # plt.scatter(plot_test[ws], plot_test[gp], c='b', s=2)
        # plt.scatter(plot_test_2[ws], plot_test_2[gp], c='red', s=2)
        # plt.subplot(2,3,2)
        # plt.scatter(plot_test[ws], plot_test[ba], c='b', s=2)
        # plt.scatter(plot_test_2[ws], plot_test_2[ba], c='red', s=2)
        # plt.subplot(2,3,3)
        # plt.scatter(plot_test[ws], plot_test[rs], c='b', s=2)
        # plt.scatter(plot_test_2[ws], plot_test_2[rs], c='red', s=2)
        # plt.subplot(2,3,4)
        # plt.scatter(plot_test[rs], plot_test[gp], c='b', s=2)
        # plt.scatter(plot_test_2[rs], plot_test_2[gp], c='red', s=2)
        # plt.subplot(2,3,5)
        #
        # plt.subplot(2,3,6)
        # plt.scatter(plot_test[rs], plot_test['torque'], c='b', s=2)
        # plt.scatter(plot_test_2[rs], plot_test_2['torque'], c='red', s=2)
        # plt.legend()
        # plt.show()
        # pass

        train = train.loc[train['label'] == 1, [ws, gp]]

        apc = power_curve_fit(train, bw, windspeed=ws, power=gp, mode='dict')

        apc = pd.DataFrame(apc)
        apc = apc.rename(columns={'power': 'GP', 'wsbin': 'WSbin'})

        pc_file_path ='..\\Resource\\power_curve_deviation\\'
        if not os.path.exists(pc_file_path):
            os.makedirs(pc_file_path)


        apc.to_csv(pc_file_path + wt_id+'_pc.csv' , encoding='gbk')

        plt.scatter(train[ws], train[gp], s=1)
        plt.plot(apc['WSbin'], apc['GP'], color='red', linewidth=2)

        plt.savefig(pc_file_path + wt_id + '.jpg')
        # plt.show()

        plt.close()

        pass







