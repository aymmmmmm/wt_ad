# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


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
        if len(slice_data) > 100: ###数据长度要足够 20221026 jimmy
            if len(slice_data) > 100:
                # 取中位数
                # 改成【25%，75%】之间的数取均值  jimmy -20221020
                gp_lower = np.nanpercentile(slice_data.astype('float'), 25)
                gp_upper = np.nanpercentile(slice_data.astype('float'), 75)
                slice_data = pd.DataFrame(slice_data)
                slice_data = slice_data.loc[(slice_data[power] >= float(gp_lower)) & (slice_data[power] <= float(gp_upper)), :]
                slice_gp = float(np.mean(slice_data.astype('float')))
            else:
                slice_gp = slice_data.astype('float').values[0]   ###数据长度不够就取第一个？？ 缺乏代表性
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


def power_curve_predict(data, ws, gp):
    """
    功率曲线预测函数
    使用插值算法计算对应功率曲线模型的输入风速值的功率值
    输入风速数值或向量
    :param data: 输入风速数据
    :param ws: 用于插值计算的数组
    :param gp: 用于插值计算的数组
    :return: 预测的功率的值
    """
    # fill_value='extrapolate' 推断数据外的点
    # 当样本数据变化归因于一个独立的变量时，就使用一维插值；反之样本数据归因于多个独立变量时，使用多维插值。
    itpl = interp1d(ws, gp, bounds_error=False,
                    fill_value='extrapolate')  # bug here
    pred_value = itpl(data)
    return pred_value
