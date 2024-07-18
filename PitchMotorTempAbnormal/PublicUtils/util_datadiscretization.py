#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time :2017/4/24 9:36
# @author : R XIE
"""
数据离散化组件
输入（stdin）：
1, 输入文件，str
2, 变量最小值，float
3, 变量最大值，float
4, 操作变量名，str
输出：
1, 新增离散化列的序列化数据文件，在输入路径目录下
"""
import pandas as pd
import numpy as np
import re


def data_discretization(data, mode='type2', minnum=0, maxnum=1500, binsize=100, digit=2):
    """
    连续数据离散化函数
    :param data: 输入数据，array like
    :param mode: 转化模式，type1应用于数据快速离散化变化，仅适用跨位数的数值离散化，type2应用于更具有自定义性质的离散化
    :param minnum: 应用于type2模式，指定转化数据最小的可能值（其他值应预处理剔除），数值型
    :param maxnum: 应用于type2模式，指定转化数据最大的可能值（其他值应预处理剔除），数值型
    :param binsize: 应用于type2模式，指定bin大小，及整数区间宽度，数值型，建议指定为10,50,100，小数值指定0.1,0.5等，结合实际业务指定
    :param digit: 应用于type1，指定digit为保留的小数点位数，数值型，通常为0，1或2，应用于整数部分指定为-1，-2等
    :return: 返回离散化后的数据，numpy array like
    """
    if isinstance(data.values[0], str):
        data = data.astype('float')

    def to_numeric(num):
        num = str(num)
        if num.find(".") < 0:
            num = int(num)
        else:
            num = float(num)
        return num

    def extract_and_comp(data, value):
        # print(data)
        # print(value)
        # left = to_numeric(re.sub("[\[\]\(\)\s*]", "", value.split(sep=',')[0])) not available in pandas update
        # right = to_numeric(re.sub("[\[\]\(\)\s*]", "", value.split(sep=',')[1]))
        left = to_numeric(value.left)
        right = to_numeric(value.right)
        if data - left >= right - data:
            result = right
        else:
            result = left
        return result

    if mode == 'type1':
        new_var = np.array(list(map(lambda s: round(s, digit), data)))
        new_var = np.array([0 if x == 0.0 or x == -0.0 else x for x in new_var]).astype('U')  # 解决出现-0.0与0.0的状况
    if mode == 'type2':
        # new_var = np.array(list(map(lambda s: re.sub("[a-zA-Z\[\]\(\)\s*]", "", s.split(sep=',')[1]),
        #                             pd.cut(data, np.arange(minnum, maxnum, binsize)).get_values())))
        # value = pd.cut(data, np.arange(minnum, maxnum, binsize)).get_values()  ##改了get_values() -jimmy 20221008
        value = pd.cut(data, np.arange(minnum, maxnum, binsize)).to_numpy()
        new_var = np.array(list(map(extract_and_comp, data, value))).astype('U')
    return new_var


if __name__ == '__main__':
    import sys
    import os
    import pickle

    params = []  # 建立空列表接收参数
    num = 4  # 程序接收参数数量
    for i in range(num):
        line = sys.stdin.readline()
        if not line:
            break
        params.append(line.strip('\n'))
    print('Reading params from stdin')
    try:
        data_path = params[0]
        minnum = float(params[1])
        maxnum = float(params[2])
        binsize = float(params[3])
        var = params[4]
    except:
        data_path = params[0]
        minnum = 0
        maxnum = 1500
        binsize = 100
        var = params[4]

    f = open(data_path, 'rb')
    data = pickle.load(data_path)
    f.close()

    data = data.iloc[(data[var] >= minnum) & (data[var] < maxnum), :]
    var_data = data.loc[:, var]
    var_bin = data_discretization(var_data, mode='type2', minnum=minnum, maxnum=maxnum, binsize=binsize)
    var_bin_name = var + '_bin'
    data[var_bin_name] = var_bin

    if os.path.isfile(data_path):
        op = os.path.dirname(data_path)
        out_folder = op
    else:
        out_folder = data_path
    opf = out_folder + os.sep + 'data.bin'
    f = open(opf, 'wb')
    pickle.dump(data, f)
    f.close()  # 保存后关闭文件连接
