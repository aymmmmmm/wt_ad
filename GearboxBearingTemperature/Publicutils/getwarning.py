# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:42:48 2022

@author: xz.fan
"""
import numpy as np
def get_warning(res_dr, res_ndr, threshold_01, threshold_00, threshold_1, border):
    """
    param res_dr: 驱动端温度残差序列
    param res_ndr: 非驱动端温度残差序列
    param threshold1: 温度残差预警阈值
    param threshold2: 残差偏移预警阈值
    param border: 判断为异常的比例边界，默认为0.2
    return: 预警值，1表示触发预警
    """
    warning_dr = 0
    warning_ndr = 0
    res_dp = res_dr - res_ndr
    res_dn = res_ndr - res_dr
    index_dr_1 = [i for i, e in enumerate(res_dr) if threshold_01 <= e < threshold_00]
    index_dr_2 = [i for i, e in enumerate(res_dp) if e >= threshold_1]
    index_ndr_1 = [i for i, e in enumerate(res_ndr) if threshold_01 <= e < threshold_00]
    index_ndr_2 = [i for i, e in enumerate(res_dn) if e >= threshold_1]
    count_index_dr = [x for x in index_dr_1 if x in index_dr_2]
    count_index_ndr = [x for x in index_ndr_1 if x in index_ndr_2]
    if len(count_index_dr) / len(res_dr) > border or np.sum(res_dr >= threshold_00) / len(res_dr) > border:
        warning_dr = 1
    if len(count_index_ndr) / len(res_ndr) > border or np.sum(res_ndr >= threshold_00) / len(res_ndr) > border:
        warning_ndr = 1

    return warning_dr, warning_ndr