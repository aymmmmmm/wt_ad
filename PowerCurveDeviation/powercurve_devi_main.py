#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @datetime :2021/08/22
# @author : Zf YU
# @version : 2.0.0

"""
风功率曲线偏移
"""
import matplotlib.pyplot as plt
import pandas as pd
from PublicUtils.basic_powercurvefit import power_curve_fit, power_curve_predict
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import shapiro
from decimal import *
import pickle
import os
import os.path


maxgp = 2000  # 额定功率
ba = "pitch_1"  # 叶片角度变量名
ba_limit = 10  # 叶片停机待机桨叶角度限值


ws = "wind_speed"  # 风速变量名
gp = "power"  # 有功功率变量名
gs = "generator_rotating_speed"  # 转速变量名
ts = "time"  # 时间变量名

reso = 600  # 时间分辨率
bw = 0.5  # 拟合bin大小
wsin = 3.0  # 切入风速

airdensity_calibrate = 'False'  # 是否生成使用空气密度校正的功率曲线
airdensity = 1.225  # 现场空气密度
cut_off_mode = 'False'  # 使用截断功率曲线方法拟合
rs_lower = 5
ba_upper = 90
gp_lower = 10
gp_percent_limit = 0.95  # 限功率剔除的功率曲线限值百分比
bin_num_limit = 50  # 功率曲线拟合bin最小数量
has_wtstatus = 'False'
wtstatus = 'main_status'
wtstatus_n = 1



def distance_transform(value, x, y):

    if len(x) == len(y):
        lenx = len(x)
        if value < x[0]:
            value_t = y[0]  ##这里应该是y[0] jimmy- 20221109
        elif (value >= x[0]) & (value < x[lenx - 1]):
            itpl = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            value_t = float(itpl(value))
        else:
            value_t = y[len(y)-1]  ##这里应该是y[len(y)-1] jimmy- 20221109
        return value_t
    else:
        raise Exception

###这里应该是80开始 jimmy -20221026
def rank_transform(distance):
    if distance > 80:
        rank = 0
    elif 60 < distance <= 80:
        rank = 1
    elif 40 < distance <= 60:
        rank = 2
    elif distance <= 40:
        rank = 3
    else:
        rank = np.nan
    return rank


def alarm_integrate(alarm1, alarm2):
    alarm = 0

    if alarm1 > 0:
        if alarm2 > 0:
            alarm = int(np.nanmax([alarm1, alarm2]))
    else:
        if alarm2 > 0:
            alarm = int(np.nanmax([alarm1, alarm2]))
    return alarm


def cutoff_pc_fit(data, ws='ws', gp='gp', binsize=0.5, min_data_size=50, wsin=3.0, wsout=25, mode='dict'):
    bin = int((wsout - wsin) / binsize + 1)
    wsbins = np.linspace(wsin, wsout, bin)
    gp_bin = list()
    for wsbin in wsbins:
        if len(data[(data[ws] >= wsbin - binsize / 2) & (data[ws] < wsbin + binsize / 2)]) > min_data_size:
            slice_data = data.loc[(wsbin - binsize / 2 <= data[ws].astype('float')) & (
                        data[ws].astype('float') < wsbin + binsize / 2), gp]
            # slice_data = slice_data.rename(columns={gp})
            # slice_data = slice_data.reset_index()

            gp_lower1 = np.nanpercentile(slice_data.astype('float'), 25)
            gp_upper1 = np.nanpercentile(slice_data.astype('float'), 75)
            # print(gp_upper1)
            # print(gp_lower1)
            # print(type(gp_lower1))
            slice_data = pd.DataFrame(slice_data)
            # print(slice_data)
            # print(type(slice_data))
            slice_data = slice_data.loc[(slice_data[gp] > float(gp_lower1)) & (slice_data[gp] < float(gp_upper1)), :]
            data_mean = float(slice_data[gp].mean())
            # print(1111111111111)
            # print(data_mean)
            # print(type(data_mean))
            # print(np.mean(slice_data.astype('float')))
            gp_bin.append(data_mean)
            # gp_bin.append(float(np.mean(slice_data.astype('float'))))
        else:
            gp_bin.append(np.nan)
    print('*' * 100)
    print(gp_bin)
    # print(type(gp_bin[0]))
    apc = pd.DataFrame({'wsbin': wsbins, 'power': gp_bin})
    apc = apc.dropna()
    print(ws + '截断拟合后的数据大小', len(apc))
    # print(apc)
    if len(apc) > 0:
        # 最大的有足够数据的风速
        stop_line = max(apc.loc[:, 'wsbin'])
    else:
        stop_line = None
    print(ws + '风速的停止线是', stop_line)
    if mode == 'dict':
        apc_ = dict()
        apc_['wsbin'] = apc['wsbin'].values.tolist()
        apc_['power'] = apc['power'].values.tolist()
        apc = apc_
    else:
        apc = apc
    return apc, stop_line


def pc_fit(data, ts, ws, gp, rs='rs', ba='ba', bw=0.5, wsin=3, maxgp=2000, airdensity_calibrate='False',
           airdensity=1.225,
           cut_off_mode='False', ba_limit=5, gp_percent_limit=0.95, bin_num_limit=50, has_wtstatus=has_wtstatus,
           wtstatus=wtstatus, wtstatus_n=wtstatus_n):
    """

    :param data:
    :param dt:
    :param ws:
    :param gp:
    :param rs:
    :param ba:
    :param bw:曲线拟合步长
    :param wsin:
    :param maxgp:
    :param airdensity_calibrate:是否生成使用空气密度校正的功率曲线
    :param airdensity:现场空气密度
    :param cut_off_mode:是否使用截断功率曲线方法拟合
    :param ba_limit:
    :param gp_percent_limit:
    :param bin_num_limit:
    :param has_wtstatus:
    :param wtstatus:
    :param wtstatus_n:
    :return:
    """
    data[ts] = pd.to_datetime(data[ts].values)
    start_time = str(min(data[ts]))
    end_time = str(max(data[ts]))
    data['label'] = 0
    if has_wtstatus == 'True':
        # data.loc[(data[ws] > 0) & (data[wtstatus] == wtstatus_n), 'label'] = 1
        data.loc[(data[ws] > wsin), 'label'] = 1
    elif ba in data.columns:
        data.loc[   (data[rs] > rs_lower)
                    & (data[ws] > wsin)
                    & (data[gp] > gp_lower)
                    & ( ((data[gp] < maxgp * gp_percent_limit) & (data[ba] < ba_limit))
                        | ( (data[gp] >= maxgp * gp_percent_limit) & (data[ba] <= ba_upper)))
                , 'label'] = 1
    else:
        data.loc[(data[ws] > wsin) & (data[gp] > gp_lower), 'label'] = 1

    # 取符合条件的数据
    apc_data1 = data.loc[data['label'] == 1, [ws, gp]]
    print('第一次数据筛选后的数据大小', len(apc_data1))
    if len(apc_data1) > 0:
        # 满足条件的数据
        plot_data1 = dict()
        plot_data1['ws'] = list(            map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))),                data.loc[data['label'] == 1, ws].values.tolist()))
        plot_data1['gp'] = list(            map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))),                data.loc[data['label'] == 1, gp].values.tolist()))
        # 不满足条件的数据
        plot_data2 = dict()
        plot_data2['ws'] = list(            map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))),                data.loc[data['label'] == 0, ws].values.tolist()))
        plot_data2['gp'] = list(            map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))),                data.loc[data['label'] == 0, gp].values.tolist()))
        status_code = '000'
        # 是否使用截断功率曲线方法拟合
        if cut_off_mode == 'False':
            # 是否生成使用空气密度校正的功率曲线
            if airdensity_calibrate == 'True':
                apc_data1['wind_speed_correct'] = apc_data1[ws].values * (airdensity / 1.225) ** (1 / 3)
                # 返回筛选后的ws和gp
                apc_ad = power_curve_fit(apc_data1, bw, windspeed='wind_speed_correct', power=gp, mode='dict')
            else:
                apc_ad = dict()
            apc = power_curve_fit(apc_data1, bw, windspeed=ws, power=gp, mode='dict')
            ###### 新增一个判断 jimmy - 20221102
            if len(apc)==0:
                status_code = '300'
            elif len(apc['wsbin']) <3:
                status_code = '300'

            sl = None
        else:
            # 使用截断功率曲线方法拟合
            # 是否生成使用空气密度校正的功率曲线
            if airdensity_calibrate == 'True':
                apc_data1['wind_speed_correct'] = apc_data1[ws].values * (airdensity / 1.225) ** (1 / 3)
                apc, sl = cutoff_pc_fit(apc_data1, ws=ws, gp=gp, binsize=0.5, min_data_size=bin_num_limit, wsin=wsin,  wsout=25)
                apc_ad, st_ad = cutoff_pc_fit(apc_data1, ws='wind_speed_correct', gp=gp, min_data_size=bin_num_limit, wsin=wsin, wsout=25)
            else:
                apc, sl = cutoff_pc_fit(apc_data1, ws=ws, gp=gp, min_data_size=bin_num_limit, wsin=wsin, wsout=25)
                apc_ad, st_ad = dict(), None
    else:
        status_code = '300'
        apc, apc_ad, plot_data1, plot_data2 = dict(), dict(), dict(), dict()
        sl = None
    return apc, apc_ad, plot_data1, plot_data2, sl, start_time, end_time, status_code


def pc_devi(data,  pc_file_path ):
    print('原始数据大小是', len(data))
    # if 'csv' not in pc_file_path:
    #     pc_file_path=pc_file_path+'.csv'

    if len(data) > 0:
        apc, apc_ad, plot_data1, plot_data2, sl, st, et, scode = pc_fit(data, ts, ws, gp, rs=gs, ba=ba, bw=bw,
                                                                        wsin=wsin,
                                                                        maxgp=maxgp,
                                                                        airdensity_calibrate=airdensity_calibrate,
                                                                        airdensity=airdensity,
                                                                        cut_off_mode=cut_off_mode, ba_limit=ba_limit,
                                                                        gp_percent_limit=gp_percent_limit,
                                                                        bin_num_limit=bin_num_limit,
                                                                        has_wtstatus=has_wtstatus,
                                                                        wtstatus=wtstatus, wtstatus_n=wtstatus_n)
        # 正常情况下的风速/功率曲线
        bpc_df = pd.read_csv(pc_file_path)
        bpc_df = bpc_df.loc[:, ['GP', 'WSbin']]
        bpc_df = bpc_df.dropna()

        raw_data = dict()
        raw_data['datetime'] = list(map(str, pd.to_datetime(data[ts].values.tolist())))
        raw_data['ws'] = list(            map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))),                data[ws].values.tolist()))
        raw_data['gp'] = list(            map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))),                data[gp].values.tolist()))
        raw_data['bpc'] = {'wsbin': list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0'))), bpc_df['WSbin'].values.tolist())),
                           'power': list(  map(lambda x: float(Decimal(x).quantize(Decimal('0.000'))), bpc_df['GP'].values.tolist()))}
        raw_data['apc'] = apc
        raw_data['plot_data1'] = plot_data1
        raw_data['plot_data2'] = plot_data2
        raw_data['stop_line'] = sl
        # if (scode == '000') & (len(apc['wsbin']) > 2):
        # if (scode == '000') & (len(apc['wsbin']) > 2): ##'000'时长度一定大于2
        if (scode == '000'):
            print('scode=000')
            print('现有曲线的数据大小大于2')
            sum_apc_gp = 0
            sum_bpc_gp = 0
            # 是否生成使用空气密度校正的功率曲线
            if airdensity_calibrate == 'True':
                apc_df = pd.DataFrame({'WSbin': apc_ad['wsbin'], 'GP': apc_ad['power']})
            else:
                apc_df = pd.DataFrame({'WSbin': apc['wsbin'], 'GP': apc['power']})
                apc_df = apc_df.dropna()




            aws = data.loc[data[ws] >= wsin, ws].values.astype('float')
            agp = np.nansum(data.loc[data[gp] >= 0, gp].astype('float')) * (reso / 3600)  # 实际发电量   ###发电量的计算用功率的累加，这个要改掉  jimmy-20221020
            # 潜在发电量是按照标准的功率曲线,根据风速，预测出功率，然后累加
            pgp = np.nansum(power_curve_predict(aws, bpc_df['WSbin'].values, bpc_df['GP'].values)) * (reso / 3600)  # 潜在发电量
            # (实际电量--拟合功率曲线计算)是按照本数据拟合的功率曲线,,根据风速，预测出功率，然后累加
            sgp = np.nansum(power_curve_predict(aws, apc_df['WSbin'].values, apc_df['GP'].values)) * (reso / 3600)  # 实际电量--拟合功率曲线计算--多此一举  （改成了np.nansum())jimmy-20221020
            pba_raw = agp / pgp  # 实际发电量与潜在发电量的比值
            ploss = pgp - sgp  # 潜在发电量与实际电量--拟合功率曲线计算之差
            print('实际发电量与潜在发电量的比值', pba_raw)
            print('潜在发电量与应发电量的差', ploss)
            power_dev = list()
            wsbin_dev = list()

            # 此步骤是为了计算在ws存在下，现有曲线与基准曲线功率的比值
            for vgp, wsbin in bpc_df.values:
                # 如果存在新功率曲线ws等于原有功率曲线
                if len(apc_df.loc[apc_df.WSbin == wsbin, :]) > 0:
                    sum_bpc_gp += vgp  # 基准曲线的gp的和
                    sum_apc_gp += apc_df.loc[apc_df.WSbin == wsbin, 'GP'].values[0]  # 现在曲线的gp的和

                    wsbin_dev.append(float(wsbin))
                    # 在当前ws下，（基准功率-现有曲线功率）/基准功率
                    power_dev.append(float(  Decimal((vgp - apc_df.loc[apc_df.WSbin == wsbin, 'GP'].values[0]) / vgp).quantize(  Decimal('0.0000'))))
            print('ws对应的基准曲线gp的和', sum_bpc_gp)
            print('ws对应的现有曲线gp的和', sum_apc_gp)
            # 在ws存在下，基准曲线与现有曲线功率和的差与基准功率和的比值，即平均功率偏差
            mean_pc_devi = 1 - float(Decimal((sum_bpc_gp - sum_apc_gp) / sum_bpc_gp).quantize(Decimal('0.0000')))
            pcd_detail = dict({'wsbin': wsbin_dev, 'gpdev': power_dev})

            ###用gpdev_mean做判断
            if len (power_dev) > 10:
                gpdev_mean = np.mean(power_dev)
            else:
                gpdev_mean = 0

            print('在ws存在下，（基准功率-现有曲线功率）/基准功率', gpdev_mean)

            breaks = np.arange(5, 11, 0.5)
            data['wg'] = pd.cut(data[ws], breaks)
            # print('cut 后的data[wg]', data['wg'])
            # np.unique去除数组中的重复数字，并进行排序之后输出
            wglist = np.unique(data['wg'].dropna())
            norm_p = list()
            for wg in wglist:
                slice_gp = data.loc[data.wg == wg, gp].values
                if len(slice_gp) >= 10:
                    norm_p.append(shapiro(slice_gp)[0])
            norm_p_mean = np.nanmean(norm_p)
            print('统计参数的平均值', norm_p_mean)
            if norm_p_mean > 0.05:
                norm_mark = 0
            else:
                norm_mark = 1
            # # 现有曲线和基准曲线的距离;;# 在ws存在下，（基准曲线功率累加 - 实际曲线功率累加）/ 基准曲线功率累加
            # distance1 = 100 - distance_transform(1 - mean_pc_devi, [0, 0.9, 0.95, 0.97, 1], [0, 40, 60, 80, 100])   ###映射得不对 Jimmy 20221102
            # # 实际功率累加 /  按照标准的功率曲线,根据风速预测出的功率，然后累加
            # distance2 = 100 - distance_transform(1 - pba_raw, [0, 0.9, 0.95, 0.97, 1], [0, 40, 60, 80, 100])   ###映射得不对 Jimmy 20221102

            distance1 = distance_transform(gpdev_mean, [0, 0.15, 0.3, 0.4, 1], [100, 80, 60, 40, 0])  ### gpdev_mean:各风速下功率下降幅度的平均值  Jimmy 20221102
            distance2= distance1  ## 原来的distance2 没有道理，以distance1 为准 ---jimmy 20221102
            if distance2 <10:
                a=1


            alarm1 = rank_transform(distance1)
            alarm2 = rank_transform(distance2)

            alarm = alarm_integrate(alarm1, alarm2)
            print('distance1=', distance1)
            print('distance2=', distance2)
            print('alarm1=', alarm1, '\t', 'alarm2=', alarm2, '\t', 'alarm=', alarm)
            # pba是实际发电量与潜在发电量的百分比值
            if pba_raw > 1:
                pba = 100
            else:
                pba = pba_raw * 100

            # pindex是在ws存在下，现有曲线与基准曲线功率和的比值
            if mean_pc_devi > 1:
                pindex = 100
            else:
                pindex = mean_pc_devi * 100

            pba = float(Decimal(pba).quantize(Decimal('0.00')))
            pba_raw = float(Decimal(pba_raw).quantize(Decimal('0.00')))
            pindex = float(Decimal(pindex).quantize(Decimal('0.00')))
            ploss = float(Decimal(ploss).quantize(Decimal('0.00')))
            agp = float(Decimal(agp).quantize(Decimal('0.00')))
            pgp = float(Decimal(pgp).quantize(Decimal('0.00')))
        else:
            scode = '301'
            pindex = None
            pba = None
            pba_raw = None
            distance1 = None
            distance2 = None
            alarm = None
            mean_pc_devi = None
            ploss = None
            norm_mark = None
            agp = None
            pgp = None
            apc_df = None
            pcd_detail = dict()

        analysis_data = dict()
        analysis_data['pba'] = pba
        raw_data['pba_raw'] = pba_raw
        analysis_data['pindex'] = pindex
        analysis_data['bpc'] = {'wsbin': list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.0'))), bpc_df['WSbin'].values.tolist())),
                                'power': list(   map(lambda x: float(Decimal(x).quantize(Decimal('0.000'))), bpc_df['GP'].values.tolist()))}
        analysis_data['apc'] = apc
        raw_data['mean_pc_devi'] = mean_pc_devi
        raw_data['mws'] = float(Decimal(np.nanmean(data.loc[data[ws] >= 0, ws])).quantize(Decimal('0.00')))
        raw_data['pcd_detail'] = pcd_detail
        raw_data['ploss'] = ploss
        raw_data['norm_test'] = norm_mark
        raw_data['agp'] = agp
        raw_data['pgp'] = pgp
        raw_data['apc_df'] = apc_df

        status_code = scode
        result = dict()
        result['start_time'] = st
        result['end_time'] = et
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data

        if (distance1 is not None) and (distance2 is not None):
            result['distance'] = min(distance2, distance1)
        else:
            result['distance'] = None

        # result['distance1'] = distance1
        # result['distance2'] = distance2
        result['alarm'] = alarm
        result['status_code'] = status_code
    else:
        status_code = '302'
        result = dict()
        raw_data = dict()
        analysis_data = dict()

        result['start_time'] = None
        result['end_time'] = None
        result['raw_data'] = raw_data
        result['analysis_data'] = analysis_data
        result['distance1'] = None
        result['distance2'] = None
        result['distance'] = None
        result['alarm'] = None
        result['status_code'] = status_code


    return result


def pcd_main(data, pc_file_path, wt_id, result_path):
    global ws
    global gp
    global ts
    global gs
    global ba
    global reso
    global bw
    global wsin
    global maxgp
    global airdensity_calibrate
    global airdensity
    global cut_off_mode
    global ba_limit
    global gp_percent_limit
    global bin_num_limit
    global has_wtstatus
    global wtstatus
    global wtstatus_n
    global rs_lower
    global ba_upper
    global gp_lower
    
    
    result=pc_devi(data, pc_file_path )


    if result['status_code'] == '000' or result['status_code'] == '100':
        start_time = str(data.loc[0, ts])
        if result['distance'] < 80:
            plt.scatter(result['raw_data']['plot_data1']['ws'], result['raw_data']['plot_data1']['gp'], s =1, label='实际散点')
            plt.plot(result['analysis_data']['bpc']['wsbin'],result['analysis_data']['bpc']['power'], color='red' , label='基线功率')
            plt.plot(result['analysis_data']['apc']['wsbin'], result['analysis_data']['apc']['power'], color='black', label='实际功率')
            plt.title('distance=' + str(result['distance']) )
            plt.legend()

            savePath = os.path.join(result_path , 'fault/')
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            plt.savefig( savePath + wt_id + '_'+ start_time[:10] +'.jpg', dpi=plt.gcf().dpi, bbox_inches='tight'  )
            plt.clf()

    return result['start_time'],result['end_time'],result['raw_data'],result['analysis_data'],result['status_code'],result['alarm'],result['distance']




if __name__ == '__main__':
    # standard_pc_path = r'C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\功率异常偏移\01_Git\ModelTrain_SourceCode\Resource\W085_pc.csv'
    # path2 = r'D:\Users\Administrator\Desktop\PowerCurveDeviation-85\PowerCurveDeviation-85-data.json'

    standard_pc_path = r'C:\Project\07_DaTang_BladeProtector\Algorithm\100_打包\功率异常偏移\01_Git\ModelTrain_SourceCode\Resource\W056_pc.csv'
    path2 = r'D:\Users\Administrator\Desktop\PowerCurveDeviation-56\PowerCurveDeviation-56-data.json'
    input_data = pd.read_json(path2, encoding='utf-8')
    result = pcd_main(input_data, standard_pc_path)

    pass








#
#     ################# test
#
#     nametime = ['2021-07']
#     alarm = []
#     distance = []
#     status_code = []
#     for nt in nametime:
#
#         train = pd.read_csv(data_path+'\\data\\' + nt + '_DTSHJK-BHFDC-Q2-W056.csv')
#         result = pcd_main(train)
#
#         with open(data_path+'\\result\\' + nt + '_' + 'result.pkl', 'wb') as f:  # 写文件
#             pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
#
#         curr_result = pd.DataFrame(columns={'pba', 'pba_raw', 'pindex', 'mean_pc_devi', 'ploss', 'norm_test', 'agp', 'pgp'})
#         # curr_result['pba'] = result['pba']
#         # curr_result['pba_raw'] = result['pba_raw']
#         # curr_result['pindex'] = result['pindex']
#         # curr_result['mean_pc_devi'] = result['mean_pc_devi']
#         # curr_result['ploss'] = result['ploss']
#         # curr_result['norm_test'] = result['norm_test']
#         # curr_result['agp'] = result['agp']
#         # curr_result['pgp'] = result['pgp']
#         print(nt)
#         print(result['distance'])
#         print(result['alarm'])
#         print(result['status_code'])
#         alarm.append(result['alarm'])
#         distance.append(result['distance'])
#         status_code.append(result['status_code'])
#         raw_data = result['raw_data']
#         plotdata1 = raw_data['plot_data1']
#         plotdata2 = raw_data['plot_data2']
#         plt.close()
#         plt.scatter(plotdata1['ws'], plotdata1['gp'], c='b', marker='*', label='useful data')
#         plt.scatter(plotdata2['ws'], plotdata2['gp'], c='r', label='useless data')
#         plt.legend(loc='upper left')
#         plt.xlabel('wind speed')
#         plt.ylabel('active power')
#         plt.xlim(0, 30)
#         plt.ylim(0, 3600)
#         plt.title('wind speed and active power' + '(' + nt + ')')
#         plt.savefig('.\\ResultFigure\\' + nt + '_data.png')
#         # plt.show()
#         plt.close()
#
#
#         # apc_df = raw_data['apc_df']
#         # plt.plot(apc_df['WSbin'], apc_df['GP'], c='r', label='current curve')
#         # plt.plot(apc['WSbin'], apc['GP'], c='b', label='previous curve')
#         # plt.legend(loc='upper left')
#         # plt.xlabel('wind speed')
#         # plt.ylabel('active power')
#         # plt.xlim(0, 30)
#         # plt.ylim(0, 4000)
#         # plt.title('previous and current curve' + '(' + nt + ')')
#         # plt.savefig('.\\ResultFigure\\' + nt + '_curve.png')
#         # plt.show()
#
#
#         # result = pd.DataFrame(result)
#         # result.to_csv('.\\result\\' + nt +'_result.csv')
#     print(alarm)
#     print(distance)
#     print(status_code)
#     plt.close()
#     plt.plot(nametime, alarm, c='b', marker='*')
#     plt.ylim(-1, 4)
#     plt.xlabel('month')
#     plt.ylabel('alarm')
#     plt.title('alarm of test data')
#     plt.savefig('.\\ResultFigure\\alarm.png')
#
#     plt.close()
#     plt.plot(nametime, distance, c='b', label='function decline distance', marker='*')
#     plt.legend(loc='lower left')
#     plt.ylim(0, 110)
#     plt.xlabel('month')
#     plt.ylabel('healthy')
#     plt.title('healthy of test data')
#     plt.savefig(data_path+'\\ResultFigure\\distance.png')
    
