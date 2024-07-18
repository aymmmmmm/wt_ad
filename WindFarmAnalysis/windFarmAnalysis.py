# -*- coding: utf-8 -*-
# @Time    : 2021/6/9 17:06
# @File    : powercurve.py
# @Software: PyCharm


import pandas as pd
from matplotlib.pyplot import MultipleLocator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.mlab as mlab
import scipy
from scipy.stats import norm
from scipy.stats import weibull_min
import math
import openpyxl
import os
import os.path
import seaborn
import warnings
warnings.filterwarnings(action='ignore')

from WindFarmAnalysis.Util_01 import *


plt.rcParams['font.sans-serif'] = ['SimHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False

ratedPower =3200
rotorRadius =130
hubHeight =90
ratedRotorSpeed =13
minRotorSpeed =6
ratedWindSpeed =9.5
# plotNum_H =4
plotNum_W =5
cut_lead=0
neg_turbine_number =0
cut_tail=0
ts= 'time'


ye= 'wind_direction'
ws= 'wind_speed'
gp = 'power'
gs= 'rotor_speed'
vib_x= 'nacelle_acc_X'
vib_y= 'nacelle_acc_Y'
yaw= 'yaw_error'
ts= 'time'
ba= 'pitch_1'
gs= 'generator_rotating_speed'

if __name__ == '__main__':
    # data_path = '_Data/1min'
    # fileNames = []
    # wind_turbine_IDs = []
    # for root, dirs, files in os.walk(data_path, topdown=False):
    #     for name in files:
    #         wind_turbine_IDs.append(name.split('.')[0])
    #         fileNames.append(os.path.join(root, name))
    #
    # for i in range(len(wind_turbine_IDs)):
    #     turbine_name = wind_turbine_IDs[i]
    #     path_10min = '_Data/10min/' + turbine_name + '_10min.csv'
    #     all_data_10min = pd.read_csv(path_10min, encoding='gbk')
    #     all_data_10min[ts] = pd.to_datetime(all_data_10min[ts])
    #     vars_10min = all_data_10min.columns.to_list()
    #
    #     path_1min = '_Data/1min/' + turbine_name + '.csv'
    #     all_data_1min = pd.read_csv(path_1min, encoding='gbk')
    #     all_data_1min[ts] = pd.to_datetime(all_data_1min[ts])
    #     vars_1min = all_data_1min.columns.to_list()

    pre_Dir = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\华润风机数据\\数据\\offline_analysis\\1min\\3.2MW'
    report_Dir = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\20221102_newCode\\result_wind_farm\\'
    list_df_WTs = []
    list_df_WTs =reLoad_ScadaData(pre_Dir,neg_number=neg_turbine_number)# 载入处理成标准格式的数据， 第二个数字表示载入数量=总数-negNumber 不删减

    plotNum_H = math.ceil((len(list_df_WTs)-cut_tail- cut_lead)/5)

    # #### 统计风速、风频分布
    # list_df_WTs= dropNull (list_df_WTs, dropWindSpeed=1, dropPower=0, dropVib=0, dropRotorSpeed=0, dropTemp=0)
    # origin_statas_windSpeed (list_df_WTs, subNumber_H=plotNum_H, subNumber_W=plotNum_W, cut_lead=cut_lead, cut_tail=cut_tail, output_Dir=report_Dir, ifShow=False)
    # #
    # # ######统计风玫瑰图
    # origin_statas_wind_rose(list_df_WTs, subNumber_H=plotNum_H, subNumber_W=plotNum_W, cut_lead=cut_lead, cut_tail=cut_tail, output_Dir=report_Dir, ifShow=False)
    # #
    # # ####统计风速相关性
    # origin_statas_wind_corelation(list_df_WTs, subNumber_H=plotNum_H, subNumber_W=plotNum_W, cut_lead=cut_lead, cut_tail=cut_tail, output_Dir=report_Dir, ifShow=False)
    #
    # #画图
    # list_df_WTs= dropNull (list_df_WTs, dropWindSpeed=0, dropPower=1, dropVib=0, dropRotorSpeed=1, dropTemp=0)
    # # cleanPower(list_df_WTs,R=rotorRadius,low_Cp=0.18,low_power=1900)



    # for i in range(len (list_df_WTs)):
    #     list_df_WTs[i]=list_df_WTs[i][(list_df_WTs[i][bp]>3) & (list_df_WTs[i][ws]<8)]
    #
    ##统计机组运行曲线
    # origin_statas (list_df_WTs, variable_x=ws, variable_y=gp, variable_y2="", power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed, ifpreprocess=1, cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # list_df_WTs= dropNull (list_df_WTs, dropWindSpeed=1, dropPower=1, dropVib=0, dropRotorSpeed=0, dropTemp=0)
    # origin_statas (list_df_WTs, variable_x=ws, variable_y=rs, variable_y2="",  power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed,    ifpreprocess=0,     cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x=rs, variable_y=gp, variable_y2="", power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed,  ifpreprocess=1, cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x=ws, variable_y=ba, variable_y2="", power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed, ifpreprocess=1, cut_lead=cut_lead, cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x=rs, variable_y=ba, variable_y2="", power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed, ifpreprocess=1, cut_lead=cut_lead, cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas(list_df_WTs, variable_x=gp, variable_y=ba, variable_y2="", power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed, ifpreprocess=1, cut_lead=cut_lead, cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x=wd, variable_y=ws, variable_y2="", power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed,   ifpreprocess=0,  cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x=wd, variable_y=gp, variable_y2="", power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed,   ifpreprocess=0,  cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    #
    # cal_TSR(list_df_WTs,rotorRadius)
    # origin_statas (list_df_WTs, variable_x=rs, variable_y='torque_measure', variable_y2="",  power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed,    ifpreprocess=1,     cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=True)
    # # origin_statas (list_df_WTs, variable_x='lambda', variable_y='pitch_1', variable_y2="", power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed,   ifpreprocess=1,  cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=True)
    #
    # pass

####振动分析
    # list_df_WTs =reLoad_ScadaData(pre_Dir,neg_number=neg_turbine_number)# 载入处理成标准格式的数据， 第二个数字表示载入数量=总数-negNumber 不删减
    # list_df_WTs= dropNull (list_df_WTs, dropWindSpeed=0, dropPower=0, dropVib=1, dropRotorSpeed=0, dropTemp=0)
    # origin_statas (list_df_WTs, variable_x=ws, variable_y=vib_x, variable_y2="",     power_lowbound=0, rotorSpeed_lowbound=0, ifpreprocess=0, cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x=ws, variable_y=vib_y, variable_y2="",   power_lowbound=0, rotorSpeed_lowbound=0,  ifpreprocess=0,  cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)

    pass
    # shiftVib (list_df_WTs)
    # # cleanPower(list_df_WTs,R=rotorRadius,low_Cp=0.18,low_power=1500)
    # # list_df_WTs[1]['vib_magn_new']=list_df_WTs[1]['vib_magn_new']-0.002
    # origin_statas (list_df_WTs, variable_x='windSpeed', variable_y='nacelleVib_X_new', variable_y2="nacelleVib_X",     power_lowbound=0, rotorSpeed_lowbound=0, ifpreprocess=0, cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x='windSpeed', variable_y='nacelleVib_Y_new', variable_y2="nacelleVib_Y",   power_lowbound=0, rotorSpeed_lowbound=0, ifpreprocess=0,  cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x='windSpeed', variable_y='vib_magn_new', variable_y2="",   power_lowbound=0, rotorSpeed_lowbound=0, ifpreprocess=0,   cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=True)

    #
    # cal_Vibration(list_df_WTs, output_Dir=report_Dir, bin=0.5, Vin=3, Vout=15,  windType='windSpeed',
    #         clean_low=80, clean_up=90,
    #         cut_lead=cut_lead, cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, ifShow=False)
    # windDirection =list_df_WTs[0][(list_df_WTs[0]['windDirection']<30) &(list_df_WTs[0]['windDirection']>-30)].windDirection
    # seaborn.distplot(windDirection,bins=60)
    # plt.show()


##空气密度修正以后，计算CP
    # list_df_WTs =reLoad_ScadaData(pre_Dir,neg_number=neg_turbine_number)# 载入处理成标准格式的数据， 第二个数字表示载入数量=总数-negNumber 不删减
    # shiftVib (list_df_WTs)
    # cleanPower(list_df_WTs,R=rotorRadius,low_Cp=0.18,low_power=1000)
    list_df_WTs= dropNull (list_df_WTs, dropWindSpeed=0, dropPower=0, dropVib=0, dropRotorSpeed=0, dropTemp=1)
    # origin_statas (list_df_WTs, variable_x='time', variable_y='temperature', variable_y2="",  power_lowbound=-100, rotorSpeed_lowbound=-100, ifpreprocess=0,cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=True)
    # origin_statas (list_df_WTs, variable_x='time', variable_y='pressure', variable_y2="",  power_lowbound=-100, rotorSpeed_lowbound=-100,  ifpreprocess=0,cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x='time', variable_y='humidity', variable_y2="",  power_lowbound=-100, rotorSpeed_lowbound=-100, ifpreprocess=0, cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)


    # #计算空气密度
    cal_Rho(list_df_WTs)
    # origin_statas (list_df_WTs, variable_x='time', variable_y='Rho_standard', variable_y2="",  power_lowbound=-100, rotorSpeed_lowbound=-100, ifpreprocess=0, cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=True)
    # origin_statas (list_df_WTs, variable_x='time', variable_y='Ps', variable_y2="",  power_lowbound=-100, rotorSpeed_lowbound=-100, ifpreprocess=0, cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)

    cal_Cp(list_df_WTs,rotorRadius,yawCorrect=1, rhoCorrect=1)
    origin_statas (list_df_WTs, variable_x=ws, variable_y='Cp', variable_y2="",  power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed, ifpreprocess=1, cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)
    # origin_statas (list_df_WTs, variable_x='windSpeed_yawCorrected', variable_y='Cp_noRhoCorrect', variable_y2="",  power_lowbound=5, rotorSpeed_lowbound=minRotorSpeed*0.8, ifpreprocess=1, cut_lead=cut_lead,cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, output_Dir=report_Dir, ifShow=False)

    # #重新保存一下
    # for i in range(len(list_df_WTs)):
    #     folder_write = report_Dir +'process01'
    #     if not os.path.exists(folder_write):
    #         os.makedirs(folder_write)
    #     filepath_write=  folder_write+r'\\'+ list_df_WTs[i].iloc[0]['turbineName'] + '.csv'
    #     list_df_WTs[i].to_csv(filepath_write, encoding='gb2312')
    #     print('write new', list_df_WTs[i].iloc[0]['turbineName'])

    cal_AEP(list_df_WTs, output_Dir=report_Dir, bin=0.5, Vin=3, Vout=20, R=rotorRadius, Vrate=ratedWindSpeed, Prate=ratedPower,
            V_average=7, weibull_k=2, windType='windSpeed_Standard',bymonth=0,in_date = '2021-01',
            clean_low=25, clean_up=75,
            cut_lead=cut_lead, cut_tail=cut_tail, subNumber_H=plotNum_H, subNumber_W=plotNum_W, ifShow=False)


    # dividedIntoMonths(list_df_WTs, output_Dir=report_Dir, bin=0.5, Vin=3, Vout=20, R=rotorRadius, Vrate=ratedWindSpeed, Prate=ratedPower,
    #         V_average=5, weibull_k=2, windType='windSpeed_Standard',bymonth=1, in_date = '2021-01',
    #         clean_low=25, clean_up=75,
    #         cut_lead=0, cut_tail=0, subNumber_H=plotNum_H, subNumber_W=plotNum_W, ifShow=False,vibExist=False)

