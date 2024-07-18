# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 14:10
# @File    : Scada_offline_analysis_main.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import os
import pickle
import datetime
from dateutil.relativedelta import relativedelta
from decimal import *

from bladeIcing.bladeicing_main import blade_icing_main
from generator_bearing_temp.generator_temp_main import generator_temp_main
from YawMisalignment.yaw_misalignment import yaw_misalignment_main
from generator_bearing_stuck.generator_bearing_stuck import generator_bs_main
from PowerCurveDeviation.powercurve_devi_main import pcd_main
from GearboxBearingTemperature.gearbox_bearing_temperature import gearbox_temp_main
from generator_stator_temp.GeneratorStatorTemp_main import generator_stator_temp_main
from AnemometerAbnormal.anemometer_abnormal_main import anemometer_wsunnormal_main
from ConverterTempAbnormal.converter_temp_abnormal_main import converter_temp_abnormal_main
from IGBT_temp_Abnormal.igbt_temp_abnormal import igbt_abnormal_main
from PitchMotorTempAbnormal.pitch_motor_temp_abnormal import pitch_motor_temp_abnormal_main
from gearbox_oil_temperature.gearbox_oil import gearbox_oil_main
from control_cabinet_temp_abnormal.ControlCabinet_main import ControlCabinetTempAbnormal_main
from pitch_control_abnormal.pitch_control_abnormal_my import pitch_control_abnormal_main
from yaw_drive_abnormal.YawDriveAbnormal import yaw_drive_abnormal_main
ts = 'time'
ba = 'pitch_1'
rs = 'rotor_speed'
gs = 'generator_rotating_speed'
gp = 'power'
ws= 'wind_speed'
ye = 'yaw_error'
wd= 'wind_direction'
turbineName = 'turbineName'
env_temp = 'environment_temp'  # 机舱外环境温度
cabin_temp = 'cabin_temp'
vib_X = 'nacelle_vib_X'
converter_temp = 'converter_temp_gridside'
wtstatus = 'main_status'  # 机组状态变量名
igbt_temp_1 = 'converter_temp_genside'
igbt_temp_2 = 'converter_temp_gridside'

pbt1 = 'pitch_motor_temp_1'
pbt2 = 'pitch_motor_temp_2'
pbt3 = 'pitch_motor_temp_3'
ba1_speed = "pitch_speed_1"
ba2_speed = "pitch_speed_2"
ba3_speed = "pitch_speed_3"
f_gen_b_temp = 'f_i_bearing_temp'
r_gen_b_temp = 'r_i_bearing_temp'
grid_current_A = 'grid_current_A'
gen_winding_temp_U = 'generator_winding_temp_U'
gbox_oil_temp = 'gearbox_oil_temp'
f_gbox_bear_temp = 'f_gearbox_bearing_temp'
r_gbox_bear_temp = 'r_gearbox_bearing_temp'
cct = 'control_carbin_temp'
nac_position = 'nacelle_position'
converter_current_genside = 'grid_current_A'

def get_data(all_data, start_time, end_time, vars ):
    all_data.sort_values(by=ts, inplace=True, ascending=True)
    data = all_data[(all_data[ts] >= start_time) & (all_data[ts] < end_time)][vars]
    data = data.reset_index(drop= True)
    return data
def distance_plot(time, distance, wt_id, savepath):
    plt.plot(pd.to_datetime(time), distance,label= '健康度' )
    plt.plot(pd.to_datetime([time.iloc[0], time.iloc[-1]]),[80, 80], color= 'red', linestyle='--', linewidth= 2, label= 'Alarm' )
    plt.gcf().autofmt_xdate()
    plt.xlabel('时间')
    plt.ylabel('健康度')
    plt.title(wt_id)
    plt.ylim(0, 110)
    plt.legend()
    plt.savefig(savepath, dpi=plt.gcf().dpi, bbox_inches='tight')
    plt.close()

class module_basic(object):
    def __init__ (self, active_frequency, data_duration, vars_input, wt_id ):
        self.vars_input = vars_input
        self.wt_id = wt_id
        self.active_frequency = active_frequency
        self.data_duration = data_duration

    def check_avaliability(self):
        if set(self.vars_require).issubset(set(self.vars_input)):
            return True
        else:
            for var in self.vars_require:
                if var not in self.vars_input:
                    print ('some variable is missing---', var)
            return False

    def batch_generator(self, all_data):
        start_time = all_data[ts].iloc[0]
        end_time = all_data[ts].iloc[-1]
        batch_WT = pd.DataFrame(columns=['Turbine_id','st','et'])
        st= start_time
        while st <end_time:
            time_range = pd.date_range(st, periods=2, freq=self.active_frequency)
            st_thisjob = time_range[0]
            et_thisjob = time_range[1]

            time_range2 = pd.date_range(st, periods=2, freq=self.data_duration)
            data_et_thisjob = time_range2[1]


            this_job = pd.DataFrame({'Turbine_id':self.wt_id, 'st':st_thisjob, 'et':data_et_thisjob  }, index=['0'])
            if data_et_thisjob< end_time:
                #batch_WT = batch_WT.append(this_job,ignore_index=True)
                batch_WT= pd.concat([batch_WT, this_job], axis=0)

            st=et_thisjob
        batch_WT.reset_index(drop=True)
        return batch_WT

class module_basic_compare(object):
    def __init__ (self, active_frequency, data_duration, vars_input, wt_ids ):
        self.vars_input = vars_input
        self.wt_ids = wt_ids
        self.active_frequency = active_frequency
        self.data_duration = data_duration

    def check_avaliability(self):
        if set(self.vars_require).issubset(set(self.vars_input)):
            return True
        else:
            for var in self.vars_require:
                if var not in self.vars_input:
                    print ('some variable is missing---', var)
            return False

    def batch_generator(self, all_data):
        start_time = all_data[ts].iloc[0]
        end_time = all_data[ts].iloc[-1]
        batch_WT = pd.DataFrame(columns=['Turbine_id','st','et'])
        st= start_time
        while st <end_time:
            time_range = pd.date_range(st, periods=2, freq=self.active_frequency)
            st_thisjob = time_range[0]
            et_thisjob = time_range[1]

            time_range2 = pd.date_range(st, periods=2, freq=self.data_duration)
            data_et_thisjob = time_range2[1]


            this_job = pd.DataFrame({ 'st':st_thisjob, 'et':data_et_thisjob  }, index=['0'])
            if data_et_thisjob < end_time:
                batch_WT= batch_WT.append(this_job,ignore_index=True)

            st=et_thisjob
        batch_WT.reset_index(drop=True)
        return batch_WT



class generator_bearing_temp (module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(generator_bearing_temp, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, wtstatus,turbineName, gp, gs, cabin_temp,f_gen_b_temp, r_gen_b_temp]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data,       model_dr,
                                        model_ndr,
                                        warning_dr_history,
                                        warning_ndr_history,
                                        alarm_dr_history,
                                        alarm_ndr_history):
        result_generator_bearing_temp =[]
        for i in range(len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)

            result = {}
            result['raw_data'],result['analysis_data'] ,result['status_code'],\
            result['start_time'],result['end_time'], result['gbt1_distance'],\
            result['gbt2_distance'],result['gbt1_alarm'], result['gbt2_alarm'],\
            result['alarm'], result['distance']= generator_temp_main(sample_data, model_dr, model_ndr,warning_dr_history,warning_ndr_history,alarm_dr_history,alarm_ndr_history)
            result_generator_bearing_temp.append(result)

            print(self.wt_id, result['start_time'], result['gbt1_distance'], result['gbt2_distance'],  result['alarm'], result['distance'])
        return result_generator_bearing_temp
def run_generator_bearing_temp( data, wt_id, vars):
    ###发电机轴承温度异常
    model_dr_path = 'Resource/generator_bearing_temp/dr/{}_generatorBearingTemp_model.bin'.format(wt_id)
    model_ndr_path = 'Resource/generator_bearing_temp/ndr/{}_generatorBearingTemp_model.bin'.format(wt_id)
    try:
        with open(model_dr_path, 'rb') as fp:
            model_dr = pickle.load(fp)
        with open(model_ndr_path, 'rb') as fp:
            model_ndr = pickle.load(fp)
    except:
        raise Exception('No Such File or Directory')

    module_gen_bearing_temp = generator_bearing_temp(active_frequency='24H', data_duration='24H', vars_input=vars, wt_id= wt_id)
    if module_gen_bearing_temp.isavalible:
        module_gen_bearing_temp.batch_list= module_gen_bearing_temp.batch_generator(all_data=data)

    warning_dr_history = []
    warning_ndr_history = []
    alarm_dr_history = []
    alarm_ndr_history = []

    result_gen_bearing_temp = pd.DataFrame(module_gen_bearing_temp.run_batch(all_data=data,
                                                                                    model_dr=model_dr,
                                                                                    model_ndr=model_ndr,
                                                                                    warning_dr_history=warning_dr_history,
                                                                                    warning_ndr_history=warning_ndr_history,
                                                                                    alarm_dr_history=alarm_dr_history,
                                                                                    alarm_ndr_history=alarm_ndr_history))

    #### Plot
    result_path = '../Result/gen_bearing_temp'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath= os.path.join(result_path, wt_id + '_gen_bearing_distance.jpg')

    distance_plot(result_gen_bearing_temp['start_time'], result_gen_bearing_temp['distance'], wt_id, savepath)


    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                         '开始时间': min(data[ts]),
                         '结束时间': max(data[ts]),
                         '运行次数': len(result_gen_bearing_temp),
                         '故障次数': len(result_gen_bearing_temp[result_gen_bearing_temp['alarm'] > 0]),
                         '最小健康度_1': float(Decimal(np.min(result_gen_bearing_temp['gbt1_distance'])).quantize( Decimal('0.00'))),
                         '最小健康度_2': float(Decimal(np.min(result_gen_bearing_temp['gbt2_distance'])).quantize(Decimal('0.00'))),
                         '最严重报警等级': np.max(result_gen_bearing_temp['alarm'])
                          }, index=['0'])
    fault.to_csv(os.path.join('../Result/gen_bearing_temp/', 'gen_bearing_temp_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_gen_bearing_temp

class generator_bearing_stuck (module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(generator_bearing_stuck, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, wtstatus,  turbineName,ws, ye, ba, gp, gs, cabin_temp, f_gen_b_temp, r_gen_b_temp ]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data,  model_dr, model_ndr, ws_gp_model):
        result_generator_bearing_temp =[]
        for i in range(len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)

            result = {}
            result['raw_data'],result['analysis_data'] ,result['status_code'],\
            result['start_time'],result['end_time'],  \
            result['alarm'], result['distance'],\
            result['gbt1_alarm'],  result['gbt2_alarm']= generator_bs_main(sample_data, pd.DataFrame(result_generator_bearing_temp), model_dr, model_ndr, ws_gp_model)
            result_generator_bearing_temp.append(result)

            print(self.wt_id, result['start_time'], result['alarm'], result['distance'], result['gbt1_alarm'],  result['gbt2_alarm'] )
        return result_generator_bearing_temp
def run_generator_bearing_stuck( data, wt_id, vars):
    ###发电机轴承卡滞
    model_dr_path = 'Resource/generator_bearing_temp/dr/{}_generatorBearingTemp_model.bin'.format(wt_id)
    model_ndr_path = 'Resource/generator_bearing_temp/ndr/{}_generatorBearingTemp_model.bin'.format(wt_id)
    ws_gp_model_path = 'Resource/blade_icing/ws_power_model_{}.pkl'.format(wt_id)
    try:
        with open(model_dr_path, 'rb') as fp:
            model_dr = pickle.load(fp)
        with open(model_ndr_path, 'rb') as fp:
            model_ndr = pickle.load(fp)
        with open(ws_gp_model_path, 'rb') as fp:
            ws_gp_model = pickle.load(fp)
    except:
        raise Exception('No Such File or Directory')

    module_gen_bearing_stuck = generator_bearing_stuck(active_frequency='24H', data_duration='24H', vars_input=vars, wt_id= wt_id)

    if module_gen_bearing_stuck.isavalible:
        module_gen_bearing_stuck.batch_list= module_gen_bearing_stuck.batch_generator(all_data=data)
    result_gen_bearing_stuck = pd.DataFrame(module_gen_bearing_stuck.run_batch(all_data=data,
                                                                                    model_dr=model_dr,
                                                                                    model_ndr=model_ndr,
                                                                             ws_gp_model= ws_gp_model))

    #### Plot
    result_path = '../Result/gen_bearing_stuck/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath = os.path.join(result_path, wt_id + '_gen_bearing_stuck_distance.jpg')
    distance_plot(result_gen_bearing_stuck['start_time'], result_gen_bearing_stuck['distance'], wt_id, savepath)

    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                         '开始时间': min(data[ts]),
                         '结束时间': max(data[ts]),
                         '运行次数': len(result_gen_bearing_stuck),
                         '故障次数': len(result_gen_bearing_stuck[result_gen_bearing_stuck['alarm'] > 0]),
                         '最小健康度': np.min(result_gen_bearing_stuck['distance']),
                         '最严重报警等级': np.max(result_gen_bearing_stuck['alarm'])
                          }, index=['0'])
    fault.to_csv(os.path.join('../Result/gen_bearing_stuck/', 'gen_bearing_stuck_summary.csv'), encoding='gbk', mode='a', header=True)

    return result_gen_bearing_stuck

class blade_icing (module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(blade_icing, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, turbineName,wtstatus, ws, gp, gs, env_temp, cabin_temp, ba,vib_X]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data, wt_id, model):

        result_path = '../Result/blade_icing/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        result_blade_icing =[]
        for i in range(len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            result = {}
            result['distance'],result['raw_data'],result['analysis_data'] ,result['status_code'],result['start_time'],result['end_time'], result['alarm']= blade_icing_main(sample_data, model)
            print (wt_id, result['start_time'], result['distance'], result['status_code'], result['alarm'])

            if (result['status_code'] == '000') or (result['status_code'] == '100') or (result['status_code'] == '101') :
                if result['alarm']>0 :
                    plt.scatter(result['raw_data']['wind_speed'], result['raw_data']['active_power'], s=1 , label= '实际功率')
                    power_d = model.predict(pd.DataFrame(np.arange(1, 20, 0.5)))
                    plt.plot(np.arange(1, 20, 0.5), power_d, color="red", linewidth=1, label= '参考功率')
                    plt.xlabel('风速')
                    plt.ylabel('功率')
                    plt.title(wt_id + '_' + result['start_time'])
                    plt.legend()
                    plt.savefig(os.path.join(result_path, wt_id + '_PowerCurve'+result['start_time'].replace(' ', '_').replace(':', '') + '.jpg'))
                    plt.close()
            result_blade_icing.append(result)
        return result_blade_icing
def run_blade_icing(data, wt_id, vars):
    ######## 叶片结冰
    model_wp = 'Resource/blade_icing/ws_power_model_' + wt_id + '.pkl'
    with open(model_wp, 'rb') as f:
        ws_power_model = pickle.load(f)

    module_blade_icing = blade_icing(active_frequency='2H',
                                     data_duration='2H',
                                     vars_input=vars,
                                     wt_id=wt_id,
                                     )
    if module_blade_icing.isavalible:
        module_blade_icing.batch_list = module_blade_icing.batch_generator(all_data= data)
    result_blade_icing = pd.DataFrame(module_blade_icing.run_batch(all_data= data, wt_id= wt_id, model = ws_power_model))

    #### Plot
    result_path = '../Result/blade_icing/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    savepath = os.path.join(result_path, wt_id + '_icing_distance.jpg')
    distance_plot(result_blade_icing['start_time'], result_blade_icing['distance'], wt_id, savepath)

    ####保存异常记录
    fault= pd.DataFrame({'机组ID': wt_id,
                         '开始时间': min(data[ts]),
                         '结束时间': max(data[ts]),
                         '运行次数': len(result_blade_icing),
                         '故障次数': len(result_blade_icing[result_blade_icing['alarm'] > 0]),
                         '最小健康度': float(Decimal(np.min(result_blade_icing['distance'])).quantize( Decimal('0.00'))),
                         '最严重报警等级': np.max(result_blade_icing['alarm'])
                         }, index=['0'])
    fault.to_csv(os.path.join('../Result/blade_icing/', 'blade_icing_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_blade_icing
    ############***************************************************###################

class yaw_misalignment_abnormal(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(yaw_misalignment_abnormal, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, turbineName , ws, gp, gs, ye, ba]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data):
        result_yaw_misalignment=[]
        for i in range(len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            # sample_data.to_csv(r'C:\Project\09_工程项目售前\20_中控技术\PLC智能卡件\YawMisalignment\YawMisalignment_demo.csv', encoding= 'gbk')

            result = {}
            result["start_time"], result["end_time"], result["raw_data"], result["analysis_data"],\
            result["status_code"], result["max_wdir"], result["alarm"], result["distance"], result["rank"]= yaw_misalignment_main(sample_data)

            result_yaw_misalignment.append(result)
            print(self.wt_id, result['start_time'], result['max_wdir'], result['distance'], result['alarm'], result['status_code'], result["rank"])
        return result_yaw_misalignment
def run_yaw_misalignment( data, wt_id, vars):
    ######## 对风不正, 1min数据3天数据量太少，放30天
    module_yaw_misalignment = yaw_misalignment_abnormal(active_frequency='72H', data_duration='720H',
                                     vars_input=vars,
                                     wt_id=wt_id )
    if module_yaw_misalignment.isavalible:
        module_yaw_misalignment.batch_list = module_yaw_misalignment.batch_generator(all_data= data)
    result_yaw_misalignment = pd.DataFrame(module_yaw_misalignment.run_batch(all_data= data))
    #### Plot
    result_path = '../Result/yaw_misalignment/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    savepath = os.path.join(result_path, wt_id + '_yaw_misalignment.jpg')
    distance_plot(result_yaw_misalignment['start_time'], result_yaw_misalignment['distance'], wt_id, savepath)

    min_ye = np.min(result_yaw_misalignment['max_wdir'])
    max_ye = np.max(result_yaw_misalignment['max_wdir'])

    maxCp_ye = max_ye if (abs(max_ye) > abs(min_ye))  else min_ye

    ####保存异常记录
    fault= pd.DataFrame({'机组ID': wt_id,
                         '开始时间': min(data[ts]),
                         '结束时间': max(data[ts]),
                         '运行次数': len(result_yaw_misalignment),
                         '故障次数': len(result_yaw_misalignment[result_yaw_misalignment['alarm'] > 0]),
                         'Cpmax_偏航误差': maxCp_ye,
                         '最小健康度': float(Decimal(np.min(result_yaw_misalignment['distance'])).quantize( Decimal('0.00'))),
                         '最严重报警等级': np.max(result_yaw_misalignment['alarm'])
                         }, index=['0'])
    fault.to_csv(os.path.join('../Result/yaw_misalignment/', 'yaw_misalignment_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_yaw_misalignment

class power_curve_devi(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(power_curve_devi, self).__init__(active_frequency, data_duration, vars_input, wt_id)
        self.vars_require = [ts, turbineName,   ws,  gp,  gs,   ba]#输入变量
        self.isavalible = self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data, pc_file_path, result_path):
        result_power_curve_devi = []
        for i in range(len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            result ={}
            result['start_time'],result['end_time'],\
            result['raw_data'],result['analysis_data'],\
            result['status_code'],result['alarm'],result['distance'] = pcd_main(sample_data, pc_file_path, wt_id=self.wt_id, result_path=result_path)


            result_power_curve_devi.append(result)
            print(self.wt_id, result['start_time'],  result['distance'], result['alarm'],  result['status_code'])
        return result_power_curve_devi
def run_power_curve_devi( data, wt_id, vars):
    ######## 功率曲线偏移, 一天数据

    model_path = 'Resource/power_curve_deviation/{}_pc.csv'.format(wt_id)
    result_path = '../Result/power_curve_devi/'

    module_power_curve_devi = power_curve_devi(active_frequency='24H', data_duration='720H',
                                               vars_input=vars,
                                               wt_id=wt_id)
    if module_power_curve_devi.isavalible:
        module_power_curve_devi.batch_list = module_power_curve_devi.batch_generator(all_data= data)
    result_power_curve_devi = pd.DataFrame(module_power_curve_devi.run_batch(all_data= data, pc_file_path= model_path, result_path =result_path ))
    #### Plot

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # savepath = os.path.join(result_path, wt_id + 'power_curve_devi.jpg')
    savepath = result_path + wt_id + 'power_curve_devi.jpg'
    distance_plot(result_power_curve_devi['start_time'], result_power_curve_devi['distance'], wt_id, savepath)

    ####保存异常记录
    fault= pd.DataFrame({'机组ID': wt_id,
                         '开始时间': min(data[ts]),
                         '结束时间': max(data[ts]),
                         '运行次数': len(result_power_curve_devi),
                         '故障次数': len(result_power_curve_devi[result_power_curve_devi['alarm'] > 0]),
                         '最小健康度': float(Decimal(np.min(result_power_curve_devi['distance'])).quantize( Decimal('0.00'))),
                         '最严重报警等级': np.max(result_power_curve_devi['alarm'])
                         }, index=['0'])
    fault.to_csv(os.path.join('../Result/power_curve_devi/', 'power_curve_devi_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_power_curve_devi


class gearbox_bearing_temp(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(gearbox_bearing_temp, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, turbineName, gp, gs, cabin_temp, gbox_oil_temp, f_gbox_bear_temp, r_gbox_bear_temp]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data, model_dr,   model_ndr,  warning_dr_history,   warning_ndr_history,   alarm_dr_history,    alarm_ndr_history):
        result_gearbox_bearing_temp =[]
        for i in range(len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)

            result = {}
            raw_data,analysis_data,status_code,\
            result['start_time'],result['end_time'],\
            result['gbt1_distance'],result['gbt2_distance'],\
            result['gbt1_alarm'],result['gbt2_alarm'],\
            result['distance'], result['alarm']= gearbox_temp_main(sample_data, model_dr, model_ndr,warning_dr_history,warning_ndr_history,alarm_dr_history,alarm_ndr_history)
            result_gearbox_bearing_temp.append(result)
            print(self.wt_id, result['start_time'], result['gbt1_distance'], result['gbt2_distance'],  result['alarm'], result['distance'])
        return result_gearbox_bearing_temp
def run_gearbox_bearing_temp( data, wt_id, vars):
    ###发电机轴承温度异常
    model_dr_path = 'Resource/gearbox_bearing_temp/{}_modeldr.pkl'.format(wt_id)
    model_ndr_path = 'Resource/gearbox_bearing_temp/{}_modelndr.pkl'.format(wt_id)
    try:
        with open(model_dr_path, 'rb') as fp:
            model_dr = pickle.load(fp)
        with open(model_ndr_path, 'rb') as fp:
            model_ndr = pickle.load(fp)
    except:
        raise Exception('No Such File or Directory')

    module_gearbox_bearing_temp = gearbox_bearing_temp(active_frequency='24H', data_duration='24H', vars_input=vars, wt_id= wt_id)
    if module_gearbox_bearing_temp.isavalible:
        module_gearbox_bearing_temp.batch_list= module_gearbox_bearing_temp.batch_generator(all_data=data)

    warning_dr_history = []
    warning_ndr_history = []
    alarm_dr_history = []
    alarm_ndr_history = []

    result_gearbox_bearing_temp = pd.DataFrame(module_gearbox_bearing_temp.run_batch(all_data=data,
                                                                                    model_dr=model_dr,
                                                                                    model_ndr=model_ndr,
                                                                                    warning_dr_history=warning_dr_history,
                                                                                    warning_ndr_history=warning_ndr_history,
                                                                                    alarm_dr_history=alarm_dr_history,
                                                                                    alarm_ndr_history=alarm_ndr_history))

    #### Plot
    result_path = '../Result/gearbox_bearing_temp'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath= os.path.join(result_path, wt_id + '_gearbox_bearing_distance.jpg')

    distance_plot(result_gearbox_bearing_temp['start_time'], result_gearbox_bearing_temp['distance'], wt_id, savepath)


    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                         '开始时间': min(data[ts]),
                         '结束时间': max(data[ts]),
                         '运行次数': len(result_gearbox_bearing_temp),
                          '故障次数': len(result_gearbox_bearing_temp[result_gearbox_bearing_temp['alarm'] > 0]),
                          '最小健康度_1': float(Decimal(np.min(result_gearbox_bearing_temp['gbt1_distance'])).quantize( Decimal('0.00'))),
                          '最小健康度_2': float(Decimal(np.min(result_gearbox_bearing_temp['gbt1_distance'])).quantize( Decimal('0.00'))),
                          '最严重报警等级': np.max(result_gearbox_bearing_temp['alarm'])
                          }, index=['0'])
    fault.to_csv(os.path.join('../Result/gearbox_bearing_temp/', 'gearbox_bearing_temp_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_gearbox_bearing_temp


class gearbox_oil_temp(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(gearbox_oil_temp, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, turbineName, gp, gs, cabin_temp, gbox_oil_temp, f_gbox_bear_temp, r_gbox_bear_temp]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data, model, base_plot, warning_history, alarm_history):
        result_gearbox_oil_temp =[]
        for i in range(len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)

            result = {}
            result['raw_data'], result['analysis_data'], result['status_code'], result['start_time'], \
            result['end_time'], result['distance'], result['alarm'] \
                = gearbox_oil_main(sample_data, model, base_plot, warning_history, alarm_history)
            result_gearbox_oil_temp.append(result)

            ##plot 故障数据显示
            if result['status_code'] == '000':
                if result['distance'] < 90:

                    plt.subplot(2, 1, 1)
                    plt.plot(result['raw_data']['datetime'], result['raw_data']['oiltemp'], label='温度_实际')
                    plt.plot(result['raw_data']['datetime'], result['raw_data']['pred_y'], label='温度_预测')
                    plt.title(str(result['start_time'])[:-6] +'distance'+ str(result['distance']))
                    plt.legend()
                    plt.gcf().autofmt_xdate()

                    plt.subplot(2, 1, 2)
                    plt.plot(result['analysis_data']['online_x'], result['analysis_data']['online_y'], label='残差')
                    plt.legend()
                    plt.gcf().autofmt_xdate()


                    plt.subplots_adjust(left=0.04, top=0.96, right=0.96, bottom=0.1, wspace=0.3, hspace=0.3)
                    import os
                    savePath = '../Result/gearbox_oil_temp/fault/'
                    if not os.path.exists(savePath):
                        os.makedirs(savePath)
                    plt.savefig(savePath + self.wt_id + str(result['start_time'])[:-6] + 'gearbox_oil.jpg',
                                format='jpg', dpi=plt.gcf().dpi, bbox_inches='tight')
                    plt.clf()


            print(self.wt_id, result['start_time'], result['distance'],  result['alarm'], result['status_code'])
        return result_gearbox_oil_temp
def run_gearbox_oil_temp( data, wt_id, vars):
    ###齿轮箱油温异常
    model_path = 'Resource/gearbox_oil_temp/{}_model.pkl'.format(wt_id)
    residual_path = 'Resource/gearbox_oil_temp/{}_residual.pkl'.format(wt_id)
    try:
        with open(model_path, 'rb') as fp:
            model = pickle.load(fp)
        with open(residual_path, 'rb') as fp:
            residual = pickle.load(fp)
    except:
        raise Exception('No Such File or Directory')

    module_gearbox_oil_temp = gearbox_oil_temp(active_frequency='24H', data_duration='24H', vars_input=vars, wt_id= wt_id)
    if module_gearbox_oil_temp.isavalible:
        module_gearbox_oil_temp.batch_list= module_gearbox_oil_temp.batch_generator(all_data=data)

    warning_history = []
    alarm_history = []

    result_gearbox_oil_temp = pd.DataFrame(module_gearbox_oil_temp.run_batch(all_data=data,
                                                                                    model=model,
                                                                                    base_plot=residual,
                                                                                    warning_history=warning_history,
                                                                                    alarm_history=alarm_history))

    #### Plot
    result_path = '../Result/gearbox_oil_temp'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath= os.path.join(result_path, wt_id + '_gearbox_oil_distance.jpg')

    distance_plot(result_gearbox_oil_temp['start_time'], result_gearbox_oil_temp['distance'], wt_id, savepath)

    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                         '开始时间': min(data[ts]),
                         '结束时间': max(data[ts]),
                         '运行次数': len(result_gearbox_oil_temp),
                          '故障次数': len(result_gearbox_oil_temp[result_gearbox_oil_temp['alarm'] > 0]),
                          '最小健康度': float(Decimal(np.min(result_gearbox_oil_temp['distance'])).quantize( Decimal('0.00'))),
                          '最严重报警等级': np.max(result_gearbox_oil_temp['alarm'])
                          }, index=['0'])
    fault.to_csv(os.path.join('../Result/gearbox_oil_temp/', 'gearbox_oil_temp_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_gearbox_oil_temp


class generator_stator_temp(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(generator_stator_temp, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, turbineName, gp, gs, cabin_temp, grid_current_A, f_gen_b_temp, r_gen_b_temp,gen_winding_temp_U ]#输入变量

        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data,  model, base_plot, warning_history,  alarm_history):
        result_generator_stator_temp =[]
        for i in range(0, len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)

            result = {}
            result['start_time'],result['end_time'], result['raw_data'],result['analysis_data'],result['status_code'], result['distance'], result['alarm']\
                = generator_stator_temp_main(sample_data, model, base_plot,warning_history,alarm_history)

            if result['alarm']> 0:
                # ################Plot 最终结果
                plt.plot(result['analysis_data']['density_x1'], result['analysis_data']['density_y1'],  label='本机组')
                plt.plot(result['analysis_data']['base_x'], result['analysis_data']['base_y'], color='black',label='base')
                plt.xlabel('温度')
                plt.ylabel('概率密度')
                plt.legend()
                plt.title(self.wt_id + '_' + str(result['start_time'])[0:10] + '_distance' + str(result['distance']))
                # plt.show()
                savePath = '../Result/generator_stator_temp/fault/'
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                plt.savefig(savePath + self.wt_id + '_' + str(result['start_time'])[0:10] + 'generator_stator_temp.jpg',   format='jpg', dpi=plt.gcf().dpi, bbox_inches='tight')
                plt.clf()
            result_generator_stator_temp.append(result)

            print(self.wt_id, result['start_time'], result['status_code'], result['distance'], result['alarm'])
        return result_generator_stator_temp
def run_generator_stator_temp( data, wt_id, vars):
    ###发电机定子温度异常
    model_path = 'Resource/generator_stator_temp/{}_model_max.pkl'.format(wt_id)
    data_residual_path = 'Resource/generator_stator_temp/{}_data_max.pkl'.format(wt_id)
    try:
        with open(model_path, 'rb') as fp:
            model = pickle.load(fp)
        with open(data_residual_path, 'rb') as fp:
            base_plot = pickle.load(fp)
    except:
        raise Exception('No Such File or Directory')

    module_generator_stator_temp = generator_stator_temp(active_frequency='24H', data_duration='240H', vars_input=vars, wt_id= wt_id)
    if module_generator_stator_temp.isavalible:
        module_generator_stator_temp.batch_list= module_generator_stator_temp.batch_generator(all_data=data)

    warning_history = []
    alarm_history = []
    result_generator_stator_temp = pd.DataFrame(module_generator_stator_temp.run_batch(all_data=data,
                                                                                    model=model,
                                                                                    base_plot=base_plot,
                                                                                    warning_history=warning_history,
                                                                                    alarm_history=alarm_history))

    #### Plot
    result_path = '../Result/generator_stator_temp'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath= os.path.join(result_path, wt_id + '_generator_stator_temp_distance.jpg')

    distance_plot(result_generator_stator_temp['start_time'], result_generator_stator_temp['distance'], wt_id, savepath)

    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                         '开始时间': min(data[ts]),
                         '结束时间': max(data[ts]),
                         '运行次数': len(result_generator_stator_temp),
                         '故障次数': len(result_generator_stator_temp[result_generator_stator_temp['alarm'] > 0]),
                         '最小健康度': float(Decimal(np.min(result_generator_stator_temp['distance'])).quantize( Decimal('0.00'))),
                         '最严重报警等级': np.max(result_generator_stator_temp['alarm'])
                          }, index=['0'])
    fault.to_csv(os.path.join('../Result/generator_stator_temp/', 'generator_stator_temp_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_generator_stator_temp

class anemometer_abnormal(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(anemometer_abnormal, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, turbineName, gp, env_temp, ws, ye, gs, ba, rs]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data,  model_ws_path, model_gp_path, benchmark_path):
        result_anemometer_abnormal =[]
        for i in range(0, len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            result = {}
            result['start_time'],result['end_time'],result['raw_data'],result['analysis_data'],result['status_code'],result['alarm'],result['distance']\
                = anemometer_wsunnormal_main(data=sample_data, rs_gp_model_p=model_gp_path ,rs_ws_model_p=model_ws_path,train_benchmark_path=benchmark_path)
            result_anemometer_abnormal.append(result)
            print(self.wt_id, result['start_time'], result['status_code'], result['distance'], result['alarm'])
        return result_anemometer_abnormal
def run_anemometer_abnormal( data, vars,wt_id):
    ###风速仪异常
    model_gp_path = 'Resource/anemometer_abnormal/{}_rs_gp_model.model'.format(wt_id)
    model_ws_path = 'Resource/anemometer_abnormal/{}_rs_ws_model.model'.format(wt_id)
    benchmark_path = 'Resource/anemometer_abnormal/{}train_benchmark.csv'.format(wt_id)


    module_anemometer_abnormal = anemometer_abnormal(active_frequency='72H', data_duration = '240H',  vars_input=vars, wt_id= wt_id)
    if module_anemometer_abnormal.isavalible:
        module_anemometer_abnormal.batch_list= module_anemometer_abnormal.batch_generator(all_data=data)

    result_anemometer_abnormal = pd.DataFrame(module_anemometer_abnormal.run_batch(all_data=data,
                                                                                    model_ws_path=model_ws_path,
                                                                                    model_gp_path=model_gp_path,
                                                                                    benchmark_path=benchmark_path
                                                                                    ))

    #### Plot
    result_path = '../Result/anemometer_abnormal'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath= os.path.join(result_path, wt_id + '_anemometer_abnormal_distance.jpg')

    distance_plot(result_anemometer_abnormal['start_time'], result_anemometer_abnormal['distance'], wt_id, savepath)

    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                          '开始时间': min(data[ts]),
                          '结束时间': max(data[ts]),
                          '运行次数': len(result_anemometer_abnormal),
                          '故障次数': len(result_anemometer_abnormal[result_anemometer_abnormal['alarm'] > 0]),
                          '最小健康度': np.nanmin(result_anemometer_abnormal['distance']),
                          '最严重报警等级': np.max(result_anemometer_abnormal['alarm'])
                          }, index=['0'])

    fault.to_csv(os.path.join('../Result/anemometer_abnormal/', 'anemometer_abnormal_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_anemometer_abnormal

class converter_temp_abnormal(module_basic_compare):
    def __init__(self, active_frequency, data_duration, vars_input, wt_ids):
        super(converter_temp_abnormal, self).__init__( active_frequency, data_duration, vars_input, wt_ids )
        self.vars_require=[ts, turbineName, gp, converter_temp]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data, wt_ids):
        result_converter_temp_abnormal =[]
        for i in range(0, len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            result = {}
            result["start_time"], result["end_time"], result["raw_data"], result["analysis_data"], result["status_code"], result["distance"], result["alarm"]\
                = converter_temp_abnormal_main(data=sample_data, _wt_ids= wt_ids, _output_wtids=wt_ids)
            result_converter_temp_abnormal.append(result)
            for turbine in wt_ids:
                print(result['start_time'], turbine, result['status_code'][turbine], result["distance"][turbine])
        return result_converter_temp_abnormal
def run_converter_temp_abnormal( data, wt_ids, vars):
    ###变流器温度异常
    module_converter_temp_abnormal = converter_temp_abnormal(active_frequency='24H', data_duration = '144H',  vars_input=vars, wt_ids= wt_ids)
    if module_converter_temp_abnormal.isavalible:
        module_converter_temp_abnormal.batch_list= module_converter_temp_abnormal.batch_generator(all_data=data)

    result_converter_temp_abnormal = module_converter_temp_abnormal.run_batch(all_data=data,wt_ids=wt_ids )

    result ={}
    if len(result_converter_temp_abnormal)>0:
        for wt_id in wt_ids:
            result[wt_id] = pd.DataFrame()
            for i in range(len(result_converter_temp_abnormal)):

                new = pd.DataFrame({'start_time': result_converter_temp_abnormal[i]['start_time'],
                                    'end_time': result_converter_temp_abnormal[i]['end_time'],
                                    # 'raw_data': result_converter_temp_abnormal[i]['raw_data'][wt_id],
                                    # 'analysis_data': result_converter_temp_abnormal[i]['analysis_data'][wt_id],
                                    'status_code': result_converter_temp_abnormal[i]['status_code'][wt_id],
                                    'distance': result_converter_temp_abnormal[i]['distance'][wt_id],
                                    'alarm': result_converter_temp_abnormal[i]['alarm'][wt_id],
                                   }, index = ['0'])
                result[wt_id]= pd.concat([result[wt_id], new], axis=0)



            #### Plot
            result_path = '../Result/converter_temp_abnormal'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            savepath= os.path.join(result_path, wt_id + '_converter_temp_abnormal_distance.jpg')

            distance_plot(result[wt_id]['start_time'], result[wt_id]['distance'], wt_id, savepath)

            ####保存异常记录
            fault = pd.DataFrame({'机组ID': wt_id,
                                  '开始时间': min(data[ts]),
                                  '结束时间': max(data[ts]),
                                  '运行次数': len(result[wt_id]),
                                  '故障次数': len(result[wt_id][result[wt_id]['alarm'] > 0]),
                                  '最小健康度': float(Decimal(np.min(result[wt_id]['distance'])).quantize( Decimal('0.00'))),
                                  '最严重报警等级': np.max(result[wt_id]['alarm'])
                                  }, index=['0'])

            fault.to_csv(os.path.join('../Result/converter_temp_abnormal/', 'converter_temp_abnormal_summary.csv'), encoding='gbk', mode='a', header=True)
    return result


class igbt_temp_abnormal(module_basic_compare):
    def __init__(self, active_frequency, data_duration, vars_input, wt_ids):
        super(igbt_temp_abnormal, self).__init__( active_frequency, data_duration, vars_input, wt_ids )
        self.vars_require=[ts, turbineName, gp, igbt_temp_1, igbt_temp_2]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data, wt_ids):
        result_igbt_temp_abnormal =[]
        for i in range(0, len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            result = {}
            result["start_time"], result["end_time"], result["raw_data"], result["analysis_data"], result["status_code"], result["distance"], result["distance1"], result["distance2"], result["alarm"]\
                = igbt_abnormal_main(data=sample_data, _wt_ids= wt_ids, _output_wtids=wt_ids)
            result_igbt_temp_abnormal.append(result)
            for turbine in wt_ids:
                print (result['start_time'], turbine, result['status_code'][turbine], result["distance"][turbine])

        return result_igbt_temp_abnormal
def run_igbt_temp_abnormal( data, wt_ids, vars):
    ###igbt温度异常
    module_igbt_temp_abnormal = igbt_temp_abnormal(active_frequency='24H', data_duration = '144H',  vars_input=vars, wt_ids= wt_ids)
    if module_igbt_temp_abnormal.isavalible:
        module_igbt_temp_abnormal.batch_list= module_igbt_temp_abnormal.batch_generator(all_data=data)

    result_igbt_temp_abnormal = module_igbt_temp_abnormal.run_batch(all_data=data,wt_ids=wt_ids )

    result ={}
    if len(result_igbt_temp_abnormal)>0:
        for wt_id in wt_ids:
            result[wt_id] = pd.DataFrame()
            for i in range(len(result_igbt_temp_abnormal)):

                new = pd.DataFrame({'start_time': result_igbt_temp_abnormal[i]['start_time'],
                                    'end_time': result_igbt_temp_abnormal[i]['end_time'],
                                    'status_code': result_igbt_temp_abnormal[i]['status_code'][wt_id],
                                    'distance': result_igbt_temp_abnormal[i]['distance'][wt_id],
                                    'alarm': result_igbt_temp_abnormal[i]['alarm'][wt_id],
                                   }, index = ['0'])
                result[wt_id]= pd.concat([result[wt_id], new], axis=0)



            #### Plot
            result_path = '../Result/igbt_temp_abnormal'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            savepath= os.path.join(result_path, wt_id + '_igbt_temp_abnormal_distance.jpg')

            distance_plot(result[wt_id]['start_time'], result[wt_id]['distance'], wt_id, savepath)

            ####保存异常记录
            fault = pd.DataFrame({'机组ID': wt_id,
                                  '开始时间': min(data[ts]),
                                  '结束时间': max(data[ts]),
                                  '运行次数': len(result[wt_id]),
                                  '故障次数': len(result[wt_id][result[wt_id]['alarm'] > 0]),
                                  '最小健康度': float(Decimal(np.min(result[wt_id]['distance'])).quantize( Decimal('0.00'))),
                                  '最严重报警等级': np.max(result[wt_id]['alarm'])
                                  }, index=['0'])

            fault.to_csv(os.path.join('../Result/igbt_temp_abnormal/', 'igbt_temp_abnormal_summary.csv'), encoding='gbk', mode='a', header=True)
    return result


class pitch_motor_temp_abnormal(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(pitch_motor_temp_abnormal, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, turbineName, ws, gp, pbt1, pbt2, pbt3, ba1_speed, ba2_speed, ba3_speed]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data,  model_compvec, model_idf, model_tf_idf,model_kde_plot ):
        result_pitch_motor_temp_abnormal =[]
        for i in range(0, len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            result = {}
            result["raw_data"], result["analysis_data"], result["status_code"], result["start_time"], result["end_time"], \
            result["alarm"], result["alarm1"], result["alarm2"], result["alarm3"], result["result1"], result["result2"], result["result3"]\
                = pitch_motor_temp_abnormal_main(data=sample_data, model_compvec=model_compvec, model_idf=model_idf, model_tf_idf=model_tf_idf, model_kde_plot=model_kde_plot)
            result_pitch_motor_temp_abnormal.append(result)
            print(self.wt_id, result['start_time'], result['status_code'], result['result1'],result['result2'],result['result3'], result['alarm'])
        return result_pitch_motor_temp_abnormal

def run_pitch_motor_temp_abnormal( data, wt_id, vars):
    ###变桨电机温度
    model_compvec_path = 'Resource/pitch_motor_temp_abnormal/{}_compvec.pkl'.format(wt_id)
    model_idf_path = 'Resource/pitch_motor_temp_abnormal/{}_idf.pkl'.format(wt_id)
    model_tf_idf_path = 'Resource/pitch_motor_temp_abnormal/{}_tf_idf.pkl'.format(wt_id)
    model_kde_plot_path= 'Resource/pitch_motor_temp_abnormal/{}_kde_plot.pkl'.format(wt_id)

    try:
        with open(model_compvec_path, 'rb') as fp:
            model_compvec = pickle.load(fp)
        with open(model_idf_path, 'rb') as fp:
            model_idf = pickle.load(fp)
        with open(model_tf_idf_path, 'rb') as fp:
            model_tf_idf = pickle.load(fp)
        with open(model_kde_plot_path, 'rb') as fp:
            model_kde_plot = pickle.load(fp)
    except:
        raise Exception('No Such File or Directory')

    module_pitch_motor_temp_abnormal = pitch_motor_temp_abnormal(active_frequency='72H', data_duration = '240H',  vars_input=vars, wt_id= wt_id)
    if module_pitch_motor_temp_abnormal.isavalible:
        module_pitch_motor_temp_abnormal.batch_list= module_pitch_motor_temp_abnormal.batch_generator(all_data=data)

    result_pitch_motor_temp_abnormal = pd.DataFrame(module_pitch_motor_temp_abnormal.run_batch(all_data=data,
                                                                                    model_compvec=model_compvec,
                                                                                    model_idf=model_idf,
                                                                                    model_tf_idf=model_tf_idf,
                                                                                    model_kde_plot=model_kde_plot
                                                                                    ))

    #### Plot
    result_path = '../Result/pitch_motor_temp_abnormal'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath= os.path.join(result_path, wt_id + '_pitch_motor_temp_abnormal_distance.jpg')

    result_pitch_motor_temp_abnormal['distance'] = list (map(  lambda x,y,z  : min (x,y,z),
                                                               result_pitch_motor_temp_abnormal['result1'],
                                                               result_pitch_motor_temp_abnormal['result1'],
                                                               result_pitch_motor_temp_abnormal['result1'] ) )
    distance_plot(result_pitch_motor_temp_abnormal['start_time'], result_pitch_motor_temp_abnormal['distance'], wt_id, savepath)

    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                          '开始时间': min(data[ts]),
                          '结束时间': max(data[ts]),
                          '运行次数': len(result_pitch_motor_temp_abnormal),
                          '故障次数': len(result_pitch_motor_temp_abnormal[result_pitch_motor_temp_abnormal['alarm'] > 0]),
                          '最小健康度': np.nanmin(result_pitch_motor_temp_abnormal['distance']),
                          '最严重报警等级': np.max(result_pitch_motor_temp_abnormal['alarm'])
                          }, index=['0'])

    fault.to_csv(os.path.join('../Result/pitch_motor_temp_abnormal/', 'pitch_motor_temp_abnormal_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_pitch_motor_temp_abnormal



####################变桨控制异常
class pitch_control_abnormal(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(pitch_control_abnormal, self).__init__(active_frequency, data_duration, vars_input, wt_id)
        self.vars_require = [ts, turbineName, ws, gp, ba1_speed, ba2_speed, ba3_speed]#输入变量
        self.isavalible = self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data):
        result_pitch_control_abnormal = []
        for i in range(0, len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            result = {}
            result["start_time"], result["end_time"], result["raw_data"], result["analysis_data"], result["status_code"],  \
            result["alarm"],result["distance"] = pitch_control_abnormal_main(data=sample_data)
            result_pitch_control_abnormal.append(result)
            print(self.wt_id, result['start_time'], result['status_code'], result['distance'], result['alarm'])


            if result['distance']<90:
                # ################Plot 最终结果
                plt.plot(result['analysis_data']['density_x1'], result['analysis_data']['density_y1'],  label='1-2')
                plt.plot(result['analysis_data']['density_x2'], result['analysis_data']['density_y2'],  label='1-3')
                plt.plot(result['analysis_data']['density_x3'], result['analysis_data']['density_y3'],  label='3-2')
                plt.ylabel('概率密度')
                plt.legend()
                plt.xlim(-3,3)
                plt.title(self.wt_id + '_' + str(result['start_time'])[0:10] + '_distance' + str(result['distance']))


                # plt.show()
                savePath = '../Result/pitch_control_abnormal/fault/'
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                plt.subplots_adjust(left=0.04, top=0.96, right=0.96, bottom=0.1, wspace=0.3, hspace=0.3)
                plt.savefig(savePath + self.wt_id + '_' + str(result['start_time'])[0:10] + 'pitch_control_abnormal.jpg',   format='jpg', dpi=plt.gcf().dpi, bbox_inches='tight')
                plt.clf()


        return result_pitch_control_abnormal


def run_pitch_control_abnormal(data, wt_id, vars):
    ###变桨控制异常

    module_pitch_control_abnormal = pitch_control_abnormal(active_frequency='24H', data_duration='24H',
                                                                 vars_input=vars, wt_id=wt_id)
    if module_pitch_control_abnormal.isavalible:
        module_pitch_control_abnormal.batch_list = module_pitch_control_abnormal.batch_generator(all_data=data)

    result_pitch_control_abnormal = pd.DataFrame(module_pitch_control_abnormal.run_batch(all_data=data   ))

    #### Plot
    result_path = '../Result/pitch_control_abnormal'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath = os.path.join(result_path, wt_id + '_pitch_control_abnormal_distance.jpg')

    distance_plot(result_pitch_control_abnormal['start_time'], result_pitch_control_abnormal['distance'], wt_id,    savepath)

    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                          '开始时间': min(data[ts]),
                          '结束时间': max(data[ts]),
                          '运行次数': len(result_pitch_control_abnormal),
                          '故障次数': len(result_pitch_control_abnormal[result_pitch_control_abnormal['alarm'] > 0]),
                          '最小健康度': np.nanmin(result_pitch_control_abnormal['distance']),
                          '最严重报警等级': np.max(result_pitch_control_abnormal['alarm'])
                          }, index=['0'])

    fault.to_csv(os.path.join('../Result/pitch_control_abnormal/', 'pitch_control_abnormal_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_pitch_control_abnormal


####################偏航驱动异常
class yaw_drive_abnormal(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(yaw_drive_abnormal, self).__init__(active_frequency, data_duration, vars_input, wt_id)
        self.vars_require = [ts, turbineName, ws, gp, nac_position, wd]#输入变量
        self.isavalible = self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data):
        result_yaw_drive_abnormal = []
        for i in range(0, len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            result = {}
            result["start_time"], result["end_time"], result["raw_data"], result["analysis_data"], result["status_code"],  \
            result["distance"], result["alarm"] = yaw_drive_abnormal_main(data=sample_data)
            result_yaw_drive_abnormal.append(result)
            print(self.wt_id, result['start_time'], result['status_code'], result['distance'], result['alarm'])

            try:
                if result['distance']<10:
                    # ################Plot 最终结果
                    plt.plot(result['analysis_data']['yaw_speed'] )
                    plt.ylabel('yaw_speed')
                    plt.legend()
                    # plt.xlim(-3,3)
                    plt.title(self.wt_id + '_' + str(result['start_time'])[0:10] + '_distance' + str(result['distance']))


                    # plt.show()
                    savePath = '../Result/yaw_drive_abnormal/fault/'
                    if not os.path.exists(savePath):
                        os.makedirs(savePath)
                    plt.subplots_adjust(left=0.04, top=0.96, right=0.96, bottom=0.1, wspace=0.3, hspace=0.3)
                    plt.savefig(savePath + self.wt_id + '_' + str(result['start_time'])[0:10] + 'yaw_drive_abnormal.jpg',   format='jpg', dpi=plt.gcf().dpi, bbox_inches='tight')
                    plt.clf()
            except:
                pass


        return result_yaw_drive_abnormal


def run_yaw_drive_abnormal(data, wt_id, vars):

    module_yaw_drive_abnormal = yaw_drive_abnormal(active_frequency='24H', data_duration='24H',
                                                                 vars_input=vars, wt_id=wt_id)
    if module_yaw_drive_abnormal.isavalible:
        module_yaw_drive_abnormal.batch_list = module_yaw_drive_abnormal.batch_generator(all_data=data)

    result_yaw_drive_abnormal = pd.DataFrame(module_yaw_drive_abnormal.run_batch(all_data=data   ))

    #### Plot
    result_path = '../Result/yaw_drive_abnormal'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath = os.path.join(result_path, wt_id + '_yaw_drive_abnormal_distance.jpg')

    distance_plot(result_yaw_drive_abnormal['start_time'], result_yaw_drive_abnormal['distance'], wt_id,    savepath)

    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                          '开始时间': min(data[ts]),
                          '结束时间': max(data[ts]),
                          '运行次数': len(result_yaw_drive_abnormal),
                          '故障次数': len(result_yaw_drive_abnormal[result_yaw_drive_abnormal['alarm'] > 0]),
                          '最小健康度': np.nanmin(result_yaw_drive_abnormal['distance']),
                          '最严重报警等级': np.max(result_yaw_drive_abnormal['alarm'])
                          }, index=['0'])

    fault.to_csv(os.path.join('../Result/yaw_drive_abnormal/', 'yaw_drive_abnormal_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_yaw_drive_abnormal








class control_cabinet_temp_abnormal(module_basic):
    def __init__(self, active_frequency, data_duration, vars_input, wt_id):
        super(control_cabinet_temp_abnormal, self).__init__( active_frequency, data_duration, vars_input, wt_id )
        self.vars_require=[ts, turbineName, gp, gs, cabin_temp, cct, converter_current_genside]#输入变量
        self.isavalible= self.check_avaliability()
        self.batch_list = None

    def run_batch(self, all_data,  residual_path, model_path, feature_path):
        result_control_cabinet_temp_abnormal =[]
        for i in range(0, len(self.batch_list)):
            sample_data = get_data(all_data, self.batch_list['st'][i], self.batch_list['et'][i], self.vars_require)
            result = {}
            result['start_time'],result['end_time'],result['raw_data'],result['analysis_data'],result['status_code'],result['distance'],result['alarm']\
                = ControlCabinetTempAbnormal_main(data=sample_data, residual_path=residual_path ,model_path=model_path,feature_path=feature_path, th = 50, rate= 0.1)

            if result['distance']<90:
                # ################Plot 最终结果
                plt.subplot(3,1,1)
                plt.plot(result['analysis_data']['benchmark_x'], result['analysis_data']['benchmark_y'],  label='base')
                plt.plot(result['analysis_data']['online_x'], result['analysis_data']['online_y'], color='black',label='本机组')
                plt.xlabel('残差')
                plt.ylabel('概率密度')
                plt.legend()
                plt.xlim(-3,3)
                plt.title(self.wt_id + '_' + str(result['start_time'])[0:10] + '_distance' + str(result['distance']))

                plt.subplot(3, 1, 2)
                plt.plot(result['raw_data']['date'], result['raw_data']['temp'], label = 'True')
                plt.plot(result['raw_data']['date'], result['raw_data']['pred'], label='Pred')
                # plt.gcf().autofmt_xdate()
                plt.legend()

                plt.subplot(3, 1, 3)
                plt.plot(result['raw_data']['date'], np.array(result['raw_data']['temp'])-np.array(result['raw_data']['pred']), label = 'res')
                plt.gcf().autofmt_xdate()
                plt.legend()

                # plt.show()
                savePath = '../Result/control_cabinet_temp_abnormal/fault/'
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                plt.subplots_adjust(left=0.04, top=0.96, right=0.96, bottom=0.1, wspace=0.3, hspace=0.3)
                plt.savefig(savePath + self.wt_id + '_' + str(result['start_time'])[0:10] + 'control_cabinet_temp_abnormal.jpg',   format='jpg', dpi=plt.gcf().dpi, bbox_inches='tight')
                plt.clf()


            result_control_cabinet_temp_abnormal.append(result)
            print(self.wt_id, result['start_time'], result['status_code'], result['distance'], result['alarm'])
        return result_control_cabinet_temp_abnormal


def run_control_cabinet_temp_abnormal( data, wt_id, vars):
    ###机舱控制柜异常
    residual_path = 'Resource/control_cabinet_temp/{}_Residual.pkl'.format(wt_id)
    model_path = 'Resource/control_cabinet_temp/{}_Model_1.pkl'.format(wt_id)
    feature_path = 'Resource/control_cabinet_temp/{}_Feature.pkl'.format(wt_id)


    module_control_cabinet_temp_abnormal = control_cabinet_temp_abnormal(active_frequency='48H', data_duration = '48H',  vars_input=vars, wt_id= wt_id)
    if module_control_cabinet_temp_abnormal.isavalible:
        module_control_cabinet_temp_abnormal.batch_list= module_control_cabinet_temp_abnormal.batch_generator(all_data=data)

    result_control_cabinet_temp_abnormal = pd.DataFrame(module_control_cabinet_temp_abnormal.run_batch(all_data=data,
                                                                                    residual_path=residual_path,
                                                                                    model_path=model_path,
                                                                                    feature_path=feature_path
                                                                                    ))

    #### Plot
    result_path = '../Result/control_cabinet_temp_abnormal'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    savepath= os.path.join(result_path, wt_id + '_control_cabinet_temp_abnormal_distance.jpg')

    distance_plot(result_control_cabinet_temp_abnormal['start_time'], result_control_cabinet_temp_abnormal['distance'], wt_id, savepath)

    ####保存异常记录
    fault = pd.DataFrame({'机组ID': wt_id,
                          '开始时间': min(data[ts]),
                          '结束时间': max(data[ts]),
                          '运行次数': len(result_control_cabinet_temp_abnormal),
                          '故障次数': len(result_control_cabinet_temp_abnormal[result_control_cabinet_temp_abnormal['alarm'] > 0]),
                          '最小健康度': np.nanmin(result_control_cabinet_temp_abnormal['distance']),
                          '最严重报警等级': np.max(result_control_cabinet_temp_abnormal['alarm'])
                          }, index=['0'])

    fault.to_csv(os.path.join('../Result/control_cabinet_temp_abnormal/', 'control_cabinet_temp_abnormal_summary.csv'), encoding='gbk', mode='a', header=True)
    return result_control_cabinet_temp_abnormal

if __name__ == '__main__':


    data_path = 'D:\\pycharm_code\\WT_TEST\\SCADA_Offline\\Code_05\\Data\\华润连州\\1min\\'

    fileNames = []
    wind_turbine_IDs = ['A02']
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            wind_turbine_IDs.append(name.split('.')[0])
            fileNames.append(os.path.join(root, name))

#     #################################################### 集群对标类算法，一次读入所有数据
#     all_data_10min = pd.DataFrame()
#     for i in range(0, len(wind_turbine_IDs)):
#         wt_id = wind_turbine_IDs[i]
#         path_10min = data_path + wt_id + '.csv'
#         oneTurbine_data_10min = pd.read_csv(path_10min, encoding='gbk')
#         oneTurbine_data_10min= oneTurbine_data_10min[int(0.8 * len(oneTurbine_data_10min)):]
#         # oneTurbine_data_10min[gs] = oneTurbine_data_10min[rs] * 106.87
#         oneTurbine_data_10min[ts] = pd.to_datetime(oneTurbine_data_10min[ts])
#         oneTurbine_data_10min = oneTurbine_data_10min.drop_duplicates([ts])
#         vars_10min = oneTurbine_data_10min.columns.to_list()
#         all_data_10min = pd.concat([all_data_10min, oneTurbine_data_10min], axis=0)
#         print (wt_id, '_Loaded')

#     # result_converter_temp_abnormal = run_converter_temp_abnormal(data=all_data_10min[:], vars=vars_10min,  wt_ids=wind_turbine_IDs[:])
#     result_igbt_temp_abnormal = run_igbt_temp_abnormal(data=all_data_10min[:], vars=vars_10min,wt_ids=wind_turbine_IDs[:])
# #
# ############################################################################################

    for i in range(len(wind_turbine_IDs)):
        wt_id = wind_turbine_IDs[i]
        path_10min = data_path + wt_id + '.csv'
        oneTurbine_data_10min = pd.read_csv(path_10min, encoding='gbk')
        oneTurbine_data_10min[turbineName] = wt_id
        # oneTurbine_data_10min[rs] = oneTurbine_data_10min[gs] * 106.87
        oneTurbine_data_10min[ts] = pd.to_datetime(oneTurbine_data_10min[ts])
        oneTurbine_data_10min= oneTurbine_data_10min.drop_duplicates([ts])

        ########变桨速率实际没有
        oneTurbine_data_10min[ba1_speed] = (oneTurbine_data_10min['pitch_1']- oneTurbine_data_10min['pitch_1'].shift(1))/60
        oneTurbine_data_10min[ba2_speed] = (oneTurbine_data_10min['pitch_2']- oneTurbine_data_10min['pitch_2'].shift(1))/60
        oneTurbine_data_10min[ba3_speed] = (oneTurbine_data_10min['pitch_3']- oneTurbine_data_10min['pitch_3'].shift(1))/60

        vars_val_data = oneTurbine_data_10min.columns.to_list()
        val_data = oneTurbine_data_10min[int(0.4 * len(oneTurbine_data_10min)):]



        ### result_yaw_misalignment = run_yaw_misalignment(data= val_data, vars=vars_val_data, wt_id= wt_id)#对风不正
        ### result_power_curve_devi= run_power_curve_devi(data=val_data, vars=vars_val_data, wt_id=wt_id)#功率偏移

        ### result_blade_icing= run_blade_icing(data= val_data[:], vars=vars_val_data, wt_id= wt_id)# 叶片结冰
        ### result_pitch_motor_temp_abnormal = run_pitch_motor_temp_abnormal(data=val_data[:], vars=vars_val_data, wt_id=wt_id)# 变桨温度异常
        ### result_pitch_control_abnormal = run_pitch_control_abnormal(data=val_data[:], vars=vars_val_data,  wt_id=wt_id)# 变桨控制异常
        ### result_yaw_drive_abnormal = run_yaw_drive_abnormal(data=val_data[:], vars=vars_val_data, wt_id=wt_id) # 偏航驱动异常

        ### result_gen_bearing_temp= run_generator_bearing_temp( data= val_data, vars=vars_val_data, wt_id= wt_id) # 发电机轴承温异常
        ### result_gen_bearing_stuck = run_generator_bearing_stuck(data=val_data, vars=vars_val_data, wt_id=wt_id) # 发电机轴承卡涩

        ### result_gearbox_bearing_temp = run_gearbox_bearing_temp(data=val_data, vars=vars_val_data,wt_id=wt_id)  # 齿轮箱轴承温度异常
        ### result_gearbox_oil_temp = run_gearbox_oil_temp(data=val_data, vars=vars_val_data, wt_id=wt_id)  # 齿轮箱油温异常
        ### result_generator_stator_temp = run_generator_stator_temp(data=val_data,vars=vars_val_data, wt_id=wt_id)# 发电机定子温度异常



        result_anemometer_abnormal = run_anemometer_abnormal(data=val_data, vars=vars_val_data, wt_id=wt_id)#
        # result_control_cabinet_temp_abnormal = run_control_cabinet_temp_abnormal(data=val_data, wt_id=wt_id, vars=vars_val_data)

    print('end')