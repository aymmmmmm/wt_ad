# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:44:34 2018
2小时运行一次，一次采用2小时数据，数据采样频率5s
@author:
"""

import numpy as np
import pandas as pd
import joblib
import pickle
import os

from xgboost.sklearn import XGBClassifier
from decimal import *
import warnings
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#
# model_wp = r'./Resource/ws_power_model_W056.pkl'
# ws_power_model = joblib.load(model_wp)




wtid = 'turbineName'
ts = 'time' # 时间戳
ws = 'wind_speed' # 风速变量名
gp = 'power' # 有功功率变量名
wtstatus = 'main_status'  # 机组状态变量名
wtstatus_n = 0  # 正常发电状态代码
cabin_temp= 'cabin_temp' # 机舱内温度变量名
outside_temp = 'environment_temp' # 舱外温度变量名
gs = 'generator_rotating_speed' # 发动机转速变量名
cv = 'nacelle_vib_X' # 塔筒振动x轴方向变量名
b1pitch ='pitch_1'



ws_threshold = 5.0 # 切入风速阈值
temp_threshold = 3.0 # 结冰温度阈值
level_0 = 80 # 预警结果阈值1
level_1 = 60 # 预警结果阈值2
level_2 = 40 # 预警结果阈值3
prob_threshold = 0.2 # 预测概率到输出结冰标签阈值
ratedPower =3200.0
# 增加了pitch参数， 20210917——jimmy
columns = [ts, wtstatus, cv, ws, gp, cabin_temp, outside_temp, gs, b1pitch]


def data_preprocessing(data):
    """
    将输入数据与点表变量名进行匹配，转换时间戳、排序等一些预处理
    :data: 输入历史2小时一台风机的数据
    :return: 返回处理后的该台风机正常发电时的数据
    """
    data_tmp = data.copy()
    # data_tmp = data_tmp[[ts, ws, gp, outside_temp, cabin_temp, gs, cv, wtid, b1pitch]]
    data_tmp = data_tmp[[ts, wtid, ws, gp, gs, b1pitch]]

    # 风向没有清洗
    data_tmp = data_tmp[(data_tmp[ws] > 3) & (data_tmp[gp] > 10) & (data_tmp[gs] > 5)
                        & ~ ((data_tmp[gp] < ratedPower * 0.95) & (data_tmp[b1pitch] > 3))
                        # & ~ ((data_tmp[gp] < 1000) & (data_tmp[ws] > 10))
                        ]

    # data_tmp = data_tmp[(data_tmp[ws] > 3) & (data_tmp[gp] > 10) & (data_tmp[gs] > 5)]
    # data_tmp = data_tmp[[ts, ws, gp, outside_temp, cabin_temp, gs, cv]]
    data_tmp[ts] = pd.to_datetime(data_tmp[ts])
    data_tmp[ts] = data_tmp[ts].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    data_tmp = data_tmp.set_index(ts, drop=True).sort_index()

    # plt.scatter(data_tmp[ws], data_tmp[gp], s=1)
    # plt.show()
    return data_tmp


def data_preprocessing_realData(data):
    # 实测数据补全
    data_tmp = data.copy()
    data_tmp = data_tmp[[ts, ws, gp, outside_temp, cabin_temp, gs, cv, b1pitch, wtid]]
    data_tmp[ts] = pd.to_datetime(data_tmp[ts])
    data_tmp[ts] = data_tmp[ts].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    data_tmp = data_tmp.set_index(ts, drop=True).sort_index()

    data_tmp.dropna(subset=[ws, gp], inplace=True)

    for i in data_tmp.columns:
        if i == cabin_temp or i == outside_temp or i == gs or i == cv or i == b1pitch:
            data_tmp[i] = data_tmp[i].interpolate(method='linear')
            # 如果还有空的，就用后一个填充-20190916
            data_tmp[i] = data_tmp[i].fillna(method='backfill')
            data_tmp[i] = data_tmp[i].fillna(method='ffill')

    return data_tmp




def import_data_check(data, columns):
    status_code = '000'

    # None
    if data is None:  # 判断数据中某列是否全部为空值
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
    # if (data[columns[3:]].dtypes != 'float').any():
    #     # raise Exception('the Dtype of Input Data_Raw is not Float')
    #     status_code = '302'
    # duplicate values
    for i in columns[3:]:
        if sum(data[i].duplicated()) == len(data):
            # raise Exception('Input Data_Raw Contains Too Many Duplicates')
            status_code = '102'
    # negative value
    if any(data[gs] < 0):
        # raise Exception('Rotating Speed Contains Negative Values')

        status_code = '302'
    # 状态码如果有缺失按照最靠近的一个做填充-20210916
    data['main_status'] = data['main_status'].fillna(method='ffill')

    if sum(data['main_status'] == wtstatus_n) == 0:  # 检查风机正常运行状态数量是否为0
        status_code = '300'


    elif sum(data['main_status'] == wtstatus_n) < 0.5 * len(data):  # 检查风机正常运行状态数量是否超过一半
        status_code = '100'
    # shortage of data
    if len(data) < 180:  ##半小时，5s采样，共360个点
        status_code = '101'
    return status_code


def feature_extraction(data, model_wp):
    """
    对输入数据筛选要保留的特征, 并根据机理和历史拟合模型计算新的风机运行工况特征
    新增特征: 风速平方 ws_square
             机舱内外温差 temp_d
             扭矩 torque
             风能利用系数 Cp
             转矩系数 Ct
             叶尖速比 TSR
             功率偏移 power_d
    :data: 输入预处理好的历史2小时某一台风机的数据
    :param wtid: 待分析的一台风机ID
    :return: 返回该台风机正常发电时新的工况特征矩阵
    """

    data['ws_square'] = data[ws] ** 2
    data['temp_d'] = data[cabin_temp] - data[outside_temp]
    data['torque'] = data[gp] / data[gs]
    data['Cp'] = data[gp] / (data[ws] ** 3)
    data['Ct'] = data['torque'] / (data[ws] ** 2)
    data['TSR'] = data[gs] / data[ws]
    data = data.fillna(0)
    data['power_pred'] = model_wp.predict(data[ws].values.reshape(-1, 1))
    power_d = model_wp.predict(data[ws].values.reshape(-1, 1)) - data[gp]
    data['power_d'] = power_d
    return data

def set_label(data,ws_power_model, ws_threshold, temp_threshold):
    '''
    根据机理规则设定结冰情况标签
    步骤S1: 判断风机外部风速是否处于风机运行范围内
    步骤S2: 判断风机外部环境温度是否低于设定值
    步骤S3: 判断当前风速下理论输出功率与风机实时输出功率差值与历史正常情况是否出现偏移
    步骤S4: 判断风机塔筒振动值与历史正常情况是否出现偏移，若没有偏移，表示风机可能结冰；若出现了偏移，表示风机有其他故障
    步骤S5: 判断风机机舱内外温差与历史正常情况是否出现偏移，若没有偏移，表示风机已经结冰，风机停机或者启动叶片加热装置，若出现了偏移，表示风机有其他故障
    :data: 输入经过特征提取的历史2小时某一台风机的数据
    :param wtid: 待诊断的一台风机ID
    :return: 返回该台风机打上结冰标签的数据
    '''

    p_d_upper = 1000
    v_upper = 0.3
    v_lower = -0.3
    t_d_upper = 3
        
    ws_condition = data[ws] >= ws_threshold
    temp_condition = data[outside_temp] <= temp_threshold
    # power_deviation = data.power_d > p_d_upper

    ##功率偏差大于10%--jimmy 20211103
    power_deviation = data['power_d'] > data['power_pred']*0.1
    vibration_anomaly = ~ ((data[cv] > v_upper) | (data[cv] < v_lower))
    temp_d_condition = data.temp_d > t_d_upper

    #增加桨距角约束，删除限功率情况-20210917
    power_constrain = data[b1pitch]<1

    data['failure'] = np.select([ws_condition & temp_condition & power_deviation & vibration_anomaly & temp_d_condition & power_constrain ], [1], 0)

#################输出查看
    # print (data[wtid][0],  'total ', len(data), ' failure ', len(data[data['failure'] == 1]), ' well ', len(data[data['failure'] == 0]))

    # model_wp = './PublicUtils/Resource/ws_power_model_'+ data[wtid][0] + '.pkl'
    # with open(model_wp, 'rb') as f:
    #     ws_power_model = pickle.load(f)  # 读取该风机对应的理论风速-功率模型, 该模型由过去长时期正常运行数据拟合 #####################
    #
    # power_d = ws_power_model.predict(pd.DataFrame(np.arange(1, 20, 0.5)))
    # if len(data[data['failure'] == 1]) > 0.2*len(data):
    #     plt.subplot(2, 3, 1)
    #     plt.scatter(data[data['failure'] == 0][ws], data[data['failure'] == 0][gp], color='blue', s=1)
    #     plt.scatter(data[data['failure'] == 1][ws], data[data['failure'] == 1][gp], color='r', s=1)
    #     plt.plot(np.arange(1, 20, 0.5), power_d, color="red", linewidth=1)
    #     plt.title(data[wtid][0] +'_' +data.index[0])
    #     plt.subplot(2, 3, 2)
    #     plt.scatter(data[data['failure'] == 0][ws], data[data['failure'] == 0][outside_temp], color='blue', s=1)
    #     plt.scatter(data[data['failure'] == 1][ws], data[data['failure'] == 1][outside_temp], color='r', s=1)
    #     plt.title('outside_temp')
    #     plt.subplot(2, 3, 3)
    #     plt.scatter(data[data['failure'] == 0][ws], data[data['failure'] == 0]['temp_d'], color='blue', s=1)
    #     plt.scatter(data[data['failure'] == 1][ws], data[data['failure'] == 1]['temp_d'], color='r', s=1)
    #     plt.title('temp_d')
    #     plt.subplot(2, 3, 4)
    #     plt.scatter(data[data['failure'] == 0][ws], data[data['failure'] == 0][cv], color='blue', s=1)
    #     plt.scatter(data[data['failure'] == 1][ws], data[data['failure'] == 1][cv], color='r', s=1)
    #     plt.title(cv)
    #     plt.subplot(2, 3, 5)
    #     plt.scatter(data[data['failure'] == 0][ws], data[data['failure'] == 0][b1pitch], color='blue', s=1)
    #     plt.scatter(data[data['failure'] == 1][ws], data[data['failure'] == 1][b1pitch], color='r', s=1)
    #     plt.ylim(-2,5)
    #     plt.title(b1pitch)
    #
    #     data.to_csv('demo_01.csv')
    #
    #     # outpath = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\result\icing\\' + data[wtid][0]
    #     outpath = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\result\\icing'
    #     if not os.path.exists(outpath):
    #         os.makedirs(outpath)
    #
    #     plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
    #     # plt.show()
    #     plt.savefig(outpath + os.sep + data[wtid][0] + 'WT' + str(data.index[0])[:-6] + '.jpg')
    #     plt.close()



    return data

def pulearning_icing(data):
    """
    将人工赋值的结冰标签作为正样本，
    若正样本数小于剩余样本数，使用基于XGBoost模型的bagging方法对其它未被规则识别的样本进行二分类，以迭代均值作为预测概率输出
    若正样本数大于剩余样本数，使用正样本所占比例作为预测概率输出
    :data: 输入打好标签的历史2小时某一台风机的数据
    :return: 输入数据的结冰概率预测值
    """
    data_P = data[data['failure'] == 1].drop('failure', axis = 1)  # drop!
    data_U = data[data['failure'] == 0].drop('failure', axis = 1)
    NP = data_P.shape[0]
    NU = data_U.shape[0]

    #增加一个条件，有标签的数量必须超过总数的5%--20210917-jimmy
    if  len(data_P) < len(data_U)  &  (len(data_P) > len(data)*0.05)  :

        T = 50 # No. of iterations
        K = NP
        train_label = np.zeros(shape = (NP + K,))
        train_label[: NP] = 1.0
        n_oob = np.zeros(shape = (NU,))
        f_oob = np.zeros(shape = (NU, 2))
        for i in range(T):
            # Bootstrap resample
            bootstrap_sample = np.random.choice(np.arange(NU), replace = True, size = K)
            # Positive set + bootstrapped unlabeled set
            data_bootstrap = pd.concat((data_P, data_U.iloc[bootstrap_sample, :]), axis = 0)
            # Train model
            model = XGBClassifier(objective = 'binary:logistic')
            model.fit(data_bootstrap, train_label)
            # Index for the out of the bag (oob) samples , 没有结冰的数据里头除去刚才被选中加入训练的部分
            idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
            # Transductive learning of oob samples
            f_oob[idx_oob] += model.predict_proba(data_U.iloc[idx_oob])
            n_oob[idx_oob] += 1
        predict_proba = f_oob[:, 1] / n_oob

        data_U['probability'] = predict_proba
        data_P['probability'] = 1

    else:

        predict_proba = np.zeros(len(data_U))
        data_U['probability'] = predict_proba
        data_P['probability'] = 1

    result = pd.concat([data_U, data_P]).sort_index()
    result['failure'] = data['failure']
    pred_prob = 100 * result.probability.mean()
    result.loc[result.probability >= prob_threshold, 'failure'] = 1

    distance = 100 - pred_prob
    return float(distance), result

def prob2alarm(distance):
    """
    根据预测结冰概率的大小对应相应的警报等级
    """
    if level_0 < distance <= 100:
        alarm = 0
    elif level_1 < distance <= level_0:
        alarm = 1
    elif level_2 < distance <= level_1:
        alarm = 2
    elif distance <= level_2:
        alarm = 3
    else:
        alarm = np.nan
    return int(alarm)

def blade_icing_main(data,model_wp):
    """
    叶片结冰算法主函数入口，包括数据预检查/预处理，构造结冰标签，分类并输出结果
    :data: 输入2小时5秒级风机运行数据
    :return: result, 以字典形式储存输出结果
             keys: status_code 该段风机运行状态码，'00'为正常
                   start_time 该段数据起始时间
                   end_time: 该段数据截止时间
                   raw_data: 输入数据风机运行原始特征
                   analysis_data: 判据特征
                   pred_prob: 结冰概率
                   alarm: 根据结冰概率所对应的警报级别
    """

    global wtid
    global ts
    global ws
    global gp
    global wtstatus
    global wtstatus_n
    global cabin_temp
    global outside_temp
    global gs
    global cv
    global b1pitch
    global ws_threshold
    global temp_threshold
    global level_0
    global level_1
    global level_2
    global prob_threshold
    global columns
    global ratedPower

    columns = [ts, wtstatus, cv, ws, gp, cabin_temp, outside_temp, gs, b1pitch, wtid]

    # Initialize
    data = data[columns]

    status_code = import_data_check(data, columns)


    raw_data = dict()
    analysis_data = dict()
    result = dict()
    if (status_code == '000') or (status_code == '100') or (status_code == '101'): # 状态码为0、1开头不影响算法运行
        data_1 = data_preprocessing_realData(data)
        data_2 = feature_extraction(data_1, model_wp)
        data_3 = set_label(data_2, model_wp, ws_threshold, temp_threshold)
        distance, result_df = pulearning_icing(data_3)[0], pulearning_icing(data_3)[1]
        
        raw_data['datetime'] = list(map(str, result_df.index.tolist()))
        raw_data['env_temp'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), result_df[outside_temp].values.tolist()))
        raw_data['wind_speed'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), result_df[ws].values.tolist()))
        raw_data['active_power'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), result_df[gp].values.tolist()))
        raw_data['label'] = list(map(int, result_df.failure.values.tolist()))
        
        analysis_data['online_x'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), result_df[ws].values.tolist()))
        analysis_data['online_y1'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), result_df[gp].values.tolist()))
        analysis_data['online_y2'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), result_df.probability.values.tolist()))
        analysis_data['online_y3'] = list(map(int, result_df.failure.tolist()))
        
        alarm = prob2alarm(distance)
        
        start_time = str(min(result_df.index))
        end_time = str(max(result_df.index))
    elif status_code == '300':
        distance = None
        alarm = None
        raw_data['datetime'] = list(map(str, data[ts].values.tolist()))
        raw_data['env_temp'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[outside_temp].values.tolist()))
        raw_data['wind_speed'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[ws].values.tolist()))
        raw_data['active_power'] = list(map(lambda x: float(Decimal(x).quantize(Decimal('0.0000'))), data[gp].values.tolist()))
        raw_data['label'] = [np.nan] * len(data)
        analysis_data = None
        start_time = str(data[ts].iloc[0])
        end_time = str(data[ts].iloc[-1])
    else:
        distance = None
        alarm = None
        raw_data = None
        analysis_data = None
        start_time = None
        end_time = None
        
    result['distance'] = distance
    result['raw_data'] = raw_data
    result['analysis_data'] = analysis_data
    result['status_code'] = status_code
    result['start_time'] = start_time
    result['end_time'] = end_time
    result['alarm'] = alarm
    return result['distance'],result['raw_data'],result['analysis_data'] ,result['status_code'],result['start_time'],result['end_time'], result['alarm']



# if __name__ == '__main__':
#     model_wp = r'./Resource/ws_power_model_85.pkl'
#
#     with open(model_wp, 'rb') as f:
#         ws_power_model = pickle.load(f)
#
#     # data_tmp = pd.read_json(r"D:\Users\Administrator\Downloads\BladeIcingDetection-85_BB\BladeIcingDetection-85-data.json")
#     data_tmp = pd.read_json(        r"D:\Users\Administrator\Downloads\BladeIcingDetection-10_BB\BladeIcingDetection-10-data.json")
#     result = {}
#     result['distance'], result['raw_data'], result['analysis_data'], result['status_code'], result['start_time'], result['end_time'], result['alarm'] = blade_icing_main(data_tmp, ws_power_model)
#     pass

def evidence_plot(data, wtid, level, alarm):
    outpath = outdir + str(wtid)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    plt.figure(figsize=(12, 16))
    plt.grid(True)
    plt.subplot(211)
    plt.scatter(np.arange(len(data)), data[outside_temp], c='g')
    plt.grid(True)
    plt.ylabel('环境温度')
    plt.title('goldwind' + str(wtid) + '号风机' + str(data[ts].iloc[0]) + str(level) + '级报警结果环境温度判据')

    plt.subplot(212)
    plt.scatter(data[ws], data[gp], c='b')
    model_wp = r'./Resource/ws_power_model_W056.pkl'
    ws_power_model = joblib.load(model_wp)  # 读取该风机对应的理论风速-功率模型, 该模型由过去长时期正常运行数据拟合 #####################
    power_d = ws_power_model.predict(pd.DataFrame(np.arange(1, 20, 0.5)))
    plt.plot(np.arange(1, 20, 0.5), power_d, color="red", linewidth=5, marker="o")

    plt.grid(True)
    plt.ylabel('风功率曲线')
    plt.title('goldwind' + str(wtid) + '号风机' + str(data[ts].iloc[0]) + str(level) + '级报警结果风功率曲线判据')
    plt.savefig(outpath + os.sep + 'WT' + str(wtid) + str(level) + '级报警结果判据' + str(alarm) + '.jpg')
    plt.close()

#############################################################
if __name__ == '__main__':
    import glob

    path = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\华润风机数据\数据\\Prep_02_generatorTemp\\1min'
    allFiles = glob.glob(path + "/*.csv")

    outdir = r'C:\Project\03_AeroImprovement\05_华润风机SCADA数据分析_202111\result\icing'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    file = []

    for file_ in allFiles:
        input_data = pd.read_csv(file_, engine='python', parse_dates=['time'], encoding='gb2312')
        input_data[wtstatus] =0

        wt_id = input_data[wtid][0]

        model_wp = './PublicUtils/Resource/ws_power_model_'+ wt_id + '.pkl'
        with open(model_wp, 'rb') as f:
            ws_power_model = pickle.load(f)


        # power_d = ws_power_model.predict(pd.DataFrame(np.arange(1, 20, 0.5)))
        # plt.plot(np.arange(1, 20, 0.5), power_d, color='red', label= wt_id)
        # plt.legend()
        # # plt.show()
        # outpath = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\result\\powerCurve'
        # plt.savefig(outpath + os.sep + wt_id +  '.jpg')
        # plt.close()


        start_time = '2021-1-1 00:00:00'
        for n in range(90):
            time_range = pd.date_range(start_time, periods=2, freq='6H')
            st = time_range[0]
            et = time_range[1]

            data_tmp = input_data[(input_data[ts] >= st) & (input_data[ts] < et)]
            data_tmp = data_tmp.dropna()
            data_tmp.reset_index(inplace=True)
            status_code = import_data_check(data_tmp, columns)
            print(status_code)

            result = {}
            result['distance'], result['raw_data'], result['analysis_data'], result['status_code'], result[
                'start_time'], result['end_time'], result['alarm'] = blade_icing_main(data_tmp, ws_power_model)
            print(wt_id , result['start_time'])


            start_time = et



        # test_df = pd.DataFrame(columns=['wtid', 'alarm_0', 'alarm_1', 'alarm_2', 'alarm_3', 'alarm_none', 'error'])


        # print('read data finished and the length is :', len(data))
        #
        # wtid = str(j)
        #
        # alarm_0 = 0
        # alarm_1 = 0
        # alarm_2 = 0
        # alarm_3 = 0
        # alarm_none = 0
        # error = 0
        # for i in range(1440, len(data), 1440):  # 5秒钟一个，2小时
        #     data_tmp = data.iloc[i - 1440: i, :]
        #     try:
        #         result = blade_icing_main(data_tmp)
        #         print('WT' + str(j), 'Date: ' + str(data_tmp[ts].iloc[0]) + ' codeState is '+ result['status_code'], ', alarm is: ' + str(result['alarm']), ',Distance is ' + str(result['distance']) )
        #         if result['alarm'] == 0:
        #             alarm_0 += 1
        #         elif result['alarm'] == 1:
        #             alarm_1 += 1
        #             evidence_plot(data_tmp, j, '一', alarm_1)
        #         elif result['alarm'] == 2:
        #             alarm_2 += 1
        #             evidence_plot(data_tmp, j, '二', alarm_2)
        #         elif result['alarm'] == 3:
        #             alarm_3 += 1
        #             evidence_plot(data_tmp, j, '三', alarm_3)
        #         elif result['alarm'] is None:
        #             alarm_none += 1
        #     except AttributeError:
        #         print('WT' + str(j), 'Date: ' + str(data_tmp[ts].iloc[0]) + ', Error!')
        #         error += 1
        # df_tmp = pd.DataFrame(np.array([j, alarm_0, alarm_1, alarm_2, alarm_3, alarm_none, error])).T
        # df_tmp.columns = ['wtid', 'alarm_0', 'alarm_1', 'alarm_2', 'alarm_3', 'alarm_none', 'error']
        # test_df = pd.concat([test_df, df_tmp])
        # test_df.to_csv(r'C:\Project\07_DaTang_BladeProtector\Algorithm\BladeIcingDetection\Pre_process_ForIcing_0917\output0917\report_w085.csv')