"""
变桨电机温度异常训练程序
"""
import os
import datetime
import numpy as np
import pandas as pd
from PublicUtils.util_datadiscretization import data_discretization
from PublicUtils.util_tfidf import tfidf_by_window, make_dist_df, idf_by_window
import pickle
import math
import statsmodels.api as sm


window = 1440
pbt1 = 'pitch_motor_temp_1'
pbt2 = 'pitch_motor_temp_2'
pbt3 = 'pitch_motor_temp_3'

ws = 'wind_speed'
temp_threshold = 10  # 取大于多少度的数据
time = 'time'  # 时间变量名
diff_upper_limit = 10
diff_lower_limit = -10
bin_size = 0.5


ba1_speed = "pitch_speed_1"
ba2_speed = "pitch_speed_2"
ba3_speed = "pitch_speed_3"

turbineName = 'turbineName'



def data_preprocess(data):  # TODO 新增预处理函数
    # for v in [pbt1,pbt2,pbt3,wind_speed,main_status]:
    data[[pbt1, pbt2, pbt3, ws]].interpolate(method='linear', limit=1500, axis=0, inplace=True)
    data.fillna(method="ffill", inplace=True, limit=1000)
    data.fillna(method="bfill", inplace=True, limit=1000)
    return data.dropna()


def resample_bytime_mean(data, minnum, dt):
    """
    resample by time window
    return mean value of resampled data
    :param data: 输入数据，dataframe
    :param minnum: 按几分钟resample，float
    :param dt: 日期时间变量名，str
    :return: 输出结果，dataframe
    """
    data.index = data[dt].values
    scale = str(minnum) + 'T'
    r = data.resample(scale).mean()
    r[dt] = r.index
    return r


def sort_and_downsample(kp):
    pd_kp = pd.DataFrame()
    pd_kp['support'] = kp['support']
    pd_kp['density'] = kp['density']
    sample_len = int(min(len(kp['support']) * 0.1, 2000))
    pd_kp = pd_kp.sample(n=sample_len)
    pd_kp = pd_kp.sort_values(by='support')
    rtn = {}
    rtn['support'] = pd_kp['support'].values
    rtn['density'] = pd_kp['density'].values
    return rtn


def pb_fit(data):
    words = np.array(list( map(str, np.arange(diff_lower_limit, diff_upper_limit + bin_size, bin_size))))

    data = data.loc[(data['pb_diff1'] < diff_upper_limit) & (data['pb_diff1'] > diff_lower_limit) & (
            data['pb_diff2'] < diff_upper_limit) & (data['pb_diff2'] > diff_lower_limit) & (
                            data['pb_diff3'] < diff_upper_limit) & (data['pb_diff3'] > diff_lower_limit), :]
    print(len(data))

    train_data = pd.DataFrame({'pb_diff': data['pb_diff1'].values.tolist() + data['pb_diff2'].values.tolist() + data[ 'pb_diff3'].values.tolist(  )})

####这里为什么要加随机数？？
    train_data = pd.concat([train_data, -train_data,
                            pd.DataFrame( {"pb_diff": diff_upper_limit / 2 * np.random.random(int(0.1 * len(train_data)))}),
                            pd.DataFrame( {"pb_diff": diff_lower_limit / 2 * np.random.random(int(0.1 * len(train_data)))})])

    train_data = train_data.loc[abs(train_data['pb_diff']) < diff_upper_limit, :]

    ##训练数据一分钟采样
    train_data[time] = pd.date_range( start='1/1/2020',  periods=len(train_data),  freq='1min')  ##本来就是1min采样
    train_data = resample_bytime_mean(train_data, 1, time)

    print(len(train_data))
    train_data[time] = pd.to_datetime(train_data[time])

    train_data['pb_diff_bin'] = data_discretization(
        train_data['pb_diff'],
        mode='type2',
        minnum=-diff_upper_limit - bin_size,
        maxnum=diff_upper_limit + bin_size,
        binsize=bin_size)

    idf = idf_by_window(train_data['pb_diff_bin'].values, words, window)
    # print(idf)
    tfidf = tfidf_by_window(train_data, 'pb_diff_bin', time, idf, window)
    compvec = np.array([np.mean(x)
                        for x in list(zip(*tfidf['TF-IDF'].values))])
    tfidf_distance = make_dist_df(
        tfidf, compvec, 'DateTime', 'TF-IDF', mode='B')

    mean_train = np.mean(tfidf_distance['distance'].values)
    std_train = np.std(tfidf_distance['distance'].values)

    kde = sm.nonparametric.KDEUnivariate(train_data['pb_diff'].values)
    kde.fit(bw=1)
    kde_plot = dict()
    kde_plot['support'] = kde.support
    kde_plot['density'] = kde.density

    kde_plot = sort_and_downsample(kde_plot)

    return idf, compvec, kde_plot, tfidf_distance


def pb_fit_main(data, i):
    # data = data.loc[(data[main_status]>=34) & (data[main_status]<=38)&(data[ws]>6)]
    # data = data.loc[data[main_status].isin(wtstatus_n) & (data[ws]>(rated_speed-1))] # 原来的风机运行状态是[34,38]
    data = data.loc[(data[ba1_speed].abs() > 0) | (data[ba2_speed].abs() > 0) | (data[ba3_speed].abs() > 0)]

    print(data.shape)

    idf, compvec, kde_plot, tfidf_distance = pb_fit(data)

    outpath = '..\\Resource\\pitch_motor_temp_abnormal'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    idf_file = outpath + os.sep + i + '_idf.pkl'
    with open(idf_file, "wb") as f:
        pickle.dump(idf, f)

    tfidf_file = outpath + os.sep + i + '_tf_idf.pkl'
    with open(tfidf_file, "wb") as f:
        pickle.dump(tfidf_distance, f)

    cv_file = outpath + os.sep + i + '_compvec.pkl'
    with open(cv_file, "wb") as f:
        pickle.dump(compvec, f)

    kde_file = outpath + os.sep + i + '_kde_plot.pkl'
    with open(kde_file, 'wb') as f:
        pickle.dump(kde_plot, f)

    return idf, compvec, kde_plot


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

        all_data_10min['pitch_speed_1']=1
        all_data_10min['pitch_speed_2'] = 1
        all_data_10min['pitch_speed_3'] = 1

        all_data_10min[time] = pd.to_datetime(all_data_10min[time])
        all_data_10min = all_data_10min.drop_duplicates([time])
        vars_10min = all_data_10min.columns.to_list()

        train = all_data_10min[:int(len(all_data_10min)*0.4)]


        vlist = [time, pbt1, pbt2, pbt3,  ws, ba1_speed, ba2_speed, ba3_speed]
        data = train.loc[:, vlist]

        data = data_preprocess(data)  ##插值

        td = pd.DataFrame()
        data = data.loc[(data[pbt1] > temp_threshold) & (data[pbt2] > temp_threshold) &  (data[pbt3] > temp_threshold)]

        data['pb_diff1'] = data[pbt1] - data[pbt2]
        data['pb_diff2'] = data[pbt1] - data[pbt3]
        data['pb_diff3'] = data[pbt2] - data[pbt3]

        if (np.mean(data['pb_diff1']) > 10) | (np.mean(data['pb_diff2']) > 10) | (np.mean(data['pb_diff3']) > 10):
            print('　not suitable for training')
        else:
            td = td.append(data)

        pb_fit_main(data, wt_id)


        pass
