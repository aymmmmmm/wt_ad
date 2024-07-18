#! /usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:33:17 2019

@author: Minxuan
"""

import numpy as np
import pandas as pd
import os
import pickle
import re
from tqdm import tqdm
import pymongo
import pandas as pd
import datetime
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt




output_path = r'../Resource/control_cabinet_temp'
t = 'time' #时间
cct = 'control_carbin_temp' #控制柜温度
wtid = 'turbineName'
gs = 'generator_rotating_speed'
gp = 'power'

def feature_selection(data, target):
    import xgboost as xgb
#    from xgboost import plot_importance
    from sklearn.feature_selection import f_regression
    from scipy.stats import pearsonr, spearmanr
    from scipy.spatial.distance import correlation
    import itertools
    import warnings
    warnings.filterwarnings('ignore')
    # XGBoost
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, objective='reg:gamma')
    model.fit(data, target)
    I = model.feature_importances_
    index = []
    for i in sorted(I, reverse=True)[:10]:
        index.extend(np.where(I == i)[0])
    fea_1 = data.columns[index]
    # F Regression
    F = f_regression(data, target)
    index = []
    for i in sorted(F[0], reverse=True)[:10]:
        index.extend(np.where(F[0] == i)[0])
    fea_2 = data.columns[index]
    # Correlation
    pears = []
    spear = []
    discor = []
    for col in data.columns:
        pears.append(pearsonr(data[col].values, target.values)[0])
        spear.append(spearmanr(data[col].values, target.values)[0])
        discor.append(correlation(data[col].values, target.values))
    index = []
    for i in sorted(np.abs(pears), reverse=True)[:10]:
        index.extend(np.where(np.abs(pears) == i)[0])
    fea_3 = data.columns[index]
    index = []
    for i in sorted(np.abs(spear), reverse=True)[:10]:
        index.extend(np.where(np.abs(spear) == i)[0])
    fea_4 = data.columns[index]
    index = []
    for i in sorted(np.abs(discor), reverse=False)[:10]: # 选取数值小的前10个
        index.extend(np.where(np.abs(discor) == i)[0])
    fea_5 = data.columns[index]
    # Combine
    flattened = list(itertools.chain(*[list(fea_1), list(fea_2), list(fea_3), list(fea_4), list(fea_5)]))
    print(flattened)
    c = [flattened.count(i) for i in set(flattened)]
    print(c)
    count = dict(zip(set(flattened), c))
    feature_selected = [key for key, val in count.items() if val >= 3]
    return feature_selected

def ensemble_fit(X, y):
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.neural_network import MLPRegressor
    #    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    #    from keras.models import Sequential
    #    from keras.layers import Dense
    
    kfold = KFold(n_splits=3, shuffle=True)
    mse = {}
    # MLP in sklearn
    model_mlp = MLPRegressor(solver='adam', hidden_layer_sizes=(32, 32, 32), random_state=1001)
    print('*******************Start Iteration: MLP*********************')
    results = cross_val_score(model_mlp, X, y, cv=kfold)
    mse['MLP'] = np.abs(results.mean())
    print("MLP Results: %.2f (%.2f) MSE" % (np.abs(results.mean()), results.std()))
    # RF
    rf_params = {
        'max_features':2, 
        'min_samples_split':4,
        'n_estimators':100, 
        'min_samples_leaf':2
    }
    model_rf = RandomForestRegressor(**rf_params)
    print('*******************Start Iteration: RF*********************')
    results = cross_val_score(model_rf, X, y, cv=kfold)
    mse['RF'] = np.abs(results.mean())
    print("RF Results: %.2f (%.2f) MSE" % (np.abs(results.mean()), results.std()))
    # XGBoost
    xgb_params = {
        'eta': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'silent': 1,
        'n_estimators': 100
    }
    model_xgb = XGBRegressor(**xgb_params)
    print('*******************Start Iteration: XGB*********************')
    results = cross_val_score(model_xgb, X, y, cv=kfold)
    mse['XGB'] = np.abs(results.mean())
    print("XGBoost Results: %.2f (%.2f) MSE" % (np.abs(results.mean()), results.std()))
    # GBR
    gbr_params = {
        'learning_rate': 0.1, 
        'n_estimators': 100, 
        'max_features': 'log2', 
        'min_samples_split': 2, 
        'max_depth': 10
    }
    model_gbr =GradientBoostingRegressor(**gbr_params)
    print('*******************Start Iteration: GBR*********************')
    results = cross_val_score(model_gbr, X, y, cv=kfold)
    mse['GBR'] = np.abs(results.mean())
    print("GBR Results: %.2f (%.2f) MSE" % (np.abs(results.mean()), results.std()))
    # NN in Keras
#     def baseline_model():
#         # create model
#         model = Sequential()
#         model.add(Dense(X.shape[1], input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
#         model.add(Dense(32, kernel_initializer='normal',activation='relu'))
#         model.add(Dense(32, kernel_initializer='normal',activation='relu'))
#         model.add(Dense(32, kernel_initializer='normal',activation='relu'))
#         model.add(Dense(1, kernel_initializer='normal'))
#         # Compile model
#         model.compile(loss='mean_squared_error', optimizer='adam')
#         return model
#     model_keras = KerasRegressor(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)
#     print('*******************Start Iteration: NNKeras*********************')
#     results = cross_val_score(model_keras, X, y, cv=kfold)
#     mse['NN'] = np.abs(results.mean())
#     print("NNKeras Results: %.2f (%.2f) MSE" % (np.abs(results.mean()), results.std()))
    # LGB
    lgb_params = {
        'boosting_type': 'gbdt', 
        'metric': 'mse', 
        'max_depth': -1, 
        'learning_rate': 0.1, 
        'verbose': 0, 
        'n_estimators': 100
    }
    model_lgb = LGBMRegressor(**lgb_params)
    print('*******************Start Iteration: LGB*********************')
    results = cross_val_score(model_lgb, X, y, cv=kfold)
    mse['LGB'] = np.abs(results.mean())
    print("LGB Results: %.2f (%.2f) MSE" % (np.abs(results.mean()), results.std()))
    model = {'MLP':model_mlp, 'RF':model_rf, 'XGB':model_xgb, 'GBR':model_gbr, 'LGB':model_lgb}
    model_list = sorted(mse, key=mse.get)
    print('Best 3 Models Selected: {}, {}, {}'.format(model_list[0], model_list[1], model_list[2]))
    model_1, model_2, model_3 = model[model_list[0]], model[model_list[1]], model[model_list[2]]

    ####固定模型  jimmy-20221109
    model_1 = model['LGB']
    model_2 = model['XGB']
    model_3 = model['RF']
    
    return model_1.fit(X, y), model_2.fit(X, y), model_3.fit(X, y)

def ensemble_predict(data, model):
    pred_1, pred_2, pred_3 = model[0].predict(data), model[1].predict(data), model[2].predict(data)
    ensemble = (pred_1 + pred_2 + pred_3) / 3
    return ensemble

def cctrain_main(data):
    global output_path
    global t
    global cct
    global wtid
    
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    result = {}
    df = data[(data[cct] < 50)] # 超过50度为异常数据
    df = df[(df[cct] >0)]
    
    
    wt = data[wtid].values[0]
    print(wt)
    

    # fea = feature_selection(df.drop([cct, t, wtid], axis=1), df[cct])
    df['cct_shift']= df[cct].shift(1) ####增加上一时刻的温度
    # df= df[df['power']>10] ###功率做筛选
    df= df.dropna()


    fea = ['power','cabin_temp', 'generator_rotating_speed', 'grid_current__A', 'cct_shift']
    print(fea)
    mod_1, mod_2, mod_3 = ensemble_fit(df[fea], df[cct])

    pred = ensemble_predict(df[fea], [mod_1, mod_2, mod_3])
    residual = df[cct].values - pred

    # df = df[1000:1300]
    # plt.subplot(4,1,1)
    # plt.scatter(df[gs], df[gp])
    # plt.subplot(4,1,2)
    # plt.plot(residual[1000:1300])
    # plt.subplot(4,1,3)
    # plt.plot(df[cct])
    # plt.show()

    plt.plot(df[cct].values,label="truth")
    plt.plot(pred,label="pred")
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('cct')
    # plt.xlim(1000, 1500)
    plt.title('the JF cct of pred and truth')
    # plt.show()
    plt.savefig(output_path + '/{}_pred.jpg'.format(wt),dpi=plt.gcf().dpi, bbox_inches='tight' )
    plt.clf()

    with open(output_path + '/{}_Residual.pkl'.format(wt), 'wb') as file:
        pickle.dump(residual, file)

    with open(output_path + '/{}_Model_1.pkl'.format(wt), 'wb') as file:
        pickle.dump(mod_1, file)

    with open(output_path + '/{}_Model_2.pkl'.format(wt), 'wb') as file:
        pickle.dump(mod_2, file)

    with open(output_path + '/{}_Model_3.pkl'.format(wt), 'wb') as file:
        pickle.dump(mod_3, file)

    with open(output_path + '/{}_Feature.pkl'.format(wt), 'wb') as file:
        pickle.dump(fea, file)

    result['status_code'] = '000'


if __name__ == '__main__':


    data_path = 'C:\\Project\\03_AeroImprovement\\05_华润风机SCADA数据分析_202111\\华润风机数据\\数据\\offline_analysis\\10min\\3.2MW\\'

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

        all_data_10min[t] = pd.to_datetime(all_data_10min[t])  ##插值的时候不能shi datatimeStamp
        all_data_10min = all_data_10min.drop_duplicates([t])
        vars_10min = all_data_10min.columns.to_list()
        all_data_10min= all_data_10min.drop(['Unnamed: 0'], axis=1)

        train = all_data_10min[:int(0.4 * len(all_data_10min))]

        varlist = [t, wtid, 'power','cabin_temp', 'generator_rotating_speed', cct ,'grid_current_A']
        train = train [varlist]

        df = train[(train[cct] < 55)] # 超过50度为异常数据
        df= df.dropna()
        cctrain_main(df)
        pass
    
    
    
    