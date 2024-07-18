import pandas as pd
import matplotlib.pyplot as plt
import pylab

readpath = '.\\result\\2021-01_result.pkl'
result = pd.read_pickle(readpath)
raw_data = result['raw_data']
# print(raw_data)
# time = raw_data['datetime']
# print(time)
# print(type(time))
data1 = raw_data['plot_data1']
data2 = raw_data['plot_data2']
#
analysis_data = result['analysis_data']
bpc = analysis_data['bpc']
apc = analysis_data['apc']
plt.close()
ax1 = plt.subplot(2, 1, 1)
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.scatter(data1['ws'], data1['gp'], c='r', marker='*', label='有意义的数据')
plt.scatter(data2['ws'], data2['gp'], c='b', marker='+', label='无效的数据')
plt.xlabel('风速')
plt.ylabel('有功功率')
# plt.xticks(rotation=30)
plt.title('预警原始数据展示')
# plt.plot(ps3, c='y', label='ps3')
plt.legend()

ax2 = plt.subplot(2, 1, 2)
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(bpc['wsbin'], bpc['power'], c='r', label='基线曲线')
plt.plot(apc['wsbin'], apc['power'], c='b', label='现有曲线')
plt.ylabel('风速')
plt.xlabel('有功功率')
# plt.xlim([-0.5, 0.9])
plt.legend()
plt.title('预警结果判据展示')
plt.show()