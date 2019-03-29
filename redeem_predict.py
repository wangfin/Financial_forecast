#!/usr/bin/env python
# @Time    : 2019/3/25 9:17
# @Author  : wb
# @File    : redeem_predict.py

from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf

# 实际预测

# 通过pandas读取csv数据

f = open('data/new_user_balance.csv')
reader = pd.read_csv(f, sep=',', iterator=True)
loop = True
chunkSize = 100000
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
df = pd.concat(chunks, ignore_index=True)

'''
对所有数据进行汇总，然后平均到每个人身上
'''

# 1.对所有数据进行汇总

# 按照时间日期合并统计次数

# 总购买量（按时间排序）
total_purchase = df.groupby(['report_date'], as_index=False)['total_purchase_amt'].sum()
# 今日消费总量
consume = df.groupby(['report_date'], as_index=False)['consume_amt'].sum()
# 今日转出总量
transfer = df.groupby(['report_date'], as_index=False)['transfer_amt'].sum()

# print(total_purchase)

# 将第一个时间列进行改造成 2013-07-01这样的格式
total_purchase['report_date'] = pd.to_datetime(total_purchase['report_date'], format='%Y%m%d')
consume['report_date'] = pd.to_datetime(consume['report_date'], format='%Y%m%d')
transfer['report_date'] = pd.to_datetime(transfer['report_date'], format='%Y%m%d')
# total_purchase.info()
# print(total_purchase['report_date'])

# 获取到了这三列的日平均数据，总共427天

# 使用时间序列
pur_series = pd.Series(total_purchase['total_purchase_amt'].values, index=total_purchase['report_date'])
consume_series = pd.Series(consume['consume_amt'].values, index=consume['report_date'])
transfer_series = pd.Series(transfer['transfer_amt'].values, index=transfer['report_date'])
# consume_series.plot(figsize=(12,8))
# plt.show()

# 对时间序列进行差分

# 分割成2014/08之前和2014/08之后，之后用于测试
consume_series_train = consume_series[214:396]
consume_series_test = consume_series[396:]

# 检测一阶差分的平稳情况
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(111)
# diff1 = consume_series_train.diff(1).dropna()
# diff1.plot(ax=ax1)
# plt.show()

# 二阶差分
# fig = plt.figure(figsize=(12, 8))
# ax2 = fig.add_subplot(111)
# diff2 = consume_series_train.diff(2)
# diff2.plot(ax=ax2)
# plt.show()

# 一阶单位根检验
# sm.tsa.stattools.adfuller(diff1[1:])

# 二阶单位根检验
# sm.tsa.stattools.adfuller(diff2[2:])

# print(sm.tsa.stattools.adfuller(diff1[1:]))
# t = sm.tsa.stattools.adfuller(diff1, )
# output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
# output['value']['Test Statistic Value'] = t[0]
# output['value']['p-value'] = t[1]
# output['value']['Lags Used'] = t[2]
# output['value']['Number of Observations Used'] = t[3]
# output['value']['Critical Value(1%)'] = t[4]['1%']
# output['value']['Critical Value(5%)'] = t[4]['5%']
# output['value']['Critical Value(10%)'] = t[4]['10%']
# print(output)
#
# plot_acf(diff1)
# plot_pacf(diff1)
# plt.show()
#
# r,rac,Q = sm.tsa.acf(diff1, qstat=True)
# prac = pacf(diff1,method='ywmle')
# table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]
# table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])
#
# print(table)

# p,d,q = (3,1,1)
# arma_mod = sm.tsa.ARMA(diff1,(p,d,q)).fit(disp=-1,method='mle')
# summary = (arma_mod.summary2(alpha=.05, float_format="%.8f"))
# print(summary)

diff1 = pur_series.diff(1).dropna()
(p, q) =(sm.tsa.arma_order_select_ic(diff1,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
print(p,q)
# 这里需要设定自动取阶的 p和q 的最大值，即函数里面的max_ar,和max_ma。ic 参数表示选用的选取标准，这里设置的为aic,当然也可以用bic。
# 然后函数会算出每个 p和q 组合(这里是(0,0)~(3,3)的AIC的值，取其中最小的,这里的结果是(p=0,q=1)。

# arma_mod = sm.tsa.ARMA(diff1,(3,1,1)).fit(disp=-1,method='mle')
# resid = arma_mod.resid
# t = sm.tsa.stattools.adfuller(resid)
# output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
# output['value']['Test Statistic Value'] = t[0]
# output['value']['p-value'] = t[1]
# output['value']['Lags Used'] = t[2]
# output['value']['Number of Observations Used'] = t[3]
# output['value']['Critical Value(1%)'] = t[4]['1%']
# output['value']['Critical Value(5%)'] = t[4]['5%']
# output['value']['Critical Value(10%)'] = t[4]['10%']
# print(output)

# 使用一阶差分
# consume_series_train = consume_series_train.diff(1)
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(consume_series_train, lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(consume_series_train, lags=40, ax=ax2)
# plt.show()

# arma_mod20 = sm.tsa.ARMA(consume_series_train,(7,0)).fit()
# print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
# arma_mod30 = sm.tsa.ARMA(consume_series_train,(0,1)).fit()
# print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
# arma_mod40 = sm.tsa.ARMA(consume_series_train,(7,1)).fit()
# print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
# arma_mod50 = sm.tsa.ARMA(consume_series_train,(8,0)).fit()
# print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)


# 选取7 1
# 观察是否符合正态分布
# resid = arma_mod40.resid  # 残差

# 检查是否符合正态分布
# fig = plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111)
# fig = qqplot(resid, line='q', ax=ax, fit=True)
# plt.show()

# 做DW检验
# print(sm.stats.durbin_watson(arma_mod40.resid.values))

# 对（7,1）模型产生的残差做自相关图
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
# plt.show()

# consume_sunspots = arma_mod40.predict('2014-08-01', '2014-08-31', dynamic=True)
# print(consume_sunspots)
# print(consume_series_test)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax = consume_series[214:].plot(ax=ax)
# consume_sunspots.plot(ax=ax)
# plt.show()

# arma_model = sm.tsa.ARMA(consume_series_train,(3,1)).fit(disp=-1,maxiter=100)
# predict_data = arma_model.predict('2014-08-01', '2014-08-31', dynamic=True)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax = consume_series[214:].plot(ax=ax)
# predict_data.plot(ax=ax)
# plt.show()