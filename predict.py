#!/usr/bin/env python
# @Time    : 2019/3/23 16:20
# @Author  : wb
# @File    : predict.py

from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

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
pur_series = pd.Series(total_purchase['total_purchase_amt'].values, index=total_purchase['report_date'])[214:]
consume_series = pd.Series(consume['consume_amt'].values, index=consume['report_date'])[214:]
transfer_series = pd.Series(transfer['transfer_amt'].values, index=transfer['report_date'])[214:]
# pur_series.plot(figsize=(12,8))
# plt.show()

# 对时间序列进行差分

# 分割成2014/08之前和2014/08之后，之后用于测试
# pur_series_train = pur_series[:396]
# pur_series_test = pur_series[396:]

# 检测一阶差分的平稳情况
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(111)
# diff1 = pur_series_train.diff(1)
# diff1.plot(ax=ax1)
# plt.show()

# 二阶差分
# fig = plt.figure(figsize=(12, 8))
# ax2 = fig.add_subplot(111)
# diff2 = pur_series_train.diff(2)
# diff2.plot(ax=ax2)

# 一阶单位根检验
# sm.tsa.stattools.adfuller(diff1[1:])

# 二阶单位根检验
# sm.tsa.stattools.adfuller(diff2[2:])

# print(sm.tsa.stattools.adfuller(diff1[1:]))

# 使用一阶差分
# diff1 = pur_series_train.diff(1)
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(pur_series_train, lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(pur_series_train, lags=40, ax=ax2)
# plt.show()

# arma_mod20 = sm.tsa.ARMA(pur_series_train,(7,0)).fit()
# print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
# arma_mod30 = sm.tsa.ARMA(pur_series_train,(0,1)).fit()
# print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
# arma_mod40 = sm.tsa.ARMA(pur_series_train,(7,1)).fit()
# print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
# arma_mod50 = sm.tsa.ARMA(pur_series_train,(8,0)).fit()
# print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)

# pur_series = pur_series.diff(1).dropna()
# consume_series = consume_series.diff(1).dropna()
# transfer_series = transfer_series.diff(1).dropna()

purchase_arma = sm.tsa.ARMA(pur_series,(8,0)).fit()
consume_arma = sm.tsa.ARMA(pur_series,(8,0)).fit()
transfer_arma = sm.tsa.ARMA(pur_series,(7,0)).fit()

# 残差
# purchase_resid = purchase_arma.resid
# consume_resid = consume_arma.resid
# transfer_resid = transfer_arma.resid

# 选取8 0
# 观察是否符合正态分布
# resid = arma_mod50.resid  # 残差

# 检查是否符合正态分布
# fig = plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111)
# fig = qqplot(resid, line='q', ax=ax, fit=True)
# plt.show()

# 对（8,0）模型产生的残差做自相关图
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
# plt.show()

predict_sunspots = purchase_arma.predict('2014-09-01', '2014-09-30', dynamic=True)
consume_sunspots = consume_arma.predict('2014-09-01', '2014-09-30', dynamic=True)
transfer_sunspots = transfer_arma.predict('2014-09-01', '2014-09-30', dynamic=True)

# print(predict_sunspots)

data = {}
date = []
pur = []
red = []
# final_data_pd = pd.DataFrame(columns=('report_date', 'purchase', 'redeem'))

for i in range(30):
    date.append(20140901+i)
    pur.append(int(predict_sunspots[i]))
    red.append(int(consume_sunspots[i] + transfer_sunspots[i]))
    # frames = [final_data, 20140901+i, int(predict_sunspots[i]), int(consume_sunspots[i] + transfer_sunspots[i])]
    # print(predict_sunspots[i])
data['report_date'] = date
data['purchase'] = pur
data['redeem'] = red
final_data_pd = pd.DataFrame(data=data, columns=('report_date', 'purchase', 'redeem'))
final_data_pd.to_csv("data/tc_comp_predict_table.csv")
# print(final_data_pd)
# print(pur_series)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax = consume_series.plot(ax=ax)
# consume_sunspots.plot(ax=ax)
# plt.show()