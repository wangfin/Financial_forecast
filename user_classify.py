#!/usr/bin/env python
# @Time    : 2019/3/15 15:17
# @Author  : wb
# @File    : user_classify.py

# 将用户进行分类

# 先以最大单次的赎回量进行分类


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 通过pandas读取csv数据

# users = pd.read_csv('data/new_user_balance.csv', names=['user_id', 'report_date', 'tBalance', 'yBalance',
#                                                         'total_purchase_amt', 'direct_purchase_amt', 'purchase_bal_amt',
#                                                         'purchase_bank_amt', 'total_redeem_amt', 'consume_amt',
#                                                         'transfer_amt', 'tftobal_amt', 'tftocard_amt', 'share_amt'])

# users = pd.read_csv('data/new_user_balance.csv', names=['total_purchase_amt', 'total_redeem_amt'])

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

# num_list = [1.5, 0.6, 7.8, 6, 20.0, 11.5, 1.9, 2.3]
# plt.hist(list(users.user_id), bins=40, density=0, facecolor='blue', edgecolor='black')
# plt.show()

# num_list = [1.5,0.6,7.8,6]
# 用户ID user_id
# 今日总购买量 total_purchase_amt
# 今日总赎回量 total_redeem_amt


# plt.bar(range(len(num_list)), num_list)
# plt.show()

# 就不画图了，直接计算标准差
# print(list(df['total_purchase_amt']))

# print(type(df))
# print(df)

# pur_array = np.array(list(df['total_purchase_amt'])).astype(np.float)
# # print(pur_array)
# # 平均值
# pur_mean = np.mean(pur_array)
# # 标准差
# pur_std = np.std(pur_array)
# # 中位数
# pur_median = np.median(pur_array)
# print(pur_mean, pur_std, pur_median)

# 计算每个用户的申购赎回平均值

# 按照用户id合并统计次数
user = df.groupby(['user_id'], as_index=False)['user_id']

# print(user['cnt'].values)

# plt.hist(user['cnt'].values, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.show()

low_users = pd.DataFrame(columns=('user_id', 'report_date', 'tBalance', 'yBalance', 'total_purchase_amt',
                                  'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt',
                                  'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt',
                                  'share_amt', 'category1', 'category2', 'category3', 'category4'))
media_users = pd.DataFrame(columns=('user_id', 'report_date', 'tBalance', 'yBalance', 'total_purchase_amt',
                                  'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt',
                                  'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt',
                                  'share_amt', 'category1', 'category2', 'category3', 'category4'))
high_users = pd.DataFrame(columns=('user_id', 'report_date', 'tBalance', 'yBalance', 'total_purchase_amt',
                                  'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt',
                                  'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt',
                                  'share_amt', 'category1', 'category2', 'category3', 'category4'))
# 把Pandas读取的csv进行分解
index = 0

for i in user['cnt'].values:
    # 获取了每个用户的操作数据长度
    if i <= 100:
        # print(df[index:index+i])
        frames = [low_users, df[index:index+i]]
        low_users = pd.concat(frames)
        print(low_users)
    elif (i > 100 and i < 200):
        frames = [media_users, df[index:index + i]]
        media_users = pd.concat(frames)
        print(media_users)
    else:
        frames = [high_users, df[index:index + i]]
        high_users = pd.concat(frames)
        print(high_users)
    index += i

# print(low_users)

# 写入csv

low_users.to_csv("data/low_users.csv")
media_users.to_csv("data/media_users.csv")
high_users.to_csv("data/high_users.csv")

# with open("data/low_users.csv","w") as csvfile:
#     writer = csv.writer(csvfile)
#
#     #先写入columns_name
#     writer.writerow(["index","a_name","b_name"])
#     #写入多行用writerows
#     writer.writerows([[0,1,3],[1,2,3],[2,3,4]])



# user = df.groupby(by=['user_id'])
# print(user.as_matrix())
# user_size = user.size()
# new_user = user_size.reset_index(name='times')
# print(new_user)

# print(new_user.values)


# users_array = np.split(new_user.values, len(new_user.values))
# print(type(users_array[0]))

# for i in users_array:
#     print(type(i[0]))

