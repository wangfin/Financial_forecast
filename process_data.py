#!/usr/bin/env python
# @Time    : 2019/3/5 16:29
# @Author  : wb
# @File    : process_data.py

# 预处理数据

# 处理数据

# 首先处理 user_balacnce_table表

import csv

# 读取csv至字典
# 获取用户余额宝进出项表
userbalancetable = open("data/user_balance_table.csv", "r")
reader = csv.reader(userbalancetable)

# 总的数组，数组里面是这些单个的数据
users = []
for item in reader:
    # 忽略第一行
    if reader.line_num == 1:
        continue
    # 建立字典
    user_balance = {}

    user_balance['user_id'] = int(item[0])
    user_balance['report_date'] = item[1]
    user_balance['tBalance'] = item[2]
    user_balance['yBalance'] = item[3]
    user_balance['total_purchase_amt'] = item[4]
    user_balance['direct_purchase_amt'] = item[5]
    user_balance['purchase_bal_amt'] = item[6]
    user_balance['purchase_bank_amt'] = item[7]
    user_balance['total_redeem_amt'] = item[8]
    user_balance['consume_amt'] = item[9]
    user_balance['transfer_amt'] = item[10]
    user_balance['tftobal_amt'] = item[11]
    user_balance['tftocard_amt'] = item[12]
    user_balance['share_amt'] = item[13]
    if item[14] != '':
        user_balance['category1'] = item[14]
    else:
        user_balance['category1'] = 0
    if item[15] != '':
        user_balance['category2'] = item[15]
    else:
        user_balance['category2'] = 0
    if item[16] != '':
        user_balance['category3'] = item[16]
    else:
        user_balance['category3'] = 0
    if item[17] != '':
        user_balance['category4'] = item[17]
    else:
        user_balance['category4'] = 0

    users.append(user_balance)

userbalancetable.close()

# 多级排序
users = sorted(users, key=lambda e: (e.__getitem__('user_id'), e.__getitem__('report_date')))

for user in users:
    user['user_id'] = str(user['user_id'])


# 获取用户信息表
# userprofiletable = open("data/user_profile_table.csv", "r")
# reader = csv.reader(userprofiletable)
#
# for item in reader:
#     # 忽略第一行
#     if reader.line_num == 1:
#         continue
#
#     # 将两张表的信息根据user_id合并
#     for i in users:
#         if item[0] == i['user_id']:
#             i['sex'] = item[1]
#             i['city'] = item[2]
#             i['constellation'] = item[3]
#
# userprofiletable.close()

# print(len(users))

# print(len(user_pros))

# for user in users[:10]:
#     print(user.values())

# 文件头，一般就是数据名
fileHeader = users[0].keys()

# 写入数据
'''
Python中的csv的writer，打开文件的时候，要小心，

要通过binary模式去打开，即带b的，比如wb，ab+等

而不能通过文本模式，即不带b的方式，w,w+,a+等，否则，会导致使用writerow写内容到csv中时，产生对于的CR，导致多余的空行。
'''
csvFile = open("data/new_user_balance.csv", "w", newline='')
writer = csv.writer(csvFile)

# 写入的内容都是以列表的形式传入函数
writer.writerow(fileHeader)
for user in users:
    writer.writerow(user.values())

csvFile.close()
