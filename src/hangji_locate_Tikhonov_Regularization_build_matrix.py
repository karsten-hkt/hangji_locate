# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/22 14:10
@Author : karsten
@File : hangji_locate_Tikhonov_Regularization_build_matrix.py.py
@Software: PyCharm
============================
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import obspy
import pandas as pd
import locate_fun
import station_fun

###############################
# 读取每个台站的编号以及相对的经纬度
stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt', header=None, names=['station', 'lon', 'lat'], sep=',')
# 创建一个字典，以裁剪后的台站名称为键，经纬度为值
station_info = {row['station'].split('_')[0]: {'lat': row['lat'], 'lon': row['lon']}
				for index, row in stations.iterrows()}
stations['station'] = stations['station'].str.split('_').str.get(0)
ref_station = '22917'  # 选择一个参考台站
rel_coordinates = station_fun.rel_coordinates(station_info, ref_station)
###############################
# 基础参数设置
S = len(rel_coordinates)
K = 20  # 总地震个数
# 总的方程数量
N = int(S / 2 * (S - 1)) * K
M = 3 * K  # 未知数个数,x,y,v以及地震个数
# 构建矩阵
d = np.zeros((N, 1))
m = np.zeros((M, 1))
# 对m给定起始位置以及速度
for i in range(M):
    if i % 3 == 0:
        m[i] = 0
    elif i % 3 == 1:
        m[i] = 50
    else:
        m[i] = 200
W = np.zeros((N, N))
# 互相关间隔与采样点
maxlag = 2000
sample_rate = 1000
sigma_t = 1 / sample_rate  # 用采样间隔来给定
stop_circle = 10
stop_critical = 0.0001
event_name = []
###############################
# 读取所有的数据
event_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_quake'
event_loc = 0 # 用来记录事件的位置
for root, dirs, files in os.walk(event_path):
    for dir in sorted(dirs):  # 确保目录按字母顺序处理
        full_dir_path = os.path.join(root, dir)
        event_name.append(dir)
        print('开始处理文件夹：', dir)
        stn = obspy.Stream()
        for file in sorted(os.listdir(full_dir_path)):  # 确保文件按字母顺序处理
            file_path = os.path.join(full_dir_path, file)
            stn += obspy.read(file_path)
        stp = stn.copy()
        stp = locate_fun.waveform_processing(stp)
        for i in range(S):
            for j in range(i + 1, S):
                cnt = int((i + 1) * ((S - i) + S) / 2 + j - i - 1 - len(
                    rel_coordinates)) + event_loc*int(S / 2 * (S - 1))
                station_i = rel_coordinates[i, 0]
                station_j = rel_coordinates[j, 0]
                # 对应的台站信息
                s_i = np.array([rel_coordinates[i, 1], rel_coordinates[i, 2]], dtype='float64')
                s_j = np.array([rel_coordinates[j, 1], rel_coordinates[j, 2]], dtype='float64')
                # 对应的波形信息
                sw_i = stp.select(station=station_i)
                sw_j = stp.select(station=station_j)
                # 需要判断是否为空才进行下一步
                if len(sw_i) > 0 and len(sw_j) > 0:
                    w_i = sw_i[0]
                    w_j = sw_j[0]
                    maxcor, lagtime = locate_fun.lag_time(w_i, w_j, maxlag, sample_rate)
                    if maxcor > 0.3:
                        W[cnt, cnt] = maxcor / sigma_t
                        d[cnt] = lagtime
        event_loc += 1
###############################
# 将构建的矩阵保存起来
np.save('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/d.npy', d)
np.save('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/W.npy', W)
np.save('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/event_name.npy', event_name)