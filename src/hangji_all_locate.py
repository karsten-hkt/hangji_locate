# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/15 17:14
@Author : karsten
@File : hangji_all_locate.py
@Software: PyCharm
============================
"""
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
# 假设表层传播的速度为v m/s
v = 200
# 互相关间隔与采样点
maxlag = 2000
sample_rate = 1000
# 台站个数
S = len(rel_coordinates)
# 总的方程数量
N = int(S / 2 * (S - 1))
M = 2  # 未知数个数
# 权重矩阵
sigma_t = 1 / sample_rate  # 用采样间隔来给定
# 最大循环个数
stop_circle = 10
stop_critical = 0.0001
# 记录每个地震事件对应的事件名称和位置
event_locate = []

###############################
# 读取某个地震事件下所有的数据
event_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_quake'
# 遍历文件夹，并对每个地震进行逐个定位
for root, dirs, files in os.walk(event_path):
	for dir in dirs:
		full_dir_path = os.path.join(root, dir)
		print('开始处理文件夹：', dir)
		stn = obspy.Stream()
		for file in os.listdir(full_dir_path):
			file_path = os.path.join(full_dir_path, file)
			stn += obspy.read(file_path)
		stp = stn.copy()
		stp = locate_fun.waveform_processing(stp)
		# 假设地震位置为E0
		E0 = np.array([50.0, 50.0]).T
		E = []  # 记录地震位置的变化
		E.append(E0)
		R = []  # 记录误差的变化
		R.append(10000)  # 为了和E对齐，假设起始误差为10000
		for circle in range(stop_circle):
			G, d, W, corr = locate_fun.build_matrices(rel_coordinates, stp, E0, v, sigma_t, maxlag, sample_rate)
			delta_m = np.linalg.pinv(W @ G) @ W @ d  # 使用伪逆来求解
			# 更新地震位置估计
			E0 = E0 + delta_m.flatten()  # 确保delta_m是一维的
			residual = np.linalg.norm(G @ delta_m - d)
			# 检查误差，如果误差增加，则停止迭代
			if residual > R[circle] or np.abs(residual - R[circle]) < stop_critical:
				break
			else:
				E.append(E0.copy())  # 存储新的位置估计
				R.append(residual.copy())  # 存储新的误差
				# 打印或记录信息
				print(f"quake_name:{dir},Iteration {circle}: Delta M = {delta_m.T}, New E0 = {E0}, Residual = {residual}")
		event_locate.append([dir,E[-1][0], E[-1][1], R[-1]])

###########################
# 保存事件位置到output文件夹
output_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output'
# 保存事件位置到output文件夹
event_locate_df = pd.DataFrame(event_locate)
event_locate_df.columns = ['event_name','location_x', 'location_y', 'residual']
# 保存并不带序号
event_locate_df.to_csv(os.path.join(output_path, 'event_locate.csv'), index=False)