# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/20 17:21
@Author : karsten
@File : hangji_locate_all_variance.py.py
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
stop_circle = 20
stop_critical = 0.0001
N_circle = 100 # 误差传播矩阵计算次数
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
		E0 = np.array([0.0, 50.0]).T
		E = [E0]  # 记录地震位置的变化
		R = [10000]  # 记录误差的变化
		m_i = []  # 存储每次迭代的位置估计
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
		print('地震位置为：', E0)
		# 读取到时差数据
		lag_time = locate_fun.lag_time_with_station(rel_coordinates, stp, maxlag, sample_rate)
		lag_times_with_error = lag_time
		valid_indices = ~np.isnan(lag_times_with_error[:, 1])
		# 模型方差估计判断，使用蒙特卡洛误差传播分析
		for tmp in range(N_circle):
			E_var = np.array([0.0, 50.0]).T
			R_var = [10000]  # 记录误差的变化
			#data_error = np.random.normal(0, 1 / sample_rate / lag_times_with_error[valid_indices, 0])
			data_error = np.random.normal(0, 0.005)
			lag_times_with_error[valid_indices, 1] = lag_times_with_error[valid_indices, 1] + data_error
			for cir in range(stop_circle):
				G_var, d_var, W_var, corr_var = locate_fun.build_matrices_with_lag_time_nan(rel_coordinates, lag_times_with_error, E_var, v,sigma_t)
				delta_m_var = np.linalg.pinv(W_var @ G_var) @ W_var @ d_var  # 使用伪逆来求解
				E_var = E_var + delta_m_var.flatten()  # 确保delta_m是一维的
				residual_var = np.linalg.norm(G_var @ delta_m_var - d_var)
				# 检查误差，如果误差增加，则停止迭代
				if np.abs(residual_var - R_var[cir]) < stop_critical:
					break
				else:
					R_var.append(residual_var.copy())  # 存储新的误差
			m_i.append([E_var[0], E_var[1]])
		# 误差传播急诊计算
		m_i_mean = np.mean(m_i, axis=0)
		A = np.array(m_i) - m_i_mean
		m_i_covariance = A.T @ A / N_circle
		event_locate.append([dir, E[-1][0], E[-1][1], m_i_covariance[0,0], m_i_covariance[1, 1], R[-1]])
		# 打印或记录信息
		print(f"Event = {E0}, covariance = {m_i_covariance}, Residual = {R[-1]}")

###########################
# 保存事件位置到output文件夹
output_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output'
# 保存事件位置到output文件夹
event_locate_df = pd.DataFrame(event_locate)
event_locate_df.columns = ['event_name','location_x', 'location_y', 'x_covariance', 'y_covariance','residual']
# 保存并不带序号
event_locate_df.to_csv(os.path.join(output_path, 'event_locate_variance_0.005.csv'), index=False)