# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/20 10:34
@Author : karsten
@File : hangji_locate_Kestoration_test.py.py
@Software: PyCharm
============================
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/15 17:08
@Author : karsten
@File : hangji_locate.py
@Software: PyCharm
============================
"""
# 对其中一个地震事件而言

import numpy as np
import os
import obspy
import pandas as pd
import matplotlib.pyplot as plt
import locate_fun
import station_fun
###############################
# 读取某个地震事件下所有的数据
event_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_quake/2022-07-14T01:42:25.000Z'
stn = obspy.Stream()  # 初始化一个空的Stream对象
for root, dirs, files in os.walk(event_path):
	for file in files:
		file_path = os.path.join(event_path, file)
		stn += obspy.read(file_path)
stp = stn.copy()
stp = locate_fun.waveform_processing(stp)
# 读取台站的位置数据并转换为相对坐标
stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt', header=None, names=['station', 'lon', 'lat'], sep=',')
# 创建一个字典，以裁剪后的台站名称为键，经纬度为值
station_info = {row['station'].split('_')[0]: {'lat': row['lat'], 'lon': row['lon']}
				for index, row in stations.iterrows()}
stations['station'] = stations['station'].str.split('_').str.get(0)
ref_station = '22917'  # 选择一个参考台站
rel_coordinates = station_fun.rel_coordinates(station_info, ref_station)

###############################
# 基础参数设置
# 假设地震位置为E0
E0 = np.array([50.0, 50.0]).T
# 假设表层传播的速度为v m/s
v = 180
# 互相关间隔与采样点
maxlag = 2000
sample_rate = 1000
# 台站个数
S = len(rel_coordinates)
# 总的方程数量
N = int(S / 2 * (S - 1))
M = 2  # 未知数个数
# 权重矩阵
W = np.zeros((N, N))  # 具体使用互相关大小以及初定误差给定
corr = np.zeros((N, 1))  # 记录互相关大小
sigma_t = 1 / sample_rate  # 用采样间隔来给定
# 记录地震位置的变化
E = [E0]
# 记录误差的变化
R = [10000]
# 设定保留的奇异值个数
k = 2
# 最大循环个数
stop_circle = 10
stop_critical = 0.0001
###############################
# 开始构建矩阵
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
		print('反演实际的结果：')
		print(f"Iteration {circle}: Delta M = {delta_m.T}, New E0 = {E0}, Residual = {residual}")
# 迭代结束
#########################
# 使用理论的结果去计算实际的互相关
E_0_first = np.array([50.0, 50.0]).T
# 记录地震位置的变化
E_theoretical = [E_0_first]
# 记录误差的变化
R_theoretical = [10000]
# 给定地震的时间
earthquake_time = 0.0
# 根据理论的位置到台站的位置去计算到时
stations_theoretical_time = []
for station in rel_coordinates:
	arrival_time = locate_fun.dis(E0, np.array([station[1].astype(float), station[2].astype(float)]).T) / v
	stations_theoretical_time.append([station[0], arrival_time])
# 生成脉冲响应
st_theoretical = station_fun.generate_pulse_response(earthquake_time, stations_theoretical_time, sample_rate, 10)
stp_theoretical = st_theoretical.copy()
###########################
for circle in range(stop_circle):
	G_theoretical, d_theoretical, W_theoretical, corr_theoretical = (
		locate_fun.build_matrices(rel_coordinates, stp_theoretical, E_0_first, v, sigma_t, maxlag, sample_rate))
	delta_m_theoretical  = np.linalg.pinv(W_theoretical  @ G_theoretical ) @ W_theoretical  @ d_theoretical   # 使用伪逆来求解
	# 更新地震位置估计
	E_0_first = E_0_first + delta_m_theoretical.flatten()  # 确保delta_m是一维的
	residual = np.linalg.norm(G_theoretical @ delta_m_theoretical - d_theoretical)
	# 检查误差，如果误差增加，则停止迭代
	if residual > R_theoretical[circle] or np.abs(residual - R_theoretical[circle]) < stop_critical:
		break
	else:
		E_theoretical.append(E_0_first.copy())  # 存储新的位置估计
		R_theoretical.append(residual.copy())  # 存储新的误差
		# 打印或记录信息
		print('反演理论的结果：')
		print(f"Iteration {circle}: Delta M = {delta_m_theoretical.T}, New E0 = {E_0_first}, Residual = {residual}")
#################
#打印结果
print('实际的结果：',E0)
print('理论的结果：',E_0_first)
# 画图
plt.figure(figsize=(12, 8))
# 绘制各个台站
for station in rel_coordinates:
	plt.plot(station[1].astype(float), station[2].astype(float), 'o', markersize=5, color='yellow')
	plt.text(station[1].astype(float), station[2].astype(float), station[0])
# 绘制理论的位置
plt.plot(E_0_first[0], E_0_first[1], 'o', markersize=5, color='red',alpha = 0.5)
plt.text(E_0_first[0], E_0_first[1]+1, 'theoretical')
# 绘制实际的位置
plt.plot(E0[0], E0[1], 'o', markersize=5, color='blue', alpha = 0.5)
plt.text(E0[0], E0[1]-1, 'actual')
# 保存图像
plt.savefig('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/Kestoration_test.jpg',dpi=300)