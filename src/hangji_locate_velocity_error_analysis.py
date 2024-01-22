# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/18 15:49
@Author : karsten
@File : hangji_locate_velocity_error_analysis.py.py
@Software: PyCharm
============================
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import locate_fun
import station_fun

# 基础参数设置
# 假设表层传播的速度为v m/s
v = 200
# 互相关间隔与采样点
maxlag = 2000
sample_rate = 1000
# 延迟时间误差
velocity_error = 50
# 读取每个台站的编号以及相对的经纬度
stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt', header=None, names=['station', 'lon', 'lat'], sep=',')
# 创建一个字典，以裁剪后的台站名称为键，经纬度为值
station_info = {row['station'].split('_')[0]: {'lat': row['lat'], 'lon': row['lon']}
				for index, row in stations.iterrows()}
stations['station'] = stations['station'].str.split('_').str.get(0)
ref_station = '22917'  # 选择一个参考台站
rel_coordinates = station_fun.rel_coordinates(station_info, ref_station)
# 给定一个理论的地震位置
E_theritical = np.array([30.0, 30.0]).T
# 给定地震的时间
earthquake_time = 0.0

# 计算各个台站之间的互相关
# 台站个数
S = len(rel_coordinates)
# 总的方程数量
N = int(S / 2 * (S - 1))
M = 2  # 未知数个数
# 权重矩阵
sigma_t = 1 / sample_rate  # 用采样间隔来给定
# 最大循环个数
stop_circle = 10
stop_critical = 10**-4
# 使用蒙特卡洛误差传播分析各个点的误差
# 重复次数
N_circle = 200
m_i = [] # 存储每次迭代的位置估计
for tmp in range(N_circle):
	# 给定一个带有误差的速度
	v_error = v + np.random.normal(0,velocity_error)
	# 根据理论的位置到台站的位置去计算到时
	stations_theoretical_time = []
	for station in rel_coordinates:
		arrival_time = locate_fun.dis(E_theritical, np.array([station[1].astype(float), station[2].astype(float)]).T) / v_error
		stations_theoretical_time.append([station[0], arrival_time])
	# 生成脉冲响应
	st = station_fun.generate_pulse_response(earthquake_time, stations_theoretical_time, sample_rate, 10)
	stp = st.copy()
	E0 = np.array([50.0, 50.0]).T
	E = [E0]  # 记录地震位置的变
	R = [10000]  # 记录误差的变化
	# 假设地震位置为E0
	E0 = np.array([50.0, 50.0]).T
	for circle in range(stop_circle):
		G, d, W, corr = locate_fun.build_matrices(rel_coordinates, stp, E0, v, sigma_t, maxlag, sample_rate)
		delta_m = np.linalg.pinv(W @ G) @ W @ d  # 使用伪逆来求解
		# 更新地震位置估计
		E0 = E0 + delta_m.flatten()  # 确保delta_m是一维的
		residual = np.linalg.norm(G @ delta_m - d)
		# 检查误差，如果误差增加，则停止迭代
		if residual > R[circle] or np.abs(residual - R[circle]) < stop_critical:
			print(f"quake_name:{dir},Iteration {circle}: Delta M = {delta_m.T}, New E0 = {E0}, Residual = {residual}")
			break
		else:
			E.append(E0.copy())  # 存储新的位置估计
			R.append(residual.copy())  # 存储新的误差
			# 打印或记录信息
			# print(f"quake_name:{dir},Iteration {circle}: Delta M = {delta_m.T}, New E0 = {E0}, Residual = {residual}")
	m_i.append([E0[0], E0[1]])
m_i_mean = np.mean(m_i, axis=0)
A = np.array(m_i) - m_i_mean
m_i_covariance = A.T @ A / N_circle
A_therotical = np.array(m_i) - E_theritical
m_i_covariance_therotical = A_therotical.T @ A_therotical / N_circle
print('平均值为: ', m_i_mean, '模型的误差传播方差矩阵为: ', m_i_covariance)
print('理论值为: ', E_theritical, '模型和理论值的方差矩阵为: ', m_i_covariance_therotical)