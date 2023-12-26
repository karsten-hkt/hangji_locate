# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/22 17:17
@Author : karsten
@File : hangji_locate_Tikhonov_Regularization_alpha.py.py
@Software: PyCharm
============================
"""
import numpy as np
import pandas as pd
import locate_fun
import station_fun
import matplotlib.pyplot as plt
# 读取各个矩阵
d = np.load('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/d.npy')
W = np.load('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/W.npy')
event_name = np.load('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/event_name.npy')
###############################
# 读取每个台站的编号以及相对的经纬度
stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt', header=None, names=['station', 'lon', 'lat'], sep=',')
# 创建一个字典，以裁剪后的台站名称为键，经纬度为值
station_info = {row['station'].split('_')[0]: {'lat': row['lat'], 'lon': row['lon']}
				for index, row in stations.iterrows()}
stations['station'] = stations['station'].str.split('_').str.get(0)
ref_station = '22917'  # 选择一个参考台站
rel_coordinates = station_fun.rel_coordinates(station_info, ref_station)

# 基础参数设置
S = len(rel_coordinates)
K = 20  # 总地震个数
# 总的方程数量
N = int(S / 2 * (S - 1)) * K
M = 3 * K  # 未知数个数,x,y,v以及地震个数
# 构建矩阵
J = np.zeros((N, M))
G_m = np.zeros((N, 1))
G_now = np.zeros((N, 1))
m_0 = np.zeros((M, 1))
v0 = np.zeros((M, 1))
alpha = np.logspace(-10, -3, 50)
beta = 0
# 对m给定起始位置以及速度
for i in range(M):
	if i % 3 == 0:
		m_0[i] = 0
		v0[i] = 0
	elif i % 3 == 1:
		m_0[i] = 50
		v0[i] = 0
	else:
		m_0[i] = 200
		v0[i] = 200
# 构建L
L = np.zeros((K-1,M))
for i in range(K-1):
	L[i,3*i:3*i+3] = 1
	L[i,3*i+3:3*i+6] = -1
stop_critical = 0.001 * K
stop_circle = 50
stop_critical_m = 0.001 * K
v_length = []
G_m_d_record = []
for a_loha in alpha:
	vk = v0.copy()
	vk_new = v0.copy()
	residual = [10000]
	delta_m_record = [0]
	m = m_0.copy()
	for circle in range(stop_circle):
		for k in range(K):
			x = m[k * 3][0]
			y = m[k * 3 + 1][0]
			event_xy = m[k * 3:k * 3 + 2].flatten()
			enent_v = m[k * 3 + 2][0]
			vk[k * 3 + 2] = m[k * 3 + 2][0]
			for i in range(S):
				for j in range(i + 1, S):
					cnt = int((i + 1) * ((S - i) + S) / 2 + j - i - 1 - S + k * int(S / 2 * (S - 1)))
					station_i = rel_coordinates[i, 0]
					station_j = rel_coordinates[j, 0]
					# 对应的台站信息
					s_i = np.array([rel_coordinates[i, 1], rel_coordinates[i, 2]], dtype='float64')
					s_j = np.array([rel_coordinates[j, 1], rel_coordinates[j, 2]], dtype='float64')
					# 需要判断是否为空才进行下一步
					if d[cnt]:
						G_m[cnt] = 1 / enent_v * (locate_fun.dis(event_xy, s_i) - locate_fun.dis(event_xy, s_j))
						J[cnt, k * 3] = 1 / enent_v * ((x - s_i[0]) / locate_fun.dis(event_xy, s_i) -
													   (x - s_j[0]) / locate_fun.dis(event_xy, s_j))
						J[cnt, k * 3 + 1] = 1 / enent_v * ((y - s_i[1]) / locate_fun.dis(event_xy, s_i) -
														   (y - s_j[1]) / locate_fun.dis(event_xy, s_j))
						J[cnt, k * 3 + 2] = -1 / enent_v ** 2 * (locate_fun.dis(event_xy, s_i) -
																 locate_fun.dis(event_xy, s_j))
		# 加入正则化项
		delta_m = (np.linalg.inv(J.T @ J + a_loha ** 2 + beta ** 2 * L.T @ L) @
				   (-J.T @ (G_m - d) - a_loha ** 2 * (vk - v0) - beta ** 2 * L.T @ L @ m))
		# 更新m
		m = m + delta_m
		# 计算新的G与Vk
		for k in range(K):
			x = m[k * 3][0]
			y = m[k * 3 + 1][0]
			event_xy = m[k * 3:k * 3 + 2].flatten()
			enent_v = m[k * 3 + 2][0]
			vk_new[k * 3 + 2] = m[k * 3 + 2][0]
			for i in range(S):
				for j in range(i + 1, S):
					cnt = int((i + 1) * ((S - i) + S) / 2 + j - i - 1 - S + k * int(S / 2 * (S - 1)))
					station_i = rel_coordinates[i, 0]
					station_j = rel_coordinates[j, 0]
					# 对应的台站信息
					s_i = np.array([rel_coordinates[i, 1], rel_coordinates[i, 2]], dtype='float64')
					s_j = np.array([rel_coordinates[j, 1], rel_coordinates[j, 2]], dtype='float64')
					# 需要判断是否为空才进行下一步
					if d[cnt]:
						G_now[cnt] = 1 / enent_v * (locate_fun.dis(event_xy, s_i) - locate_fun.dis(event_xy, s_j))
		residual.append(np.linalg.norm(G_now - d))
		delta_m_record.append(np.linalg.norm(delta_m))
		print(
			f"alpha:{a_loha}, beta:{beta},Iteration {circle}: norm(Delta M) = {np.linalg.norm(delta_m)}, Residual = {residual[-1]}")
		if np.linalg.norm(delta_m) > 100000 or np.abs(residual[-1] - residual[-2]) < stop_critical or np.abs(
				delta_m_record[-1] - delta_m_record[-2]) < stop_critical_m:
			G_m_d_record.append(np.linalg.norm(G_now - d))
			v_length.append(np.linalg.norm(vk_new-v0))
			break

plt.figure(figsize=(12, 8))
plt.plot(G_m_d_record, v_length, '-', 'r')
# 假设 G_m_d_record 和 v_length 是长度相等的列表
for i in range(len(G_m_d_record)):
	# 绘制散点
	plt.scatter(G_m_d_record[i], v_length[i], color='r')
	# 为每个点添加文本
	plt.text(G_m_d_record[i], v_length[i], alpha[i])

plt.xlabel('log(G(m) - d)')
plt.ylabel('log(v-v0)')

# 设置对数坐标轴
plt.xscale('log')
plt.yscale('log')
#plt.show()
plt.savefig('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/all_G_m_without_velocity_alpha.jpg',dpi=300)