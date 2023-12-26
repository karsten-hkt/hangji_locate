# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/15 17:11
@Author : karsten
@File : locate_fun.py
@Software: PyCharm
============================
"""
import numpy as np
from obspy.signal.cross_correlation import correlate

# 计算两个点之间距离的函数（常用）
def dis(p1, p2):
	'''
	计算两个点之间的距离
	'''
	return np.linalg.norm(p1 - p2)


def waveform_processing(st):
	'''
	对全部stream进行预处理
	滤波5hz以上，去除趋势，去除仪器响应，归一化，去均值
	'''
	# 滤波
	st.filter("highpass", freq=5.0)
	# 归一化
	st.normalize()
	# 去均值
	st.detrend("demean")
	# 去除趋势
	st.detrend("linear")
	# taper
	st.taper(0.05)
	return st


# 计算两个波形之间的互相关函数并返回相对时间
def lag_time(waveform1, waveform2, maxlag, sample_rate):
	'''
	计算两个波形之间的互相关以及延迟时间
	'''
	# 计算互相关函数
	corr = correlate(waveform1, waveform2, maxlag)
	# 找到互相关函数的最大值
	max_corr = np.max(corr)
	max_corr_index = np.argmax(corr)
	# 计算实际的时间延迟
	# 如果max_corr_index大于maxlag，实际延迟应为负值
	lag_samples = max_corr_index - maxlag
	# 假设采样率为 sample_rate，单位为 Hz（每秒样本数）
	lag_time = lag_samples / sample_rate  # 转换为秒
	return max_corr, lag_time


def build_matrices(rel_coordinates, stp, E0, v, sigma_t, maxlag, sample_rate):
	N = len(rel_coordinates) * (len(rel_coordinates) - 1) // 2
	d = np.zeros((N, 1))
	G = np.zeros((N, 2))
	corr = np.zeros(N)
	W = np.zeros((N, N))

	for i in range(len(rel_coordinates)):
		for j in range(i + 1, len(rel_coordinates)):
			# 绝对位置
			cnt = int((i + 1) * ((len(rel_coordinates) - i) + len(rel_coordinates)) / 2 + j - i - 1 - len(rel_coordinates))
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
				maxcor, lagtime = lag_time(w_i, w_j, maxlag, sample_rate)

				if maxcor > 0.3:
					corr[cnt] = maxcor
					W[cnt, cnt] = maxcor / sigma_t
					d[cnt] = lagtime - (dis(E0, s_i) - dis(E0, s_j)) / v
					G[cnt, 0] = ((E0[0] - s_i[0]) / dis(E0, s_i) - (E0[0] - s_j[0]) / dis(E0, s_j)) / v
					G[cnt, 1] = ((E0[1] - s_i[1]) / dis(E0, s_i) - (E0[1] - s_j[1]) / dis(E0, s_j)) / v

	return G, d, W, corr

def build_matrices_without_velocity(rel_coordinates, stp, E0, V0, sigma_t, maxlag, sample_rate):
	N = len(rel_coordinates) * (len(rel_coordinates) - 1) // 2
	d = np.zeros((N, 1))
	G = np.zeros((N, 3))
	corr = np.zeros(N)
	W = np.zeros((N, N))

	for i in range(len(rel_coordinates)):
		for j in range(i + 1, len(rel_coordinates)):
			# 绝对位置
			cnt = int((i + 1) * ((len(rel_coordinates) - i) + len(rel_coordinates)) / 2 + j - i - 1 - len(rel_coordinates))
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
				maxcor, lagtime = lag_time(w_i, w_j, maxlag, sample_rate)

				if maxcor > 0.3:
					corr[cnt] = maxcor
					W[cnt, cnt] = maxcor / sigma_t
					d[cnt] = lagtime - (dis(E0, s_i) - dis(E0, s_j)) / V0
					G[cnt, 0] = ((E0[0] - s_i[0]) / dis(E0, s_i) - (E0[0] - s_j[0]) / dis(E0, s_j)) / V0
					G[cnt, 1] = ((E0[1] - s_i[1]) / dis(E0, s_i) - (E0[1] - s_j[1]) / dis(E0, s_j)) / V0
					G[cnt, 2] = -1/V0**2 * (dis(E0, s_i) - dis(E0, s_j))

	return G, d, W, corr

# 网格搜索中计算误差的函数
def grid_search_misfit(S_corr, grid_point, velocity):
	'''
	计算理论到时
	:param S_corr: 各个台站对位置以及观测到的到时差
	:param grid_point: 对应的网格点
	:param velocity: 介质速度
	:return:
	'''
	obs_arrival = S_corr[:, 6]
	errors = []

	for point in grid_point:
		# 计算理论到达时间
		theoretical_arrival = np.array([dis(point, S_corr[i, 2:4]) - dis(point, S_corr[i, 4:6])
										if not np.isnan(obs_arrival[i]) else np.nan
										for i in range(len(obs_arrival))]) / velocity

		# 计算误差，忽略 NaN 值
		valid_indices = ~np.isnan(obs_arrival)
		error = np.linalg.norm(theoretical_arrival[valid_indices] - obs_arrival[valid_indices]) ** 2 / len(
			theoretical_arrival[valid_indices])
		errors.append(error)
	return errors

def theritcal_time(rel_coordinates, grid_points, velocity):
	num_stations = len(rel_coordinates)
	num_pairs = int(num_stations * (num_stations - 1) / 2)
	station_pair = np.zeros((num_pairs, 6), dtype=object)

	cnt = 0
	for i in range(num_stations):
		for j in range(i + 1, num_stations):
			station_pair[cnt, :2] = rel_coordinates[i, 0], rel_coordinates[j, 0]
			# 存储台站坐标
			coords_i = rel_coordinates[i, 1:].astype(float)
			coords_j = rel_coordinates[j, 1:].astype(float)
			station_pair[cnt, 2:] = np.concatenate((coords_i, coords_j))
			cnt += 1
	theoretical_arrival = {}
	for point in grid_points:
		point = tuple(point)  # 转换为元组
		theoretical_arrival[point] =np.array([dis(point, station_pair[i, 2:4]) - dis(point, station_pair[i, 4:6])
										for i in range(len(station_pair))]) / velocity
	return theoretical_arrival

def lag_time_error(rel_coordinates, stp, data_error, maxlag, sample_rate):
	'''
	计算两个波形之间的互相关以及延迟时间,并给互相关lagtime一定的误差
	:param rel_coordinates:
	:param stp:
	:param data_error:
	:param maxlag:
	:param sample_rate:
	:return:
	lag_times_with_error：maxcor和lagtime
	'''
	N = len(rel_coordinates) * (len(rel_coordinates) - 1) // 2
	lag_times_with_error = np.zeros((N, 2))
	for i in range(len(rel_coordinates)):
		for j in range(i + 1, len(rel_coordinates)):
			# 绝对位置
			cnt = int((i + 1) * ((len(rel_coordinates) - i) + len(rel_coordinates)) / 2 + j - i - 1 - len(rel_coordinates))
			station_i = rel_coordinates[i, 0]
			station_j = rel_coordinates[j, 0]
			# 对应的波形信息
			sw_i = stp.select(station=station_i)
			sw_j = stp.select(station=station_j)
			# 需要判断是否为空才进行下一步
			if len(sw_i) > 0 and len(sw_j) > 0:
				w_i = sw_i[0]
				w_j = sw_j[0]
				maxcor, lag_time_with_error = lag_time(w_i, w_j, maxlag, sample_rate)
				if maxcor > 0.3:
					error = np.random.normal(0,data_error)
					lag_times_with_error[cnt,0] = maxcor
					# 给其添加一个误差
					lag_times_with_error[cnt,1] = lag_time_with_error + error
	return lag_times_with_error

def lag_time_with_station(rel_coordinates, stp, maxlag, sample_rate):
	'''
	读取数据并返回lagtime
	:param rel_coordinates:
	:param stp:
	:param data_error:
	:param maxlag:
	:param sample_rate:
	:return:
	lag_times_with_error：maxcor和lagtime
	'''
	N = len(rel_coordinates) * (len(rel_coordinates) - 1) // 2
	lag_times = np.zeros((N, 2))
	for i in range(len(rel_coordinates)):
		for j in range(i + 1, len(rel_coordinates)):
			# 绝对位置
			cnt = int((i + 1) * ((len(rel_coordinates) - i) + len(rel_coordinates)) / 2 + j - i - 1 - len(rel_coordinates))
			station_i = rel_coordinates[i, 0]
			station_j = rel_coordinates[j, 0]
			# 对应的波形信息
			sw_i = stp.select(station=station_i)
			sw_j = stp.select(station=station_j)
			# 需要判断是否为空才进行下一步
			if len(sw_i) > 0 and len(sw_j) > 0:
				w_i = sw_i[0]
				w_j = sw_j[0]
				maxcor, lag_time_with_error = lag_time(w_i, w_j, maxlag, sample_rate)
				if maxcor > 0.3:
					lag_times[cnt,0] = maxcor
					# 给其添加一个误差
					lag_times[cnt,1] = lag_time_with_error
				else:
					lag_times[cnt,0] = np.nan
					lag_times[cnt,1] = np.nan
			else:
				lag_times[cnt,0] = np.nan
				lag_times[cnt,1] = np.nan
	return lag_times

def build_matrices_with_lag_time_0(rel_coordinates, lag_time, E0, v, sigma_t):
	N = len(rel_coordinates) * (len(rel_coordinates) - 1) // 2
	d = np.zeros((N, 1))
	G = np.zeros((N, 2))
	corr = np.zeros(N)
	W = np.zeros((N, N))
	for i in range(len(rel_coordinates)):
		for j in range(i + 1, len(rel_coordinates)):
			# 绝对位置
			cnt = int((i + 1) * ((len(rel_coordinates) - i) + len(rel_coordinates)) / 2 + j - i - 1 - len(rel_coordinates))
			# 对应的台站信息
			s_i = np.array([rel_coordinates[i, 1], rel_coordinates[i, 2]], dtype='float64')
			s_j = np.array([rel_coordinates[j, 1], rel_coordinates[j, 2]], dtype='float64')
			# 对应的波形信息
			corr[cnt] = lag_time[cnt,0]
			W[cnt, cnt] = lag_time[cnt,0] / sigma_t
			d[cnt] = lag_time[cnt,1] - (dis(E0, s_i) - dis(E0, s_j)) / v
			G[cnt, 0] = ((E0[0] - s_i[0]) / dis(E0, s_i) - (E0[0] - s_j[0]) / dis(E0, s_j)) / v
			G[cnt, 1] = ((E0[1] - s_i[1]) / dis(E0, s_i) - (E0[1] - s_j[1]) / dis(E0, s_j)) / v

	return G, d, W, corr

def build_matrices_with_lag_time_nan(rel_coordinates, lag_time, E0, v, sigma_t):
	N = len(rel_coordinates) * (len(rel_coordinates) - 1) // 2
	d = np.zeros((N, 1))
	G = np.zeros((N, 2))
	corr = np.zeros(N)
	W = np.zeros((N, N))
	for i in range(len(rel_coordinates)):
		for j in range(i + 1, len(rel_coordinates)):
			# 绝对位置
			cnt = int((i + 1) * ((len(rel_coordinates) - i) + len(rel_coordinates)) / 2 + j - i - 1 - len(rel_coordinates))
			if ~np.isnan(lag_time[cnt, 0]):
				# 对应的台站信息
				s_i = np.array([rel_coordinates[i, 1], rel_coordinates[i, 2]], dtype='float64')
				s_j = np.array([rel_coordinates[j, 1], rel_coordinates[j, 2]], dtype='float64')
				# 对应的波形信息
				corr[cnt] = lag_time[cnt,0]
				W[cnt, cnt] = lag_time[cnt,0] / sigma_t
				d[cnt] = lag_time[cnt,1] - (dis(E0, s_i) - dis(E0, s_j)) / v
				G[cnt, 0] = ((E0[0] - s_i[0]) / dis(E0, s_i) - (E0[0] - s_j[0]) / dis(E0, s_j)) / v
				G[cnt, 1] = ((E0[1] - s_i[1]) / dis(E0, s_i) - (E0[1] - s_j[1]) / dis(E0, s_j)) / v
	return G, d, W, corr

if __name__ == '__main__':
	# 使用此函数的示例
	# G, d, W, corr = build_matrices(rel_coordinates, stp, E0, v, sigma_t, maxlag, sample_rate)
	print('This is the function to locate Earthquake')

