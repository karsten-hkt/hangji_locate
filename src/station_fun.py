# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/15 17:13
@Author : karsten
@File : station_fun.py
@Software: PyCharm
============================
"""
import numpy as np
from pyproj import CRS, Transformer
import obspy
import locate_fun

def rel_coordinates(station_info, ref_station):
	# 设置投影 - WGS84 经纬度到 UTM
	crs_latlon = CRS.from_epsg(4326)  # WGS84
	crs_utm = CRS.from_epsg(32648)  # UTM Zone 48, WGS84
	transformer = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)

	# 转换到UTM并找出参考点

	ref_x, ref_y = transformer.transform(station_info[ref_station]['lon'], station_info[ref_station]['lat'])

	# 计算相对坐标
	rel_coordinates = []
	for station, coords in station_info.items():
		x, y = transformer.transform(coords['lon'], coords['lat'])
		rel_coordinates.append([station, x - ref_x, y - ref_y])
	rel_coordinates = np.array(rel_coordinates)
	return rel_coordinates

# 2. 生成脉冲响应和计算互相关
def generate_pulse_response(earthquake_time, stations_theoretical_time, sampling_rate, duration):
	"""
	生成脉冲响应并添加噪声。

    :param earthquake_time: 地震发生的UTC时间
    :param stations: 台站列表
    :param sampling_rate: 采样率
    :param noise_std: 噪声标准差
    :return: 包含脉冲响应的Stream对象
	"""
	stream = obspy.Stream()

	for station in stations_theoretical_time:

		theoretical_arrival_time = station[1]# 对应的到时

		# 创建一个空的Trace对象
		npts = int(sampling_rate * duration)
		trace = obspy.Trace(data=np.zeros(npts))

		# 设置基本属性
		trace.stats.station = station[0]
		trace.stats.sampling_rate = sampling_rate
		trace.stats.starttime = earthquake_time

		# 在理论到达时间生成脉冲
		arrival_sample = int((theoretical_arrival_time - earthquake_time) * sampling_rate)
		trace.data[arrival_sample] = 1

		stream.append(trace)

	return stream

if __name__ == '__main__':
	print('This is the function to make change with stations')


