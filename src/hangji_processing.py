# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/15 16:56
@Author : karsten
@File : hangji_processing.py
@Software: PyCharm
============================
"""
import os
import obspy
import pandas as pd
from obspy import UTCDateTime

# 基础参数设置
target_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_quake'
path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_using'

stations = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/station_TDS.txt',
					   header=None, names=['station', 'lon', 'lat'], sep=',')
# 创建一个字典，以裁剪后的台站名称为键，经纬度为值
station_info = {row['station'].split('_')[0]: {'lat': row['lat'], 'lon': row['lon']}
				for index, row in stations.iterrows()}
# 需要切割数据的时间
hangji = pd.read_csv('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/data/hangji.csv')
hangji_time = hangji['time']
pre_event_time = 5  # 发震前10秒
post_event_time = 45  # 发震后45秒

# 创建母文件夹
os.makedirs(target_path, exist_ok=True)

# 开始遍历文件
for root, dirs, files in os.walk(path):
	for dir in dirs:
		full_dir_path = os.path.join(root, dir)
		stn = obspy.Stream()  # 初始化一个空的Stream对象

		for file in os.listdir(full_dir_path):
			if file.endswith('000'):
				file_path = os.path.join(full_dir_path, file)
				stn += obspy.read(file_path)

		# 将读取的数据合并
		stn_merge = stn.merge(method=1)
		st_z = stn_merge.select(channel="*Z")
		if len(st_z) > 0:
			# 更新头文件信息
			for tr in st_z:
				station_code = tr.stats.station  # 假设这已经是短格式名称
				if station_code in station_info:
					tr.stats.latitude = station_info[station_code]['lat']
					tr.stats.longitude = station_info[station_code]['lon']
				else:
					print(f"Warning: Station {station_code} not found in station info.")

			# 保存merge数据
			output_filename = os.path.join(full_dir_path, f'{st_z[0].stats.station}_merge.mseed')
			st_z.write(output_filename, format="MSEED")

			# 根据hangji发震时间切割数据
			for quake_time in hangji_time:
				stz_copy = st_z.copy()
				earthquake_time = UTCDateTime(quake_time)
				start_time = earthquake_time - pre_event_time
				end_time = earthquake_time + post_event_time

				# 切割 Stream
				stz_copy.trim(starttime=start_time, endtime=end_time)

				# 检查切割后的 Stream 是否为空
				if len(stz_copy) > 0:
					# 创建一个目录来存储所有切割后的数据
					event_path = os.path.join(target_path, quake_time)
					os.makedirs(event_path, exist_ok=True)

					# 保存文件
					quake_output_filename = os.path.join(event_path, f'{st_z[0].stats.station}_{quake_time}.mseed')
					stz_copy.write(quake_output_filename, format='MSEED')
				else:
					print(f"No data available for {quake_time} in station {st_z[0].stats.station}.")