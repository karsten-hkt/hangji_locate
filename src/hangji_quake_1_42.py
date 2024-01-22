# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2024/1/19 11:59
@Author : karsten
@File : hangji_quake_1_42.py
@Software: PyCharm
============================
"""

import matplotlib.pyplot as plt
from obspy import read
import obspy
from obspy.signal.util import next_pow_2
import numpy as np
import os
# 读取地震波形文件

font = {
'weight' : 'normal',
'size'   : 3,
        }

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

event_name = '01:44:55.000Z'
event_path = '/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/hangji_quake/2022-07-14T01:44:55.000Z'
stn = obspy.Stream()  # 初始化一个空的Stream对象
for root, dirs, files in os.walk(event_path):
	for file in files:
		file_path = os.path.join(event_path, file)
		stn += obspy.read(file_path)
stp = stn.copy()
fig = plt.figure(figsize=(12, 6))
for i in range(len(stp)):
	plt.subplot(4,6,i+1)  # 两行一列的第一个
	trace = waveform_processing(stp[i])
	# 绘制波形图
	plt.plot(trace.times(), trace.data)
	plt.title("Seismic Waveform", font)
	plt.xlabel("Time (s)", font)
	plt.ylabel("Amplitude", font)
plt.show()