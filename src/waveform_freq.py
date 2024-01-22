# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2024/1/16 15:06
@Author : karsten
@File : waveform_freq.py
@Software: PyCharm
============================
"""


import matplotlib.pyplot as plt
from obspy import read
import obspy
from obspy.signal.util import next_pow_2
import numpy as np
# 读取地震波形文件

font = {
'weight' : 'normal',
'size'   : 15,
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
st = obspy.read('../hangji_quake/2022-07-14T01:04:05.000Z/22394_2022-07-14T01:04:05.000Z.mseed')
st = waveform_processing(st)
trace = st[0]  # 选择第一个轨迹

# 绘制波形图
plt.figure(figsize=(12, 6))
plt.subplot(211)  # 两行一列的第一个
plt.plot(trace.times(), trace.data)
plt.title("Seismic Waveform",font)
plt.xlabel("Time (s)",font)
plt.ylabel("Amplitude",font)

# 计算并绘制频谱图
npts = len(trace.data)  # 数据点数
nfft = next_pow_2(npts)  # 下一个2的幂
p = np.fft.rfft(trace.data, n=nfft)  # 快速傅里叶变换
frequencies = np.fft.rfftfreq(nfft, d=trace.stats.delta)  # 频率轴

plt.subplot(212)  # 两行一列的第二个
plt.plot(frequencies, np.abs(p))  # 绘制
plt.xlim(0, 150)
plt.title("Spectrum",font)
plt.xlabel("Frequency (Hz)",font)
plt.ylabel("Amplitude",font)

plt.tight_layout()
plt.savefig('../output/waveform_freq.png',dpi=300)