# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================
@Time : 2023/12/16 14:34
@Author : karsten
@File : test.py
@Software: PyCharm
============================
"""
# 绘制最终定位结果
# 开始画图
plt.figure(figsize=(12, 8))
# 绘制各个台站
for station in rel_coordinates:
	plt.plot(station[1].astype(float), station[2].astype(float), 'o', markersize=5, color='yellow')
	plt.text(station[1].astype(float), station[2].astype(float), station[0])
# 绘制地震
for i in range(K):
	plt.plot(m[i*3], m[i*3+1], 'o', markersize=5, color = 'b')
	plt.text(m[i*3], m[i*3+1], event_name[i], fontsize=8)
# 设置x，y坐标值范围
plt.xlim(-60,60)
plt.ylim(-20,100)
plt.xlabel('Relative X coordinate (meters)')
plt.ylabel('Relative Y coordinate (meters)')
plt.title('Relative event and station positions of stations(22917)')
plt.savefig('/Users/karsten_hkt/PycharmProjects/seismo_live_local/obspy_learning/output/all_G_m_without_velocity.jpg',dpi=300)