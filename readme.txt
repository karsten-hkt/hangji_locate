编写格式使用python3.9
需要安装包为：
pandas,obspy,numpy,matplotlib,pyproj
####################################
主要进行夯机地震位置的定位
定位方法根据台站之间的到时差进行
####################################
具体代码放在src文件夹下
————————————
hangji_processing.py：是对所有台站数据做一个merge操作并根据粗略的夯机发震时间截取出对应的夯机波片段
该文件结果放在hangji_quake中，依照每个地震的发震时间放置文件
————————————
station_draw.py：绘制夯机的位置以及某个地震到所有台站的射线, 该结果成图放在output文件夹下方
————————————
hangji_locate.py：初始对单一夯机地震事件的定位，属于一个测试版本，并未全部地震的数据进行测试
————————————
locate_fun.py：具体放置定位函数的py文件
————————————
station_fun.py：放置计算station的py文件
————————————
hangji_locate_all.py：对所有夯机地震事件进行定位，遍历文件夹的操作，最终结果保存在output之中，以csv格式保存
————————————
hangji_event_draw.py: 对所有夯机地震事件进行绘图，结果保存在output中
————————————
hangji_locate_data_error_analysis.py：使用蒙特卡洛误差传播方式分析在互相关lagtime求的时候存在误差给解带来的影响。
————————————
hangji_locate_velocity_error_analysis.py：使用蒙特卡洛误差传播方式分析在速度与理论不符合的时候误差给解带来的影响。
————————————
hangji_model_resolution_matrix.py: 分析分辨率矩阵的，看看哪个地方反演的更好
————————————
hangji_locate_graph_search.py：初始对单一夯机地震事件的定位，属于一个测试版本,对夯机地震事件使用网格搜索方法
————————————
hangji_locate_graph_search_all.py：对所有夯机地震事件进行网格搜索定位，遍历文件夹的操作，最终结果保存在output之中，以csv格式保存
————————————
hangji_event_draw_graph_search.py: 对使用网格搜索的所有夯机地震事件进行绘图，结果保存在output中
————————————
hagnji_velocity_analysis.py：对所有夯机地震事件进行速度分析，由于不清楚夯机地震时间是一种什么样子的波，只清楚它是一个高频信号，具体的传播速度并不了解，因此需要对速度进行分析
这是一个错误代码，因为将时间项看做了一个矢量去分析。
————————————
hangji_locate_without_velocity.py：对其中一个夯机地震事件进行定位，当速度也是未知数
————————————
hangji_locate_without_velocity_all.py：对所有夯机地震事件进行定位，当速度也是未知数
————————————
hangji_locate_all_variance.py:对定位的结果分析，就是在原先得到的互相关结果中加入误差，使用蒙特卡洛误差传播的方式去给定最终的模型结果方差。
————————————
hangji_locate_all_variance_draw.py:绘制定位的结果以及带上误差分布，结果保存在output中
————————————
hangji_locate_Kestoration_test.py：使用Kestoration test方法对反演结果进行探究，也就是用实际计算得到的模型结果去生成理论到各个台站之间的互相关到时，
然后重新定位结果并与实际的结果进行对比，看看是否有差别
————————————
hangji_locate_Tikhonov_Regularization_build_matrix.py:构建矩阵，用于Tikhonov Regularization方法，由于其他的参数需要重复更新
所以只用构建d矩阵
————————————
hangji_locate_Tikhonov_Regularization.py:使用Tikhonov Regularization方法对反演结果进行探究，
在反演的过程中加入正则化项，对速度约束以及对各个地震位置进行约束，使得反演结果更加平滑,初始版本，还未对正则化的alpha以及beita取值进行分析
————————————
hangji_locate_Tikhonov_Regularization_alpha.py:使用Tikhonov Regularization方法对反演结果进行探究，
在反演的过程中加入正则化项，对速度约束以及对各个地震位置进行约束，使得反演结果更加平滑
对alpha的取值进行探究
————————————
hangji_locate_Tikhonov_Regularization_beta.py:使用Tikhonov Regularization方法对反演结果进行探究，
在反演的过程中加入正则化项，对速度约束以及对各个地震位置进行约束，使得反演结果更加平滑
对beta的取值进行探究
————————————
hangji_locate_Tikhonov_Regularization_variance.py:使用Tikhonov Regularization方法对反演结果进行探究，
同时重复反演并给定方差，使用蒙特卡洛误差传播的方式去给定最终的模型结果方差。结果保存在output中
————————————
hangji_locate_Tikhonov_Regularization_variance_draw.py：绘制正则化最终的结果

#############
.ipynb文件为jupyter notebook文件，用于测试代码