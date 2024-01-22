import numpy as np

def bootstrap_sample(data, sample_ratio=0.8):
    """
    对数据进行bootstrap采样。
    选取sample_ratio比例的数据，其余数据赋值为0。

    :param data: 原始数据，一维数组。
    :param sample_ratio: 采样的比例，默认为0.8。
    :return: 经过bootstrap采样处理的数据。
    """
    n = len(data)
    sample_size = int(n * sample_ratio)
    sampled_indices = np.random.choice(n, size=sample_size, replace=True)
    new_data = np.zeros(n)
    np.put(new_data, sampled_indices, data[sampled_indices])
    return new_data

# 示例数据
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 进行bootstrap采样
sampled_data = bootstrap_sample(data)

print("原始数据:", data)
print("采样后的数据:", sampled_data)