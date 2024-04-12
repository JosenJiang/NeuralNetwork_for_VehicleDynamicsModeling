# encoding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch


if torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
# 创建一个示例 DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df = pd.DataFrame(data)

# 使用 to_numpy() 方法将 DataFrame 转换为 NumPy 数组
numpy_array = df.to_numpy()

print(numpy_array)

x_list = np.arange(3)

plt.figure()
plt.plot(x_list, numpy_array[:, 0].T)
plt.show()
