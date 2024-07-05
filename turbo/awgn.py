#
# AWGN Channel
#

#!/usr/bin/env python3

import numpy as np 


class AWGN:
    @staticmethod
    def convert_to_symbols(vector): # 将输入向量 vector 中的每个元素转换为符号
        return np.add(np.multiply(vector, 2), -1) # 将向量中的每个元素乘以2减1

    def __init__(self, noise_dB):
        self.scale = 1.0 / (10.0**(noise_dB / 20.0)) # 设置噪声的缩放比例

    def execute(self, vector):
        noise = np.random.normal(0, 1, len(vector)) # 生成一个长度与输入向量 vector 相同的正态分布（均值为0，标准差为1）的随机噪声向量
        noise = np.multiply(noise, self.scale) # 缩放，以模拟特定分贝级别的噪声
        return np.add(vector, noise) # 将输入向量 vector 和缩放后的噪声向量 noise 相加，得到模拟后的接收信号向量
