#
# Recursive Systematic Encoder
#

import numpy as np
import collections
# collections用于双端队列（deque）的操作

class RSC:
    def __init__(self):
        self.reset()

    def reset(self):
        self.registers = collections.deque([0, 0])

    # 处理输入值并更新寄存器状态
    def push(self, value):
        result = value ^ self.registers[-1] # 将当前输入比特与编码器的过去状态进行组合，以生成新的输出比特。这增加了编码的冗余度，使得在接收端更容易检测和纠正错误。
        self.registers.rotate(1) # 将寄存器内容右移一位
        self.registers[0] = result # 将计算结果存储到寄存器的第一位
        return result

    # 处理编码结束时的寄存器状态，确保清空寄存器
    def terminate(self):
        result = self.registers[-1]
        self.registers.rotate(1)
        self.registers[0] = 0
        return result

    # 对输入向量进行编码并返回编码结果和系统输出
    def execute(self, vector):
        result = np.zeros(len(vector) + len(self.registers)) # 创建一个长度为输入向量加上寄存器长度的全零数组，用于存储编码结果

        result[:len(vector):] = [self.push(v) for v in vector] # 对输入向量中的每个值调用 push 方法，并将结果存储到 result 的前半部分
        result[len(vector)::] = [self.terminate() for _ in range(len(self.registers))] # 调用 terminate 方法处理寄存器状态，并将结果存储到 result 的后半部分
        # 尾比特用于将编码器的寄存器状态清空，确保编码器在处理完一个数据块后回到初始状态。这对于连续数据块的编码和解码非常重要，因为它避免了状态传播导致的错误累积

        systematic = np.concatenate((vector, result[len(vector)::])) # 将输入向量和 result 的后半部分（尾比特）连接起来
        return result, systematic
