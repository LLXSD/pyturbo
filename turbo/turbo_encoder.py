#
# Turbo Encoder
#

import numpy as np

from .rsc import RSC


class TurboEncoder:
    def __init__(self, interleaver):
        self.interleaver = interleaver
        self.block_size = len(self.interleaver)
        self.encoders = 2 * [RSC()]

    def reset(self):
        for e in self.encoders:
            e.reset()

    def interleave(self, vector):
        interleaved = np.zeros(self.block_size, dtype=int)
        for i in range(self.block_size):
            interleaved[i] = vector[self.interleaver[i]]

        return interleaved

    def execute(self, vector):
        # 考虑了尾比特：
        # 1. 清空寄存器状态：在编码结束时，通过生成尾比特，确保编码器的寄存器状态返回到初始状态（全零状态）。
        # 2. 确保连续性：在处理连续的数据块时，清空寄存器状态可以避免状态传播导致的错误累积。
        output_size = 3 * (len(vector) + len(self.encoders[0].registers))
        output = np.zeros(output_size, dtype=int)
        interleaved = self.interleave(vector)

        output[1::3], output[::3] = self.encoders[0].execute(vector)
        output[2::3], _ = self.encoders[1].execute(interleaved)

        return output
