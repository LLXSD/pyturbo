#
# Turbo Decoder
#

import numpy as np

from .siso_decoder import SISODecoder


class TurboDecoder:
    @staticmethod
    def demultiplex(a, b, extrinsic):
        return list(zip(a, b, extrinsic))

    @staticmethod
    def early_exit(LLR, LLR_ext):
        LLR = [int(s > 0) for s in LLR]
        LLR_ext = [int(s > 0) for s in LLR_ext]
        # 如果当前迭代的 LLR 和上次迭代的 LLR 完全相同，返回 True，表示可以提前退出
        return LLR == LLR_ext

    def __init__(self, interleaver, tail_bits=2, max_iter=16):
        self.interleaver = interleaver
        self.block_size = len(self.interleaver)
        self.tail_bits = tail_bits
        self.max_iter = max_iter

        self.decoders = 2 * [SISODecoder(self.block_size + tail_bits)]

        self.reset()

    def reset(self):
        for d in self.decoders:
            d.reset()

        self.LLR_ext = np.zeros(self.block_size + self.tail_bits, dtype=float)

    def interleave(self, vector):
        interleaved = np.zeros(len(vector), dtype=int)
        interleaved[:self.block_size:] = vector[self.interleaver]

        return interleaved

    def deinterleave(self, vector):
        deinterleaved = np.zeros(len(vector), dtype=int)
        deinterleaved[self.interleaver] = vector[:self.block_size:]

        return deinterleaved

    def iterate(self, vector):
        input_tuples = self.demultiplex(vector[::3], vector[1::3], self.LLR_ext) # 将输入信号向量分解为多个部分，如系统比特和奇偶比特，以便在解码过程中分别处理，并添加外部信息（LLR），外部信息在迭代过程中用于调整解码结果

        LLR_1 = self.decoders[0].execute(input_tuples)
        LLR_1 = LLR_1 - self.LLR_ext - 2 * vector[::3]
        LLR_interleaved = self.interleave(LLR_1)

        input_interleaved = self.interleave(vector[::3])

        input_tuples = self.demultiplex(input_interleaved, vector[2::3], LLR_interleaved)

        LLR_2 = self.decoders[1].execute(input_tuples)
        LLR_2 = LLR_2 - LLR_interleaved - 2 * input_interleaved # 消除之前迭代中已经考虑过的外部信息（extrinsic information）和系统比特的影响，从而为下一次迭代提供更准确的输入

        self.LLR_ext = self.deinterleave(LLR_2)

        return self.early_exit(LLR_1, self.LLR_ext)

    def execute(self, vector):
        for _ in range(self.max_iter):
            if self.iterate(vector):
                break

        return self.LLR_ext
