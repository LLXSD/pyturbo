#
# Soft Input Soft Output (SISO) Decoder
# SISO 解码器通常使用 Viterbi 算法来实现。
# Viterbi 算法是一种动态规划算法，用于在卷积编码解码过程中，根据接收信号的硬或软信息，寻找最有可能的发送序列。
#

import math
import itertools
import numpy as np

from .trellis import Trellis


class SISODecoder:
    @staticmethod
    def init_branch_metrics(m, n, depth):
        return np.array(depth * [m * [n * [0.0]]]) # 分支度量矩阵 [depth, m, n]

    @staticmethod
    def init_path_metric(m, depth):
        matrix = np.array(depth * [m * [-math.inf]]) # 路径度量矩阵 [depth, m] 记录了每个时间步每个状态的最优路径度量
        matrix[:, 0] = 0
        return matrix

    @staticmethod
    def demultiplex(vector):
        result = list(zip(vector[0::3], vector[1::3], vector[2::3]))
        return [(x, y, 0.0) for x, y, _ in result] # 只返回第一个的原因是解码器通常只需要处理前一个编码器生成的校验比特流，而不需要同时处理两个校验比特流

    def __init__(self, block_size):
        self.trellis = Trellis()
        self.block_size = block_size

        self.reset()

    def reset(self):
        self.branch_metrics = self.init_branch_metrics(4, 4, self.block_size) # 状态数通常是4，较小的状态数可以在保证较高编码效率的同时，限制状态转移图的复杂性

        self.forward_metrics = self.init_path_metric(4, self.block_size + 1)
        self.backward_metrics = self.init_path_metric(4, self.block_size + 1)

        self.LLR = np.zeros(self.block_size)

    def compute_branch(self, tuples):
        for k in range(0, self.block_size):
            for transition in self.trellis.possible_transitions:
                m, n = transition # Turbo 编码中每个卷积编码器的状态之间的转移。在解码器中，这些转移对应于Trellis码本中的状态转移路径
                i, o = self.trellis.transition_to_symbols(m, n) # 从状态m到状态n的转移时，对应的输入i和输出符号o，可能为+1或-1

                self.branch_metrics[k, m, n] = i * tuples[k][0] + o * tuples[k][1] + i * tuples[k][2]

    def compute_forward(self, k, state):
        past_states = self.trellis.past_states[state] # 返回在当前状态state下可能的前一状态集合

        forward_metrics = self.forward_metrics[k - 1, past_states] # 从起始状态到当前时间步 k-1 past_states状态的前向度量
        branch_metrics = self.branch_metrics[k - 1, past_states, state] # 当前时间步 k-1 到当前状态 state 的分支度量

        self.forward_metrics[k, state] = self.trellis.butterfly(forward_metrics, branch_metrics)

    def compute_backward(self, k, state):
        future_states = self.trellis.future_states[state] # 在当前状态 state 下可能的后一状态集合

        r = self.block_size - k # r 是当前时间步 k 到最后时间步的剩余步数

        backward_metrics = self.backward_metrics[k - 1, future_states] # 从时间步 k-1 到最后时间步各状态的后向度量
        branch_metrics = self.branch_metrics[r, state, future_states] # 时间步 r 到当前状态 state 和后一时间步可能状态的分支度量

        self.backward_metrics[k, state] = self.trellis.butterfly(backward_metrics, branch_metrics)

    def compute_LLR(self, k):
        r = self.block_size - k - 1 # 计算从当前时间步到结束时间步的剩余步数

        positive = [] # 1
        negative = [] # 0

        for transition in self.trellis.possible_transitions: # 遍历所有可能的状态转移
            m, n = transition
            i, _ = self.trellis.transition_to_symbols(m, n)

            forward_metric = self.forward_metrics[k, m]
            branch_metric = self.branch_metrics[k, m, n]
            backward_metric = self.backward_metrics[r, n]

            if i < 0:
                negative.append(forward_metric + branch_metric + backward_metric)
            else:
                positive.append(forward_metric + branch_metric + backward_metric)

        self.LLR[k] = np.max(positive) - np.max(negative) # 使用最大度量差来近似对数似然比

    def execute(self, tuples):
        self.compute_branch(tuples) # tuples 是一个包含软信息的元组列表，每个元组包含 (系统比特, 第一个校验比特, 第二个校验比特)，得到所有状态转移路径上的分支度量

        for k in range(1, self.block_size + 1):
            for state in range(0, 4): # 假设 4 个状态
                self.compute_forward(k, state)
                self.compute_backward(k, state)

        for k in range(0, self.block_size):
            self.compute_LLR(k)

        return self.LLR
