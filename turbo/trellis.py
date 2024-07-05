#
# Trellis Object
# Trellis 类通过定义状态转移矩阵、前驱状态和后继状态，描述了编码器或解码器的状态转移图.
# 在SISODecoder中，利用这些信息计算路径度量和分支度量，从而实现Turbo解码器的前向后向计算和LLR的估计.
#

import itertools
import numpy as np


class Trellis:
    # 定义一个静态方法，不需要实例化对象即可调用
    @staticmethod
    def butterfly(path_metrics, branch_metrics):
        result = [path + branch for path, branch in zip(path_metrics, branch_metrics)]
        return np.max(result)

    def __init__(self):
        self.transition_matrix = np.array(
                                  [[(-1, -1), None, (1, 1), None],
                                  [(1, -1), None, (-1, 1), None],
                                  [None, (-1, -1), None, (1, 1)],
                                  [None, (1, -1), None, (-1, 1)]]) # [4,4] 表示从一个状态到另一个状态的转移符号

        self.past_states = [(0, 1), (2, 3), (0, 1), (2, 3)] # 状态0的前驱状态是(0, 1)，即状态0和1可以转移到状态0
        self.future_states = [(0, 2), (0, 2), (1, 3), (1, 3)] # 状态0的后继状态是(0, 2)，即状态0可以转移到状态0和2

        all_transitions = list(itertools.product([0, 1, 2, 3], repeat=2)) # 生成所有可能的状态对
        self.possible_transitions = [t for t in all_transitions if self.transition_matrix[t] is not None] # 过滤出所有有效的状态转移对

    def transition_to_symbols(self, state, next_state):
        # 将状态转移对 (state, next_state) 映射到对应的符号对 (input, output)
        return self.transition_matrix[state, next_state]
