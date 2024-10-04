import heapq
import random
import time

import numpy as np
from matplotlib import pyplot as plt


class UnionFind(object):
    def __init__(self, n):
        self.parent = list(range(n+1))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.parent[x] = y
class Solution(object):
    def minimumCost(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: int
        """
        result = 0
        edge_Tree=[]
        edge_uesd=0
        sorted_connections=sorted(connections,key=lambda x:x[2])
        union=UnionFind(n)
        for x,y,cost in sorted_connections:
            if union.find(x) != union.find(y):
                union.union(x,y)
                edge_uesd+=1
                result+=cost
                edge_Tree.append([x,y,cost])
        return result if edge_uesd==n-1 else -1

    def generate_graph(self,n, m):
        edges = []
        nodes = list(range(n))
        random.shuffle(nodes)
        for i in range(1, n):
            u = nodes[i - 1]
            v = nodes[i]
            cost = random.randint(1, 100)
            edges.append((u, v, cost))

        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                cost = random.randint(1, 100)
                edges.append((u, v, cost))

        return edges
if __name__ == '__main__':
    input_sizes = []
    execution_times_2 = []
    for n in range(10000, 100000,10000):
        m = 10*n
        s=Solution()
        edges = s.generate_graph(n, m)
        start_time = time.time()
        mst_cost = s.minimumCost(n, edges)
        end_time = time.time()

        # print(f"MST cost: {mst_cost}")
        print(f"Input size:{n} Execution time: {end_time - start_time} seconds")

        input_sizes.append(m)
        execution_times_2.append(end_time - start_time)


    input_sizes = np.array(input_sizes)
    execution_times_2 = np.array(execution_times_2)

    # theory execution time
    execution_times_1 = input_sizes * np.log10(input_sizes)
    normalized_factor = np.sum(execution_times_2) / np.sum(input_sizes * np.log10(input_sizes))
    print(f"normalized_factor:{normalized_factor}")
    print(execution_times_1)
    execution_times_1 = normalized_factor * execution_times_1
    # print(execution_times_1)

    # 绘制理论执行时间
    plt.plot(input_sizes, execution_times_1, marker='o', label='Adj Theory Time')

    # 绘制实际执行时间
    plt.plot(input_sizes, execution_times_2, marker='x', label='Actual Execution Time')

    # 添加标签和标题
    plt.xlabel('Input size (n)')
    plt.ylabel('Execution time (seconds)')
    plt.title('Execution Time vs Input Size for Kruskal Algorithm')

    # 添加网格和图例
    plt.grid(True)
    plt.legend()

    # 显示图表
    plt.show()
