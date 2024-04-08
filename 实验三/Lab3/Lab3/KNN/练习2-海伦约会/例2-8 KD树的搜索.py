# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:35:54 2019

@author: Harry
"""

from math import sqrt
from collections import namedtuple
# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple", "nearest_point  nearest_dist  nodes_visited")

# KD树每个结点中主要包含的数据结构如下
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  	# k维向量节点(k维空间中的一个样本点)
        self.split = split      	# 整数（进行分割维度的序号）
        self.left = left        	# 该结点分割超平面左子空间构成的kd-tree
        self.right = right      	# 该结点分割超平面右子空间构成的kd-tree
class KdTree(object):
    def __init__(self, data):
        k = len(data[0])  		# 数据维度
	   # 按第split维划分数据集exset创建KdNode
        def CreateNode(split, data_set):
            if not data_set:    	# 数据集为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2   	# //为Python中的整数除法
            median = data_set[split_pos]     	# 中值分割点
            split_next = (split + 1) % k     	# cycle coordinates
            # 递归的创建kd树
            return KdNode(median, split,
					   # 创建左子树
                          CreateNode(split_next, data_set[:split_pos]),
					   # 创建右子树
                          CreateNode(split_next, data_set[split_pos + 1:]))
        self.root = CreateNode(0, data) # 从第0维分量开始构建kd树,返回根节点

def find_nearest(tree, point):
    k = len(point) # 数据维度
    def travel(kd_node, target, max_dist):
        if kd_node is None:     
            return result([0] * k, float("inf"), 0) # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split        # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:           # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node  = kd_node.left     # 下一个访问节点为左子树根节点
            further_node = kd_node.right    # 同时记录下右子树
        else:                               # 目标离右子树更近
            nearer_node  = kd_node.right    # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point       # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist           # 更新最近距离

        nodes_visited += temp1.nodes_visited  

        if dist < max_dist:     
            max_dist = dist    # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])    # 第s维上目标点与分割超平面的距离
        if  max_dist < temp_dist:                # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited) # 不相交则可以直接返回，不用继续判断

        #----------------------------------------------------------------------  
        # 计算目标点与分割点的欧氏距离  
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))     

        if temp_dist < dist:         # 如果“更近”
            nearest = pivot          # 更新最近点
            dist = temp_dist         # 更新最近距离
            max_dist = dist          # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist) 

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:        # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point    # 更新最近点
            dist = temp2.nearest_dist        # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归
if __name__ == "__main__":
    		data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    		kd = KdTree(data)
    		ret = find_nearest(kd, [3,4.5])
    		print(ret)