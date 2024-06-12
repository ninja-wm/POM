#实现torch tensor的矩阵求和
def sumtensor(tensor):
    sum = 0
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            sum += tensor[i][j]
    return sum

# 实现一个minist分类网络
