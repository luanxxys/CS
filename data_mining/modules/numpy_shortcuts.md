# NumPy 快捷操作

- ## 向量/矩阵/变换/计算

    元素类型要相同

    ```python
    import numpy as np
    # 创建向量,矩阵
    vector = np.array([1, 3])
    matrix = np.array([1, 2],[4, 5])
    # 查看结构
    vector.shape
    # 查看数据类型 data type
    vector.dtype(.name)
    # 类型转换
    vector = vector.astype(float)
    vector.min()
    # 按行求和
    matrix.sum(axis=1)
    # 按列求和
    matrix.sum(axis=0)
    # 构造矩阵
    np.arrange(15).reshape(3, 5)
    # (起始值,终止值,步长)
    np.arrange(10,30,5)
    # 初始化矩阵
    np.zeros((2,3,4), dtype=np.int32)
    # 随机数,[-1,+1]
    np.random.random((2,3))
    # (起始值,终止值,生成数数量)
    from numpy import pi
    np.linspace(0, 2*pi, 100)
    # 矩阵运算函数
    ...
    # 向下取整 floor
    matrix = np.floor(10*np.random.random((2,3)))
    # 矩阵转成向量
    vector = martix.ravel()
    # 再变回来
    vector.shape = (2, 3)
    # 矩阵拼接 行:v, 列:h
    np.vstack((a, b))
    # 矩阵分割 行:v, 列:h
    # 等分三份
    np.hsplit(a, 3)
    # 指定位置切割
    np.hsplit(a, (3, 5))
    # 返回列最大值位置
    ind = matrix.argmax(axis=0)
    # 矩阵变换行列
    new_matrix = np.tile(matrix, (2,3))
    # 按行排序
    a.sort(a, axis=1)
    # 索引值从小到大排序
    j = np.argsort(a)
    j
    a[j]
    ```

- ## 复制

    ```python
    # 复制后两者完全等价(地址,值空间)
    a = b
    # 浅复制 - ID 不同,但值共享
    c = a.view()
    # 深复制 - 互不关联
    c = a.copy()
    ```
