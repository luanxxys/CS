# Numpy - Numerical Python

NumPy 是 Python 中的一个线性代数库。对每一个数据科学或机器学习 Python 包而言，这都是一个非常重要的库，**SciPy（Scientific Python）、Mat-plotlib（plotting library）、Scikit-learn** 等都在一定程度上依赖 NumPy。

对数组执行数学运算和逻辑运算时，NumPy 是非常有用的。在用 Python 对 n 维数组和矩阵进行运算时，NumPy 提供了大量有用特征。

以下是它提供的一些功能：

- N 维数组，一种快速、高效使用内存的多维数组，它提供矢量化数学运算。

- 你可以不需要使用循环，就对整个数组内的数据行标准数学运算。

- 非常便于传送数据到用低级语言（如 C 或 C++）编写的外部库，也便于外部库以 Numpy 数组形式返回数据。

NumPy 不提供高级数据分析功能，但有了对 NumPy 数组和面向数组的计算的理解，能帮助你更有效地使用像 Pandas 之类的工具。

## function

- ### 从 Python 列表中创建 NumPy 数组

    创建一维数组

        my_list = [1, 2, 3, 4, 5]
        my_numpy_list = np.array(my_list)

    创建 3 行 3 列的二维数组

        second_list = [[1,2,3], [5,4,1], [3,6,7]]
        new_2d_arr = np.array(second_list)

- ### 使用 arange() 内置函数创建 NumPy 数组

    与 Python 的 range() 内置函数相似，我们可以用 arange() 创建一个 NumPy 数组。

        my_list = np.arange(10)
        # OR my_list = np.arange(0,10)

        my_zeros = np.zeros(7)
        my_ones = np.ones(5)
        two_d = np.zeros((3,5))

- ### 使用 linspace() 内置函数创建 NumPy 数组

    linspace() 函数返回的数字都具有指定的间隔。也就是说，如果我们想要 1 到 3 中间隔相等的 15 个点(一维向量)，我们只需使用以下命令：

        lin_arr = np.linspace(1, 3, 15)

    > 与 arange() 函数不同，linspace() 的第三个参数是要创建的数据点数量。

- ### 在 NumPy 中创建一个恒等矩阵

    恒等矩阵是一个二维方矩阵，恒等矩阵的对角线都是 1，其他的都是 0。恒等矩阵一般只有一个参数

        # 6 is the number of columns/rows you want
        my_matrx = np.eye(6)

- ### 用 NumPy 创建一个随机数组成的数组

    我们可以使用 rand()、randn() 或 randint() 函数生成一个随机数组成的数组。

    + 用 random.rand()，我们可以生成一个从 0~1 均匀产生的随机数组成的数组。

            # 由 4 个对象组成的一维数组
            my_rand = np.random.rand(4)

            # 5 行 4 列的二维数组
            my_rand = np.random.rand(5, 4)

    + 使用 randn()，我们可以从以 0 为中心的标准正态分布或高斯分布中产生随机样本。

            # 生成 7 个随机数
            my_randn = np.random.randn(7) my_randn

            np.random.randn(3,5)

    + 使用 randint() 函数生成整数数组。

        randint() 函数最多可以有三个参数：最小值（包含），最大值（不包含）以及数组的大小。

            np.random.randint(20)
            np.random.randint(2, 20)
            # generates 7 random integers including 2 but excluding 20
            np.random.randint(2, 20, 7)

- ### 将一维数组转换成二维数组

        arr = np.random.rand(25)
        arr.reshape(5,5)

    > reshape() 仅可转换成行列数目相等，且行列数相乘后要与元素数量相等。

- ### 定位 NumPy 数组中的最大值和最小值

    使用 max() 和 min() 函数，我们可以得到数组中的最大值或最小值：

        arr_2 = np.random.randint(0, 20, 10)
        arr_2.max()
        arr_2.min()

    使用 argmax() 和 argmin() 函数，我们可以定位数组中最大值和最小值的索引：

        arr_2.argmax()
        arr_2.argmin()

    + ndarray.shape：显示在每个维度里数组的大小
    + ndarray.ndim：显示数组的轴线数量（或维度）
    + ndarray.size：数组中所有元素的总量，相当于数组的 shape 中所有元素的乘积，例如矩阵的元素总量为行与列的乘积
    + ndarray.dtype：显示数组元素的类型
    + numpy.float64，其中「int」和「float」代表数据的种类是整数还是浮点数，「32」和「16」代表这个数组的字节数（存储大小）
    + ndarray.itemsize：数组中每个元素的字节存储大小

- ### 从 NumPy 数组中索引／选择多个元素（组）

    在 NumPy 数组中进行索引与 Python 类似，只需输入想要的索引即可：

        my_array = np.arange(0,11)
        my_array[8]  #This gives us the value of element at index 8

    为了获得数组中的一系列值，我们可以使用切片符「:」，就像在 Python 中一样：

        my_array[2:6]
        my_array[:6]
        my_array[5:]

    类似地，我们也可以通过使用 [ ][ ] 或 [,] 在二维数组中选择元素。

        # 使用 [ ][ ] 从下面的二维数组中抓取出值「60」
        two_d_arr = np.array([[10,20,30], [40,50,60], [70,80,90]])
        two_d_arr[1][2]

        # 使用 [,] 从上面的二维数组中抓取出值「20」
        two_d_arr[0,1]

    也可以用切片符抓取二维数组的子部分

        two_d_arr[:1, :2] # This returns [[10, 20]] two_d_arr[:2, 1:] # This returns ([[20, 30], [50, 60]]) two_d_arr[:2, :2] #This returns ([[10, 20], [40, 50]])

    我们还可以索引一整行或一整列。只需使用索引数字即可抓取任意一行：

        two_d_arr[0] #This grabs row 0 of the array ([10, 20, 30])
        two_d_arr[:2] #This grabs everything before row 2 ([[10, 20, 30], [40, 50, 60]])

    还可以使用 &、|、<、> 和 == 运算符对数组执行条件选择和逻辑选择，从而对比数组中的值和给定值：

        new_arr = np.arange(5,15)

        #This returns TRUE where the elements are greater than 10 [False, False, False, False, False, False,  True,  True,  True, True]
        new_arr > 10

    现在我们可以输出符合上述条件的元素：

        bool_arr = new_arr > 10

        # This returns elements greater than 10 [11, 12, 13, 14]
        new_arr[bool_arr]  

        new_arr[new_arr>10] #A shorter way to do what we have just done

    组合使用条件运算符和逻辑运算符，我们可以得到值大于 6 小于 10 的元素：

        new_arr[(new_arr>6) & (new_arr<10)] # 预期结果为：([7, 8, 9])

- ### 广播机制

    广播机制是一种快速改变 NumPy 数组中的值的方式。

    # 将索引为 0 到 3 的元素的初始值改为 50
    my_array[0:3] = 50 #Result is:  [50, 50, 50, 3, 4,  5,  6,  7,  8,  9, 10]

- ### 对 NumPy 数组执行数学运算

        arr = np.arange(1,11)
        arr * arr #Multiplies each element by itself
        arr - arr
        arr + arr
        arr / arr

    我们还可以对数组执行标量运算，NumPy 通过广播机制使其成为可能： 

        arr + 50 #This adds 50 to every element in that array

    NumPy 还允许在数组上执行通用函数，如平方根函数、指数函数和三角函数等。

        np.sqrt(arr) #Returns the square root of each element
        np.exp(arr) #Returns the exponentials of each element
        np.sin(arr) #Returns the sin of each element
        np.cos(arr) #Returns the cosine of each element
        np.log(arr) #Returns the logarithm of each element
        np.sum(arr) #Returns the sum total of elements in the array
        np.std(arr) #Returns the standard deviation of in the array

    我们还可以在二维数组中抓取行或列的总和：

        mat = np.arange(1,26).reshape(5,5)
        mat.sum() #Returns the sum of all the values in mat
        mat.sum(axis=0) #Returns the sum of all the columns in
        mat mat.sum(axis=1) #Returns the sum of all the rows in mat
