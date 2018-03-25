# Python module

- ## numpy

    矩阵操作

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

    + ### 复制

        ```python
        # 复制后两者完全等价(地址,值空间)
        a = b
        # 浅复制 - ID 不同,但值共享
        c = a.view()
        # 深复制 - 互不关联
        c = a.copy()
        ```

- ## pandas

    在 numpy 基础上封装

    DataFrame 为核心,pandas 将文件读成此种格式,以便进行后续的操作

    ```python
    import pandas as pd
    # 读取以 , 分割的文件(包括 csv 文件)
    f = pd.read_csv("file.csv")
    f.head()/.tail(3)
    # 打印 列名字
    print(f.columns)
    # 取第一二行数据(各个属性都取出来)
    f.loc[0,1]
    # 取 3-5 样本数据
    f.loc[3:5]
    # 取某列
    f["comumns_name"]
    ```

    NaN 代表缺失值

    + 常用函数

        ```python
        # 统计相互关系
        survival = tianic_survival.pivot_table(index="Pclass",values="Survived",aggfunc=np.mean)
        # 统计多个相互关系
        survival = tianic_survival.pivot_table(index="Pclass",values=["Survived","Fare"],aggfunc=np.mean)
        # 去掉某行/列
        f.dropna(axis=0,subset=["Age","Sex"])
        # 定位 - 第 83 个样本的 Age 值
        value = f.loc[83,"Age"]
        # apply() - 提升可读性
        def hundredth_row(column):
            hundredth_item = column.loc[99]
            return hundredth_item
        hundredth_row = survival.apply(hundredth_row)
        ```python

    + ### series: DataFrame 的子集

- ## Matplotlib

    + ### 基础知识

        ```python
        import matplotlib.pyplot as plt
        # 画图，将参数传入
        plt.plot(x, y)
        # 将要画的东西显示出来
        plt.show()
        # x 轴名称倾斜
        # 类似下面将介绍的 ax.set_xticklables(rotation=45)
        plt.xticks(rotation=45)
        # 加标签，标题
        plt.xlable('Month')
        plt.ylable('Day')
        plt.title('Relativation')
        ```

    + ### 子图操作

        ```python
        # 指定画图的区间
        fig = plt.figure()
        # 参数指定画图大小
        fig = plt.figure(figsize=(3,3))
        # x 代表第几个子图
        fig.add_subplot(4,1,x)
        # 分别对子图进行操作
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        ax1.plot(np.random.randint(1,5,5), np.arange(5))
        ax2 = plot(np.arange(10)*3, np.arange(10))
        # 一个图中画多个对象 plt.show() 之前定义好即可
        fig = plt.figure(figsize(10,6))
        colors = ['red', 'blue', 'green']
        for i in range(3):
            start_index = i*12
            end_index = (i+1)*12
            subset = unrate[startx_index:end_index]
            plt.plot(subset['Month'], subset['Day'], c=colors[i])

        plot.show()
        # 加图例，并自动选择最佳位置
        plt.legend(loc='best')
        ```

    + ### 条形图

        ```python
        # 指定条形图的位置、高度
        num_cols = ['Pclass', 'Sex']
        bar_heights = norm_review.ix[0, num_cols].values
        bar_positions = arange(5) + 0.75
        # ax 作图实际上的一个轴，之后在轴上定义内容；fig 定义整体风格
        fig, ax = plt.subplots()
        # 0.3 宽度可调
        ax.bar(bar_positions, bar_heights, 0.3)
        # 将图横起来
        ax.barh(bar_positions, bar_heights, 0.3)
        ```

    + ### 散点图

        ```python
        ax.scatter(norm_reviews['Age'], norm_reviews['Sex'])
        ax.set_xlable('Age')
        ax.set_xlable('Sex')
        ```

    + ### 柱形图 bins - 统计某区间段的值

        ```python
        fig, ax = plt.subplots()
        # range() 指定起始结束时间
        ax.hist(norm_reviews['Pclass'], range=(4,5), bins=20)
        # 限制轴上区间
        ax.set_xlim(4, 7)
        ```

    + ### 盒图

        ```python
        fig, ax = plt.subplots()
        ax.boxplot(norm_reviews['Pclass'])
        ```

    + ### 细节设置

        ```python
        # 去掉下面坐标轴上的小齿
        ax.tick_params(bottom="off")
        # 颜色
        dark_blue = (0/255, 107/255, 164/255)
        # 图中曲线上添加文字
        ax.test(x, y, 'text')
        ```

- ## Seaborn

    > 在 matplotlib 上面进行了封装，节省代码量

    + ### 基础操作

        ```python
        import seaborn as sns
        # 将图直接显示在 notebook 中
        %matplotlib inline
        def sinplot(flip=1):
            # 在 (0, 14) 生成 100 个点
            x = np.linspace(0, 14, 100)
            for i in range(1, 7):
                plt.plot(x, np.sin(x+i*.5)*(7-i)*flip)
        # 使用 seaborn 默认风格
        sns.set()
        sinplot()
        # 五种主题风格 darkgrid, whitegrid, dark, white, ticks
        sns.set_style("whitegrid")
        data = np.random.normal(size=(20,6)+np.arrange(6)/2)
        sns.boxplot(data = data)
        # 去掉上面、右边的轴
        sns.despine()
        # 隐藏左边的轴
        sns.despine(left=True)
        # 设置图离轴线距离
        sns.despine(offset=10)
        # 整体风格
        sns.set_context("notebook",font_scale=1.5,rc={"lines.linewidth":2.5})
        ```

    + ### 调色板

        ```python
        #  能传入任何 Matplotlib 所支持的颜色
        current_palette = color_palette()
        sns.palplot(current_palette)
        # 默认颜色循环主题
        deep, muted, pastel, bright, dark, colorblind
        # 圆形画板 - 六个以上分类要区分
        sns.palplot(sns.color_palette("hls", 8))
        # lightness 亮度；saturation 饱和度
        sns.palplot(sns.hls_palette(8, l=.7, s=.9))
        # 相邻颜色成对出现
        sns.palplot(sns.color_palette("Paired", 8))
        # 使用 xkcd 颜色来命名颜色
        plt.plot([0, 1], [0, 1], sns.xkcd_rgb["pale red"], lw=3)
        # 连续色板 - 色彩随数据变换
        # 数据越重要颜色越深
        sns.palplot(sns.color_palette("Blues"))
        # 翻转渐变： _r 后缀
        sns.palplot(sns.color_palette(BuGn_r"))
        # 色调线性变换
        ```

    + ### 单变量分析绘图

        ```python
        # 直方图绘出随机生成得数据
        x = np.random.narmal(size=100)
        # kde 是否做呵密度估计
        sns.displot(x,kde=False)
        # 轮廓
        sns.displot(x, bins=20, kde=False， fit=stats.gramma)
        # 根据均值和协方差生成数据
        mean, cov = [0,1], [(1,.5), (.5,1)]
        data = np.random.multivarate_normal(mean, cov, 200)
        df = pd.DataFrame(data, columns=["x", "y"])
        # 观测两个变量之间分布关系最好用散点图
        sns.jiontplot(x="x", y="y", data=df)
        # hex 图(马赛克)，改进:数据太多，重叠看不出来单个样本点
        sns.jiontplot(x="x", y="y", kind="hex", data=df)
        ```

    + ### 回归分析绘图

        ```python
        sns.replot(x="x", y="y",data=data_name)
        ```

    + ### 多变量分析绘图

        ```python
        # 将全部数据画到图中
        sns.stripplot(x="x", y="y",data=data_name)
        # 加入偏移量，改善样本重叠现象
        sns.stripplot(x="x", y="y",data=data_name， jitter=True)
        # 盒图 - 分析离群点
        sns.boxplot(x="x", y="y", hue="time", data=data_name)
        # 小提琴图 - 依 性别 属性颜色区分
        sns.violinplot(x="x", y="y", hue="sex", data=data_name， split=True)
        ```

    + ### 分类属性绘图

        ```python
        # 显示值的集中趋势可以用条形图
        sns.barplot(x="sex", y="survived", hue="class", data=titanic)
        # 点图可以更好地描述变化差异
        sns.pointplot(x="sex", y="survived", hue="class", data=titanic)
        # 定制化
        sns.pointplot(x="sex", y="survived", hue="class", data=titanic， palette={"male":"g", "female":"m"}, markers=["^","o"], linestyles=["-","--"])
        ```

    + ### 热度图

        ```python
        # 生成随机 3*3 矩阵并作图
        uniform_data = np.random.rand(3,3)
        heatmap = sns.heatmap(uniform_data)
        # 取值限定
        ax = sns.heatmap(uniform_data, vmin=0.2, vmax=0.5)
        # 限定中心值
        ax = sns.heatmap(uniform_data, center=0)
        # 指定坐标轴，及对应位置地值
        flights = flights.pivot("month", "year", "passengers")
        ax = sns.heatmap(flights)
        # 热度图中显示相应区块地值，并指定字体
        ax = sns.heatmap(flights， annot=True,fmt="d")
        ```
