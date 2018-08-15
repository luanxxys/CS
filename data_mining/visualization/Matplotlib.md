# Matplotlib

- ## 基础知识

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

- ## 子图操作

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

- ## 条形图

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

- ## 散点图

    ```python
    ax.scatter(norm_reviews['Age'], norm_reviews['Sex'])
    ax.set_xlable('Age')
    ax.set_xlable('Sex')
    ```

- ## 柱形图 bins - 统计某区间段的值

    ```python
    fig, ax = plt.subplots()

    # range() 指定起始结束时间
    ax.hist(norm_reviews['Pclass'], range=(4,5), bins=20)

    # 限制轴上区间
    ax.set_xlim(4, 7)
    ```

- ## 盒图

    ```python
    fig, ax = plt.subplots()
    ax.boxplot(norm_reviews['Pclass'])
    ```

- ## 细节设置

    ```python
    # 去掉下面坐标轴上的小齿
    ax.tick_params(bottom="off")

    # 颜色
    dark_blue = (0/255, 107/255, 164/255)

    # 图中曲线上添加文字
    ax.test(x, y, 'text')
    ```
