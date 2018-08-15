# Seaborn

在 matplotlib 上面进行了封装, 节省代码量.

- ## 基础操作

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

- ## 调色板

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

- ## 单变量分析绘图

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

- ## 回归分析绘图

    ```python
    sns.replot(x="x", y="y",data=data_name)
    ```

- ## 多变量分析绘图

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

- ## 分类属性绘图

    ```python
    # 显示值的集中趋势可以用条形图
    sns.barplot(x="sex", y="survived", hue="class", data=titanic)

    # 点图可以更好地描述变化差异
    sns.pointplot(x="sex", y="survived", hue="class", data=titanic)

    # 定制化
    sns.pointplot(x="sex", y="survived", hue="class", data=titanic， palette={"male":"g", "female":"m"}, markers=["^","o"], linestyles=["-","--"])
    ```

- ## 热度图

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
