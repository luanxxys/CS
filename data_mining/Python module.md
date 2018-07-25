# Python 库

### Scipy

Scipy 库依赖于 NumPy，它提供便捷和快速的 N 维向量数组操作。

SciPy 库的建立就是和 NumPy 数组一起工作，并提供许多对用户友好的和有效的数值例程，如：数值积分和优化。

SciPy 提供模块用于优化、线性代数、积分以及其它数据科学中的通用任务。

### Matplotlib

Matlplotlib 是 Python 的一个可视化模块。它让你方便地制作线条图、饼图、柱状图以及其它专业图形。

使用 Matplotlib，你可以定制所做图表的任一方面。

在 IPython 中使用时，Matplotlib 有一些互动功能，如：缩放和平移。它支持所有的操作系统下不同的 GUI 后端(back ends)，并且可以将图形输出为常见地矢量图和图形格式，如：PDF、SVG、JPG、PNG、BMP 和 GIF 等。

### Scikit-learn

Scikit-learn 是一个用于机器学习的 Python 模块。

它建立在 Scipy 之上，提供了一套常用机器学习算法，让使用者通过一个统一的接口来使用。

Scikit-learn 有助于你迅速地在你的数据集上实现流行的算法。

看一下 Scikit-learn 中提供的算法列表，你就会马上意识到它包含了许多用于标准机器学习任务的工具，如：聚类、分类和回归等。

#### 部分功能介绍

##### 管道（Pipeline）

这可以用来将多个估计量链化合一。因为在处理数据时，通常有着一系列固定的步骤，比如特征选择、归一化和分类，此时这个方法将非常有用。

更多信息：http://scikit-learn.org/stable/modules/pipeline.html

##### 网格搜索（Grid-search）

超参数在参数估计中是不直接学习的，在scikit-learn库中，超参数会作为参数传递给估计类的构造函数，然后在超参数空间中搜索最佳的交叉验证分数在构建参数估计量时提供的任何参数都是可以用这种方式进行优化的。

更多信息：http://scikit-learn.org/stable/modules/grid_search.html#grid-search

##### 验证曲线（Validation curves）

每种估计方法都有其优缺点，它的泛化误差可以用偏差、方差和噪音来分解。估计量的偏差就是不同训练集的平均误差；估计量的方差是表示对不同训练集的敏感程度；噪声是数据本身的一个属性。

绘制单个超参数对训练分数和验证分数的影响是非常有用的，因为从图中可以看出估计量对于某些超参数值是过拟合还是欠拟合。在Scikit-learn库中，有一个内置方法是可以实现以上过程的。

更多信息：http://scikit-learn.org/stable/modules/learning_curve.html

##### 分类数据的独热编码（One-hot encoding of categorical data）

这是一种非常常见的数据预处理步骤，在分类或预测任务中（如混合了数量型和文本型特征的逻辑回归），常用于对多分类变量进行二分类编码。Scikit-learn库提供了有效而简单的方法来实现这一点。它可以直接在Pandas数据框或Numpy数组上运行，因此用户就可以为这些数据转换编写一些特殊的映射函数或应用函数。

更多信息：http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features

##### 多项式特征生成（Polynomial feature generation）

对于无数的回归建模任务来说，一种常用的增加模型复杂程度的有效方法是增加解释变量的非线性特征。一种简单而常用的方法就是多项式特征，因为它可以得到特征的高阶项和交叉项。而Scikit-learn库中有现成的函数，它可根据给定的特征集和用户选择的最高多项式生成更高阶的交叉项。

更多信息：http://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features

##### 数据集生成器（Dataset generators）

Scikit-learn库包含各种随机样本生成器，可以根据不同大小和复杂程度来构建人工数据集，且具有分类、聚类、回归、矩阵分解和流形测试的功能。

更多信息：http://scikit-learn.org/stable/datasets/index.html#sample-generators
