# Learn Tensorflow

- ## 综述

    TensorFlow 是一个编程系统, 使用图来表示计算任务.

    **图中的节点被称之为 *op* (operation)**. 一个 op 获得 0 个或多个 `Tensor`, 执行计算, 产生 0 个或多个 `Tensor `.

    **每个 Tensor 是一个类型化的多维数组**. 例如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 `[batch, height, width, channels]`.

    一个 TensorFlow 图*描述*了计算的过程. 为了进行计算, 图必须在 `会话` 里被启动.

    `会话` 将图的 op 分发到诸如 CPU 或 GPU 之类的 `设备` 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回.

    在 Python 语言中, 返回的 tensor 是 [numpy](http://www.numpy.org) `ndarray` 对象; 在 C 和 C++ 语言中, 返回的 tensor 是 `tensorflow::Tensor` 实例.

    **即 TensorFlow 基本使用:**

    + 使用 `graph` 来表示计算任务
    + 在被称之为 `会话 (Session)` 的上下文 (context) 中执行图
    + 使用 `tensor` 表示数据
    + 通过 `变量 (Variable)` 维护状态
    + 使用 `feed` 和 `fetch` 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据

- ## 运算流程 - 计算图

    TensorFlow 程序通常被组织成一个**构建阶段**和一个**执行阶段**.

        在构建阶段, op 的执行步骤被描述成一个图
        在执行阶段, 使用会话执行执行图中的 op

    例如, 通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练 op.

    + ### 构建图

        创建源 op (source op)

            源 op 不需要任何输入, 例如 `常量 (Constant)`
            源 op 的输出被传递给其它 op 做运算

        Python 库中, op 构造器的返回值代表被构造出的 op 的输出, 这些返回值可以传递给其它 op 构造器作为输入.

        TensorFlow Python 库有一个*默认图 (default graph)*, op 构造器可以为其增加节点. 这个默认图对许多程序来说已经足够用了. 阅读 [Graph 类](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/framework.md#Graph) 文档来了解如何管理多个图.

    + ### 在一个会话中启动图

        构造阶段完成后, 才能启动图. 启动图的第一步是创建一个 `Session` 对象, 如果无任何创建参数, 会话构造器将启动默认图.

        欲了解完整的会话 API, 请阅读[Session 类](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/client.md#session-management).

        两种方式

        1. sess = tf.Session(); sess.close()
        2. with tf.Session() as sess:

        ##### 使用 GPU 进行分布式计算

            with tf.device("/gpu:1"):

        ##### 交互式使用

### _以上操作都辅有[代码说明](learn_tensorflow.ipynb)及[示例](tensorflow_examples.ipynb)._

- ## 一些概念

    + ### Devices

        一块可以用来运算并且拥有自己的地址空间的硬件，比如 GPU 和 CPU.

    + ### eval

        Tensor 的一个方法，返回 Tensor 的值. 触发任意一个图计算都需要计算出这个值.

        只能在一个已经启动的会话的图中才能调用该 Tensor 值.

    + ### Feed

        把一个 Tensor 直接连接到一个会话图表中的任意节点.

        feed 不是在构建图(graph)的时候创建，而是在触发图的执行操作时去申请.

        一个 feed 临时替代一个带有 Tensor 值的节点. 把feed数据作为run( )方法和eval( )方法的参数来初始化运算. 方法运行结束后，替换的 feed 就会消失，而最初的节点定义仍然还在.

        可以通过tf.placeholder( )把特定的节点指定为 feed 节点来创建它们. 详见[Basic Usage](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/get_started/basic_usage.md).

    + ### Fetch

        取回运算操作的输出结果.

        取回的申请发生在触发执行图操作的时候，而不是发生在建立图的时候.

        如果要取回一个或多个节点（node）的 Tensor 值，可以通过在 Session 对象上调用run( )方法并将待取回节点（node）的列表作为参数来执行图表(graph). 详见[Basic Usage](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/get_started/basic_usage.md).

    + ### Graph

        把运算任务描述成一个直接的无环图形（DAG），图表中的节点（node）代表必须要实现的一些操作. 图中的边代表数据或者可控的依赖.

    + ### Node

        图中的一个元素.

        把启动一个特定操作的方式称为特定运算图表的一个节点，包括任何用来配置这个操作的属性的值.

    + ### 操作（Op/operation）

        在 TensorFlow 的运行中，它是一种类似 add 或 matmul 或 concat的运算. 可以用[how to add an op](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/how_tos/adding_an_op/index.md)中的方法来向运行时添加新的操作.

        在 Python 的API中，它是图中的一个节点. 在[tf.Operation](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/framework.md#Operation)类中列举出了这些操作. 一个操作(Operation)的 type 属性决定这个节点（node）的操作类型，比如add和matmul.

    + ### Run

        在一个运行的图中执行某种操作的行为. 要求图必须运行在会话中.

        在 Python 的 API 中，它是 Session 类的一个方法[tf.Session.run](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/client.md#Session). 可以通过 Tensors 来订阅或获取run( )操作.

        在C++的API中，它是[tensorflow::Session](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/client.md#Session)类的一个方法.

    + ### Session

        启动图的第一步是创建一个 Session 对象. Session 提供在图中执行操作的一些方法.

        在 Python API中，使用[tf.Session](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/client.md#Session).

        在 C++ 的API中，[tensorflow::Session](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/cc/ClassSession.md)是用来创建一个图并运行操作的类.

    + ### Shape

        Tensor 的维度和它们的大小.

        在一个已经启动的图中，它表示流动在节点（node）之间的 Tensor 的属性. 一些操作对 shape 有比较强的要求，如果没有 Shape 属性则会报告错误.

        在 Python API中，用创建图的 API 来说明 Tensor 的 Shape 属性. Tensor 的Shape 属性要么只有部分已知，要么全部未知. 详见[tf.TensroShape](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/framework.md#TensorShape)

        在C++中，Shape 类用来表示 Tensor 的维度. [tensorflow::TensorShape](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/cc/ClassTensorShape.md).

    + ### SparseTensor

        在 Python API 中，它用来表示在 TensorFlow 中稀疏散落在任意地方的 Tensor .

        SparseTensor 以字典-值格式来储存那些沿着索引的非空值. 换言之，m个非空值，就包含一个长度为m的值向量和一个由m列索引(indices)组成的矩阵.

        为了提升效率，SparseTensor 需要将 indice（索引）按维度的增加来按序存储，比如行主序. 如果稀疏值仅沿着第一维度，就用 IndexedSlices.

    + ### Tensor

        Tensor是一种特定的多维数组.

        在一个运行的图(graph)中，它是一种流动在节点（node）之间的数据.

        在 Python 中，Tensor 类表示添加到图的操作中的输入和输出，见[tf.Tensor](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/framework.md#Tensor)，这样的类不持有数据.

        在C++中，Tensor是方法[Session::Run( )](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/cc/ClassSession.md)的返回值，见[tensorflow::Tensor](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/cc/ClassTensor.md)，这样的 Tensor 持有数据.

    + ### 变量

        当训练模型时，用[变量](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/api_docs/python/state_ops.md)来存储和更新参数.

        变量包含张量 (Tensor), 存放于内存的缓存区. 建模时它们需要被明确地初始化，模型训练后它们必须被存储到磁盘. 这些变量的值可在之后模型训练和分析是被加载.

        * #### 创建

            当创建一个[变量](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/api_docs/python/state_ops.md)时，你将一个`张量`作为初始值传入构造函数`Variable()`.

            TensorFlow提供了一系列操作符来初始化张量，初始值是[常量或是随机值](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/api_docs/python/constant_op.md).

            注意，所有这些操作符都需要你指定张量的shape. 那个形状自动成为变量的shape. 变量的shape通常是固定的，但TensorFlow提供了高级的机制来重新调整其行列数.

            > 所有这些操作符都需要你指定张量的 shape. 那个形状自动成为变量的 shape

            变量的 shape 通常是固定的，但 TensorFlow 提供了高级的机制来重新调整其行列数.

            ```python
            # Create two variables.
            weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                                  name="weights")
            biases = tf.Variable(tf.zeros([200]), name="biases")
            ```

            `tf.Variable`的返回值是Python的`tf.Variable`类的一个实例.

        * #### 初始化

            1. 并行地初始化所有变量

                变量的初始化必须在模型的其它操作运行之前先明确地完成. 最简单的方法就是添加一个给所有变量初始化的操作，并在使用模型之前首先运行那个操作.

                使用`tf.initialize_all_variables()`添加一个操作对变量做初始化.
                > 记得在完全构建好模型并加载之后再运行此操作

            1. 由另一个变量初始化

                用其它变量的值初始化一个新的变量时，使用其它变量的initialized_value()属性.

                ```python
                # Create a variable with a random value.
                weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                                      name="weights")
                # Create another variable with the same value as 'weights'.
                w2 = tf.Variable(weights.initialized_value(), name="w2")
                # Create another variable with twice the value of 'weights'
                w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")
                ```

            1. 自定义初始化

                给 `tf.initialize_all_variables()` 函数传入一组变量进行初始化. 详情请见[Variables Documentation](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/api_docs/python/state_ops.md)，包括检查变量是否被初始化.

### reference

- [TensorFlow 官方文档中文版](https://github.com/jikexueyuanwiki/tensorflow-zh/)

- [Learn TensorBoard](tensorboard/readme.md)

- [CUDA 指定 GPU 运行 TensorFlow](tensorflow_GPU.md)
