# support vector machines - SVM

## Gist

![](images/gist.png)

## Theory

1. ### linear SVM target function

    <img src="images/svm.jpg" width="50%" height="50%" align=center />

    > hyperplane is <a href="https://www.codecogs.com/eqnedit.php?latex=W^{T}X^{'}&plus;b=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{T}X^{'}&plus;b=0" title="W^{T}X^{'}+b=0" /></a>

    + #### 支持向量到分类直线的距离
        > 参考投影方法(点到直线距离)

        ![](images/dis_vec_to_hyperplane.PNG)

        <img src="https://latex.codecogs.com/gif.latex?\Rightarrow&space;distance(X,b,W)=\left&space;|&space;\frac{W^{T}}{\left&space;\|&space;W&space;\right&space;\|}&space;(X-X^{'})\right&space;|=\frac{1}{\left&space;\|&space;W&space;\right&space;\|}\left&space;|&space;W^{T}X&plus;b&space;\right&space;|" title="\Rightarrow distance(X,b,W)=\left | \frac{W^{T}}{\left \| W \right \|} (X-X^{'})\right |=\frac{1}{\left \| W \right \|}\left | W^{T}X+b \right |" />

        * ##### 函数间隔、几何间隔

            样本类别设置为 {-1, +1}，带入样本判断所属类别

            <img src="https://latex.codecogs.com/gif.latex?\begin{cases}&space;&&space;y_{i}=&plus;1,\text{&space;if&space;}&space;y(x_{i})>&space;0&space;\\&space;&&space;y_{i}=-1,\text{&space;if&space;}&space;y(x_{i})<&space;0&space;\end{cases}&space;\Rightarrow&space;y(i)*y(x_{i})>&space;0" title="\begin{cases} & y_{i}=+1,\text{ if } y(x_{i})> 0 \\ & y_{i}=-1,\text{ if } y(x_{i})< 0 \end{cases} \Rightarrow y(i)*y(x_{i})> 0" />

            故 <img src="https://latex.codecogs.com/gif.latex?w·x&plus;b" title="w·x+b" /> 的符号与类标记 <img src="https://latex.codecogs.com/gif.latex?y" title="y" /> 的符号是否一致能够表示分类是否正确。所以可用量 <img src="https://latex.codecogs.com/gif.latex?y(w·x&plus;b)" title="y(w·x+b)" /> 来表示分类的正确性和确信度，这就是**函数间隔**(functional margin)的概念。函数间隔为

            <img src="https://latex.codecogs.com/gif.latex?\widehat{\gamma}=y_{i}(w·x_{i}&plus;b)" title="\widehat{\gamma}=y_{i}(w·x_{i}+b)" />

            选择超平面时，成比例改变 w 和 b，超平面没有改变，而函数间隔却会发生变化。故，对法向量 w 加某些约束，如规范化，<img src="https://latex.codecogs.com/gif.latex?\left&space;\|&space;w&space;\right&space;\|=1" title="\left \| w \right \|=1" />，使间隔确定 ==> **几何间隔(geometric margin)**。

            即，<img src="https://latex.codecogs.com/gif.latex?\frac{w}{\left&space;\|&space;w&space;\right&space;\|}·x_{i}&plus;\frac{b}{\left&space;\|&space;w&space;\right&space;\|}" title="\frac{w}{\left \| w \right \|}·x_{i}+\frac{b}{\left \| w \right \|}" />
            > 若点在超平面负的一侧，则整体 <img src="https://latex.codecogs.com/gif.latex?*(-1)" title="*(-1)" />

            当样本点 <img src="https://latex.codecogs.com/gif.latex?(x_{i},y_{i})" title="(x_{i},y_{i})" /> 被超平面正确分类时，点 <img src="https://latex.codecogs.com/gif.latex?x_{i}" title="x_{i}" /> 与超平面距离是

            <img src="https://latex.codecogs.com/gif.latex?\gamma&space;=y_{i}(\frac{w}{\left&space;\|&space;w&space;\right&space;\|}·x_{i}&plus;\frac{b}{\left&space;\|&space;w&space;\right&space;\|})" title="\gamma =y_{i}(\frac{w}{\left \| w \right \|}·x_{i}+\frac{b}{\left \| w \right \|})" />

            > 样本类别设置为 {-1, +1}，使距离始终为正值

    + #### 下面围绕着求得一个间隔最大的分离超平面进行展开

        **_间隔与分类问题的结构风险有关，最大化间隔等于最小化结构风险，从而得到一个更好的分类器 !_**

        <img src="https://latex.codecogs.com/gif.latex?\underset{w,b}{max}\text{&space;}\gamma" title="\underset{w,b}{max}\text{ }\gamma" />


        <img src="https://latex.codecogs.com/gif.latex?\text{s.t.&space;}y_{i}(\frac{w}{\left&space;\|&space;w&space;\right&space;\|}·x_{i}&plus;\frac{b}{\left&space;\|&space;w&space;\right&space;\|})\geq&space;\gamma&space;,i=1,2,...,N" title="\text{s.t. }y_{i}(\frac{w}{\left \| w \right \|}·x_{i}+\frac{b}{\left \| w \right \|})\geq \gamma ,i=1,2,...,N" />

1. ### 化简

    - #### 化为函数间隔形式

        <img src="https://latex.codecogs.com/gif.latex?\underset{w,b}{max}\text{&space;}\frac{\widehat{\gamma&space;}}{\left&space;\|&space;w&space;\right&space;\|}" title="\underset{w,b}{max}\text{ }\frac{\widehat{\gamma }}{\left \| w \right \|}" />


        <img src="https://latex.codecogs.com/gif.latex?\text{s.t.&space;}y_{i}(w·x_{i}&plus;b)\geq&space;\widehat{\gamma}&space;,i=1,2,...,N" title="\text{s.t. }y_{i}(w·x_{i}+b)\geq \widehat{\gamma} ,i=1,2,...,N" />

        函数间隔 <img src="https://latex.codecogs.com/gif.latex?\widehat{\gamma}" title="\widehat{\gamma}" /> 取值不影响最优化问题的解，可令 <img src="https://latex.codecogs.com/gif.latex?\widehat{\gamma}=1" title="\widehat{\gamma}=1" />

        > w,b 按比例改变为 <img src="https://latex.codecogs.com/gif.latex?\lambda&space;w,\lambda&space;b" title="\lambda w,\lambda b" />，这时函数间隔为 <img src="https://latex.codecogs.com/gif.latex?\lambda\widehat{\gamma}" title="\lambda\widehat{\gamma}" />，故并无影响

    + #### 去绝对值

        <img src="https://latex.codecogs.com/gif.latex?max\text{&space;}\frac{1}{\left&space;\|&space;w&space;\right&space;\|}\Leftrightarrow&space;min\text{&space;}\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2}" title="max\text{ }\frac{1}{\left \| w \right \|}\Leftrightarrow min\text{ }\frac{1}{2}\left \| w \right \|^{2}" />

    + #### 最终目标函数登场

        <img src="https://latex.codecogs.com/gif.latex?\underset{w,b}{min}\text{&space;}\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2}" title="\underset{w,b}{min}\text{ }\frac{1}{2}\left \| w \right \|^{2}" />


        <img src="https://latex.codecogs.com/gif.latex?\text{s.t.&space;}y_{i}(w·x_{i}&plus;b)-1\geq&space;0&space;,i=1,2,...,N" title="\text{s.t. }y_{i}(w·x_{i}+b)-1\geq 0 ,i=1,2,...,N" />

        > 条件是必不可少的 - 表达意思等价于:为了使得所有样本数据都在间隔区(两条虚线)以外。否则，引出*soft hard SVM* (容错)

1. ### 引入拉格朗日乘子法

    上述为凸优化问题。常用**拉格朗日乘子法(Method of lagrange multiplier)**，它用来求解在约束条件目标函数的极值。其标准形式为

    <img src="https://latex.codecogs.com/gif.latex?\begin{cases}&space;min&space;f(x)&space;\\&space;\text{&space;s.t.&space;}&space;g_{i}(x)\leq&space;0,&space;i=1,...,n&space;\end{cases}" title="\begin{cases} min f(x) \\ \text{ s.t. } g_{i}(x)\leq 0, i=1,...,n \end{cases}" />

    + #### 带入到目标函数中

        引入 n 个拉格朗日乘法子(因为有 n 个约束),记为<img src="https://latex.codecogs.com/gif.latex?\lambda&space;=(\lambda_{1},..,\lambda_{n})^{T}" title="\lambda =(\lambda_{1},..,\lambda_{n})^{T}" />

        <img src="https://latex.codecogs.com/gif.latex?L(w,b,\lambda&space;)=\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2}-\sum_{i=1}^{n}\lambda&space;_{i}(y_{i}(w^{T}\Phi&space;(x_{i})&plus;b)-1)" title="L(w,b,\lambda )=\frac{1}{2}\left \| w \right \|^{2}-\sum_{i=1}^{n}\lambda _{i}(y_{i}(w^{T}\Phi (x_{i})+b)-1)" />

        > <img src="https://latex.codecogs.com/gif.latex?\Phi&space;(x)" title="\Phi (x)" /> 代替 x,意指经过了某种变换

        解这个拉格朗日方程,即对 **w,b** 求偏导,得

        <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;w}=0\Rightarrow&space;w=\sum_{i=1}^{n}\lambda&space;_{i}y_{i}\Phi&space;(x_{n})" title="\frac{\partial L}{\partial w}=0\Rightarrow w=\sum_{i=1}^{n}\lambda _{i}y_{i}\Phi (x_{n})" />

        <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;b}=0\Rightarrow&space;0=\sum_{i=1}^{n}\lambda&space;_{i}y_{i}" title="\frac{\partial L}{\partial b}=0\Rightarrow 0=\sum_{i=1}^{n}\lambda _{i}y_{i}" />

        通过上面两个公式,完成了第一步求解

        <img src="https://latex.codecogs.com/gif.latex?\underset{w,b}{min}L(w,b,\lambda&space;)" title="\underset{w,b}{min}L(w,b,\lambda )" />

        将上面两个条件带回<img src="https://latex.codecogs.com/gif.latex?L(w,b,\lambda&space;)" title="L(w,b,\lambda )" />,进行下一步的计算。

    + #### 通过其对偶问题求解,以简化计算

        > 原问题的解包含在对偶问题的解中，等价最优化。转换为对偶问题能得到更高效的解法，也方便了核函数的引入。

        为什么支持向量机要用拉格朗日对偶算法来解最大化间隔问题？

            1) 不等式约束一直是优化问题中的难题，求解对偶问题可以将支持向量机原问题约束中的不等式约束转化为等式约束；
            2) 支持向量机中用到了高维映射，但是映射函数的具体形式几乎完全不可确定，而求解对偶问题之后，可以使用核函数来解决这个问题。

        > 用拉格朗日对偶并没有改变最优解，而是改变了算法复杂度

        在当前情境下

        <img src="https://latex.codecogs.com/gif.latex?\underset{w,b}{min}\text{&space;}\underset{\lambda&space;}{max}L(w,b,\lambda&space;)&space;\rightarrow&space;\underset{\lambda&space;}{max}\text{&space;}\underset{w,b}{min}L(w,b,\lambda&space;)" title="\underset{w,b}{min}\text{ }\underset{\lambda }{max}L(w,b,\lambda ) \rightarrow \underset{\lambda }{max}\text{ }\underset{w,b}{min}L(w,b,\lambda )" />

        转化为对偶形式,则目标函数变为最大化<img src="https://latex.codecogs.com/gif.latex?L(\lambda)" title="L(\lambda)" />

    + #### 即,对偶形式后的任务是求解

        <img src="https://latex.codecogs.com/gif.latex?\text{arg&space;}\underset{\lambda&space;}{max}L(w,b,\lambda&space;)=\sum_{i=1}^{n}\lambda&space;_{i}-\frac{1}{2}\sum_{i=1}^{n}\lambda&space;_{i}\lambda&space;_{j}y_{j}\Phi&space;^{T}(x_{i})\Phi&space;(x_{i})" title="\text{arg }\underset{\lambda }{max}L(w,b,\lambda )=\sum_{i=1}^{n}\lambda _{i}-\frac{1}{2}\sum_{i=1}^{n}\lambda _{i}\lambda _{j}y_{j}\Phi ^{T}(x_{i})\Phi (x_{i})" />

        <img src="https://latex.codecogs.com/gif.latex?\text{s.t.&space;}&space;\lambda&space;_{i}\geq&space;0,\forall&space;i;\sum_{i=1}^{n}\lambda&space;_{i}y&space;_{i}=0" title="\text{s.t. } \lambda _{i}\geq 0,\forall i;\sum_{i=1}^{n}\lambda _{i}y _{i}=0" />

    + #### 最终模型

        以上表达式通过**二次规划**算法解出<img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" />后,带回,求出 w 和 b,即可得到模型

        <img src="https://latex.codecogs.com/gif.latex?f(x)=W^{T}X&plus;b=\sum_{i=1}^{m}\lambda&space;_{i}y&space;_{i}X&space;_{i}^{T}X&plus;b" title="f(x)=W^{T}X+b=\sum_{i=1}^{m}\lambda _{i}y _{i}X _{i}^{T}X+b" />

    _以上即为求解**支持向量**和**分类超平面**的过程._

    _具体问题求解中,首先根据公式求得<img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" />,之后利用<img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" />与 w,b 的关系,求得 w,b,即得分类超平面._

1. ### 软间隔 SVM

    <img src="images/软间隔.PNG" width="35%" height="35%" />

    离群点/异常点 情况下,引入**松弛因子 <img src="https://latex.codecogs.com/gif.latex?\xi&space;_{i}" title="\xi _{i}" />** (slack variables)以放宽条件,提高分类性能。

    则约束条件变为

    <img src="https://latex.codecogs.com/gif.latex?y_{i}(w^{T}\Phi&space;(x_{i})&plus;b)&space;\right\geq&space;1-\xi_{i}" title="y_{i}(w^{T}\Phi (x_{i})+b) \right\geq 1-\xi_{i}" />

    则目标函数变成了

    <img src="https://latex.codecogs.com/gif.latex?\underset{w,b,\xi&space;_{i}&space;}{min}\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2}&plus;C\sum_{i=1}^{n}\xi&space;_{i}" title="\underset{w,b,\xi _{i} }{min}\frac{1}{2}\left \| w \right \|^{2}+C\sum_{i=1}^{n}\xi _{i}" />

    > 其中 C 为系数,C 趋近于无穷大时,意味着分类严格不能有错误;C 趋近于很小时,意味着有更大的错误容忍(C 值自己设定)

    拉格朗日形式为

    <img src="https://latex.codecogs.com/gif.latex?L(w,b,\xi&space;,\lambda&space;,\mu&space;)\equiv&space;\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^{2}&plus;C\sum_{i=1}^{n}\xi&space;_{i}-\sum_{i=1}^{n}\lambda&space;_{i}(y_{i}(wx_{i}&plus;b)-1&plus;\xi_{i})-\sum_{i=1}^{n}\mu&space;_{i}\xi_{i}" title="L(w,b,\xi ,\lambda ,\mu )\equiv \frac{1}{2}\left \| w \right \|^{2}+C\sum_{i=1}^{n}\xi _{i}-\sum_{i=1}^{n}\lambda _{i}(y_{i}(wx_{i}+b)-1+\xi_{i})-\sum_{i=1}^{n}\mu _{i}\xi_{i}" />

    其求解过程和 hard margin 时类似.

## 回顾上述求解过程

![](images/procedure.png)

## non-linear SVM

### kernel function

![](images/kernel_func.gif)

### 使用高斯核函数变换示意

![](images/gauss_kernel.PNG)

- 常见核函数

    ![](images/kernel_function.jpg)

## 多类问题

多类问题可以使用两两做支持向量机，再由所有支持向量机投票选出这个类别的归属。即 *one-versus-one approace*.

## practice on data mining

```python
from sklearn.svm import SVC, LinearSVC
#{'C':[0.01,0.1,1,10],'gamma':[0.01,0.1,1,10]}
svc = SVC(C=0.88, gamma=0.8)
svc.fit(train_set, test_set)
prediction = svc.predict(verify_set)
```
> Best(Titanic) --> kernel='linear', C=0.025

- g gamma : set gamma in kernel function(default 1/num_features)
- c cost : set the parameter C of C-SVC,epsilon-SVR, and nu-SVR (default 1)

        C是惩罚系数，即对误差的宽容度。
        c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。
        C过大或过小，泛化能力变差

        gamma是选择RBF函数作为kernel后，该函数自带的一个参数。
        隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，gamma值越小，支持向量越多。
        支持向量的个数影响训练与预测的速度。

之前看《统计学习方法》时候留的粗略的记录,可以找到 [SVM 部分](https://github.com/luanxxys/computer_science/blob/master/machine_learning/%E3%80%8A%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E3%80%8B/readme.md)看看.
