# Bayes algorithm

- ## 贝叶斯公式

    <img src="https://latex.codecogs.com/gif.latex?P(A|B)=\frac{P(B|A)P(A)}{P(B)}" title="P(A|B)=\frac{P(B|A)P(A)}{P(B)}" />

        P(A) 是 A 的先验概率或边缘概率，称作 "先验" 是因为它不考虑 B 因素
        P(A|B) 是已知 B 发生后 A 的条件概率，也称作 A 的后验概率
        P(B|A) 是已知 A 发生后 B 的条件概率，也称作 B 的后验概率，这里称作似然度
        P(B) 是 B 的先验概率或边缘概率，这里称作标准化常量
        P(B|A)/P(B) 称作标准似然度

    ==> 贝叶斯法则又可表述为

    **后验概率 =(似然度先验概率)/ 标准化常量 = 标准似然度 * 先验概率**

- ## 贝叶斯方法

    + ### 基本思路

        假定**要估计的模型参数是服从一定分布的随机变量**，根据经验给出待估参数的**先验分布（也称为主观分布）**，关于这些先验分布的信息被称为**先验信息**；然后**根据这些先验信息，并与样本信息相结合，应用贝叶斯定理求出待估参数的后验分布**；再应用**损失函数**，得出后验分布的一些特征值，并把它们作为待估参数的估计量。

    + ### 与 经典估计方法 对比

        > 经典估计方法 : 完全根据**观测值**和**建立的模型**（被估计量和观测值之间的关系）对参数进行估计

    + ### 意义

        贝叶斯公式为利用搜集到的信息对原有判断进行修正提供了有效手段。(下文垃圾邮件分类案例后重新理解这句话的含义)

    > [更深入地理解贝叶斯定理](https://blog.csdn.net/yanghonker/article/details/51505068)

- ## 拼写纠错实例

    用户输入一个不在字典中的单词,猜测其真正想输入的单词

    即, P(猜测想输入的单词|实际输入的单词)

    + ### 抽象化

        <img src="https://latex.codecogs.com/gif.latex?P(h_{i}|D)=\frac{P(D|h_{i})P(h_{i})}{P(D)}" title="P(h_{i}|D)=\frac{P(D|h_{i})P(h_{i})}{P(D)}" />

            D : 实际输入的单词
            h_i : 不同的猜测值

            P(h_i) : 单词 h_i 的先验概率
            P(D|h_i) : 单词 h_i 错输成 D 的概率
            P(D) : 对于不同的具体猜测 h_i, P(D) 都是一样的,所以比较 P(h_i|D) 时,可以忽略这个常数

    + ### 化简

        <img src="https://latex.codecogs.com/gif.latex?P(h|D)\propto&space;P(D|h)*P(h)" title="P(h|D)\propto P(D|h)*P(h)" />

        > 即, 对于给定的观测数据, 一个猜测是好是坏, 取决于 **这个猜测本身独立的可能性大小(先验概率, Prior)** 和 **这个猜测生成观测到的数据的可能性大小**(eg: 对应 the 的词频概率; the 写成 tha 的概率).

    + ### 实例

        输入 tlp --> top or tip ?

        贝叶斯方法分别计算 <img src="https://latex.codecogs.com/gif.latex?P(tlp|top)*P(top)" title="P(tlp|top)*P(top)" /> 和 <img src="https://latex.codecogs.com/gif.latex?P(tlp|tip)*P(tip)" title="P(tlp|tip)*P(tip)" />

        > 不能作出决定性判断时,先验概率插手, top 出现程度高,想选 top

- ## 模型比较理论

    最大似然 MLE :最符合观测数据的最有优势, 即 <img src="https://latex.codecogs.com/gif.latex?max\text{&space;}P(D|h)" title="max\text{ }P(D|h)" />

    奥卡姆剃刀 Ockham's Razor : <img src="https://latex.codecogs.com/gif.latex?P(h)" title="P(h)" /> 较大的模型有较大优势

        Entities should not be multiplied unnecessarily（如无必要，勿增实体）
        越是高阶的多项式越不常见(eg: 回归模型分析,过拟合)

    掷一次硬币, 为正面, 此时猜测正面向上概率, MLE 方法令 P=1; Ockham's Razor 令 P=1/2

- ## 垃圾邮件过滤实例

    D(d1, ... , dn) 表示邮件, 由 N 个单词 di 组成.

    yes 表示垃圾邮件, no 表示正常.

    <img src="https://latex.codecogs.com/gif.latex?P(yes|D)=P(D|yes)*P(yes)|P(D)" title="P(yes|D)=P(D|yes)*P(yes)|P(D)" />

    <img src="https://latex.codecogs.com/gif.latex?P(no|D)=P(D|no)*P(no)|P(D)" title="P(no|D)=P(D|no)*P(no)|P(D)" />

        P(yes/no) 邮件库中 垃圾邮件/正常邮件 比例
        P(D|yes) 当前被分类为垃圾邮件的 D 由什么词组成
        P(D) 不影响结果, 去掉

    P(D|yes)=P(d1,d2,...,dn|yes), 表示垃圾邮件中出现跟当前这封邮件一模一样的的一封邮件的概率.

    规格太严格, 不必出现和垃圾邮件中一模一样的字词

    ==>

        P(d1,...,dn|yes) 扩展为 : P(d1|yes) * P(d2|d1,yes) *...

    假设 di 和 di-1 完全条件无关, 简化为

        P(d1|yes)*P(d2|yes)*...


    > 即**朴素贝叶斯理论**:特征之间独立, 互不影响

    则只要统计 di 这个单词在垃圾邮件中出现的频率即可.

- ## Supplement

    + ### 经典估计方法

        * #### 参数估计

            - 普通最小二乘估计 OLS - Ordinary least squares

            - 最大似然 ML - Maximum likelihood

            - 矩估计 MM - Moment method

                由辛钦大数定律知，简单随机样本的原点矩依概率收敛到相应的总体原点矩，这就启发我们想到用样本矩替换总体矩，进而找出未知参数的估计，基于这种思想求估计量的方法称为矩法，用矩法求得的估计称为矩法估计，简称矩估计。

                利用样本矩来估计总体中相应的参数。最简单的矩估计法是用一阶样本原点矩估计总体期望，而用二阶样本中心矩估计总体方差。

        * #### 非参数模型估计

            - 核估计

            - 局部多项式估计

            - k 近邻
