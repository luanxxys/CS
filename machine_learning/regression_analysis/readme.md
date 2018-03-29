# Regression algorithm

- ## Linear regression

    + 预测模型

        <img src="https://latex.codecogs.com/gif.latex?y^{(i)}=\theta&space;^{T}x^{(i)}&plus;\varepsilon&space;^{(i)}" title="y^{(i)}=\theta ^{T}x^{(i)}+\varepsilon ^{(i)}" />

        > 误差<img src="https://latex.codecogs.com/gif.latex?\varepsilon&space;^{(i)}" title="\varepsilon ^{(i)}" />是独立并且具有相同的分布,通常认为服从均值为 0 方差为<img src="https://latex.codecogs.com/gif.latex?\sigma&space;^{2}" title="\sigma ^{2}" />的高斯分布

    + 误差分布函数

        <img src="https://latex.codecogs.com/gif.latex?p(\varepsilon&space;^{(i)})=\frac{1}{\sqrt{2\pi&space;}\sigma&space;}exp(-\frac{(\varepsilon&space;^{(i)})&space;^{2})}{2\sigma&space;^{2}})" title="p(\varepsilon ^{(i)})=\frac{1}{\sqrt{2\pi }\sigma }exp(-\frac{(\varepsilon ^{(i)}) ^{2})}{2\sigma ^{2}})" />

        将优化问题转化成：找到参数<img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />,求得最小的误差<img src="https://latex.codecogs.com/gif.latex?\varepsilon }" title="\varepsilon }" />,使预测值和真实值相等的概率最大

        <img src="https://latex.codecogs.com/gif.latex?p(y&space;^{(i)}|x&space;^{(i)};\theta&space;)=\frac{1}{\sqrt{2\pi&space;}\sigma&space;}exp(-\frac{(y^{(i)}-\theta&space;^{T}x^{(i)})&space;^{2})}{2\sigma&space;^{2}})" title="p(y ^{(i)}|x ^{(i)};\theta )=\frac{1}{\sqrt{2\pi }\sigma }exp(-\frac{(y^{(i)}-\theta ^{T}x^{(i)}) ^{2})}{2\sigma ^{2}})" />

    + ### 求解

        1. 建立似然函数(针对所有样本)

            <img src="https://latex.codecogs.com/gif.latex?L(\theta&space;)=\prod_{i=1}^{m}p(y&space;^{(i)}|x&space;^{(i)};\theta&space;)" title="L(\theta )=\prod_{i=1}^{m}p(y ^{(i)}|x ^{(i)};\theta )" />

            <img src="https://latex.codecogs.com/gif.latex?=\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi&space;}\sigma&space;}exp(-\frac{(y^{(i)}-\theta&space;^{T}x^{(i)})&space;^{2})}{2\sigma&space;^{2}})" title="=\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi }\sigma }exp(-\frac{(y^{(i)}-\theta ^{T}x^{(i)}) ^{2})}{2\sigma ^{2}})" />

        1. 取 log，乘法操作 --> 加法操作

            <img src="https://latex.codecogs.com/gif.latex?l(\theta&space;)=logL(\theta&space;)" title="l(\theta )=logL(\theta )" />

            <img src="https://latex.codecogs.com/gif.latex?=log\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi&space;}\sigma&space;}exp(-\frac{(y^{(i)}-\theta&space;^{T}x^{(i)})&space;^{2})}{2\sigma&space;^{2}})" title="=log\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi }\sigma }exp(-\frac{(y^{(i)}-\theta ^{T}x^{(i)}) ^{2})}{2\sigma ^{2}})" />

            <img src="https://latex.codecogs.com/gif.latex?=\sum_{i=1}^{m}log\frac{1}{\sqrt{2\pi&space;}\sigma&space;}exp(-\frac{(y^{(i)}-\theta&space;^{T}x^{(i)})&space;^{2})}{2\sigma&space;^{2}})" title="=\sum_{i=1}^{m}log\frac{1}{\sqrt{2\pi }\sigma }exp(-\frac{(y^{(i)}-\theta ^{T}x^{(i)}) ^{2})}{2\sigma ^{2}})" />

            <img src="https://latex.codecogs.com/gif.latex?=m&space;log\frac{1}{\sqrt{2\pi&space;}\sigma&space;}-\frac{1}{\sigma&space;^{2}}·\frac{1}{2}\sum_{i=1}^{m}(y^{(i)}-\theta&space;^{T}x^{(i)})^{2}" title="=m log\frac{1}{\sqrt{2\pi }\sigma }-\frac{1}{\sigma ^{2}}·\frac{1}{2}\sum_{i=1}^{m}(y^{(i)}-\theta ^{T}x^{(i)})^{2}" />

        1. 减号后面大于 0,使整体最大,则需要使后半部分整体最小.目标函数变为

            <img src="https://latex.codecogs.com/gif.latex?J(\theta&space;)=\frac{1}{2}\sum_{i=1}^{m}(h_{\theta&space;}(x^{(i)})-\theta&space;^{T}x^{(i)})^{2}" title="J(\theta )=\frac{1}{2}\sum_{i=1}^{m}(h_{\theta }(x^{(i)})-\theta ^{T}x^{(i)})^{2}" />

            > <img src="https://latex.codecogs.com/gif.latex?h_{\theta&space;}(x^{(i)})" title="h_{\theta }(x^{(i)})" />表示预测值
            >
            > 上式即说明了目标函数为何是 均方差(MSE)函数

        1. 借助矩阵形式求解

            <img src="https://latex.codecogs.com/gif.latex?J(\theta&space;)=\frac{1}{2}\sum_{i=1}^{m}(h_{\theta&space;}(x^{(i)})-\theta&space;^{T}x^{(i)})^{2}=\frac{1}{2}(X\theta&space;-y)^{T}(X\theta&space;-y)" title="J(\theta )=\frac{1}{2}\sum_{i=1}^{m}(h_{\theta }(x^{(i)})-\theta ^{T}x^{(i)})^{2}=\frac{1}{2}(X\theta -y)^{T}(X\theta -y)" />

            对<img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />求导

            <img src="https://latex.codecogs.com/gif.latex?\bigtriangledown&space;_{\theta&space;}J(\theta&space;)=\bigtriangledown&space;_{\theta&space;}(\frac{1}{2}(X\theta&space;-y)^{T}(X\theta&space;-y))=\bigtriangledown&space;_{\theta&space;}(\frac{1}{2}(\theta^{T}X^{T}&space;-y^{T})(X\theta&space;-y))" title="\bigtriangledown _{\theta }J(\theta )=\bigtriangledown _{\theta }(\frac{1}{2}(X\theta -y)^{T}(X\theta -y))=\bigtriangledown _{\theta }(\frac{1}{2}(\theta^{T}X^{T} -y^{T})(X\theta -y))" />

            <img src="https://latex.codecogs.com/gif.latex?=\bigtriangledown&space;_{\theta&space;}(\frac{1}{2}(\theta^{T}X^{T}X\theta&space;-\theta^{T}X^{T}y^{T}-y^{T}X\theta&plus;y^{T}y)" title="=\bigtriangledown _{\theta }(\frac{1}{2}(\theta^{T}X^{T}X\theta -\theta^{T}X^{T}y^{T}-y^{T}X\theta+y^{T}y)" />

            <img src="https://latex.codecogs.com/gif.latex?=\frac{1}{2}(2X^{T}X\theta-X^{T}y-(y^{T}X)^{T})=X^{T}X\theta-X^{T}y" title="=\frac{1}{2}(2X^{T}X\theta-X^{T}y-(y^{T}X)^{T})=X^{T}X\theta-X^{T}y" />

            > 矩阵转置、求导

            <img src="https://latex.codecogs.com/gif.latex?\theta&space;=(X^{T}X)^{-1}X^{T}y" title="\theta =(X^{T}X)^{-1}X^{T}y" />

- ## Logistics regression

    > 用于分类

    + ### Sigmoid function

        * 数学表达式

            <img src="https://latex.codecogs.com/gif.latex?g(z)=\frac{1}{1&plus;e^{-z}}" title="g(z)=\frac{1}{1+e^{-z}}" />

        * 求导

            <img src="https://latex.codecogs.com/gif.latex?g^{'}(x)=(\frac{1}{1&plus;e^{-x}})^{'}=...=(\frac{1}{1&plus;e^{-x}})·(1-\frac{1}{1&plus;e^{-x}})" title="g^{'}(x)=(\frac{1}{1+e^{-x}})^{'}=...=(\frac{1}{1+e^{-x}})·(1-\frac{1}{1+e^{-x}})" />

        * ==>

            <img src="https://latex.codecogs.com/gif.latex?h_{\theta&space;}(x)=g(\theta&space;^{T}x)=\frac{1}{1&plus;e^{-\theta&space;^{T}x}}" title="h_{\theta }(x)=g(\theta ^{T}x)=\frac{1}{1+e^{-\theta ^{T}x}}" />

- ## 梯度下降

    > 只考虑一个输入的情况

    <img src="https://latex.codecogs.com/gif.latex?h_{\theta&space;}(x)=\theta&space;_{1}&plus;\theta&space;_{0}" title="h_{\theta }(x)=\theta _{1}+\theta _{0}" />

    <img src="https://latex.codecogs.com/gif.latex?J(\theta&space;_{0},\theta&space;_{1})=\frac{1}{2m}{\sum}_{i=1}^{m}(h_{\theta&space;}(x_{i})-y_{i})^{2}" title="J(\theta _{0},\theta _{1})=\frac{1}{2m}{\sum}_{i=1}^{m}(h_{\theta }(x_{i})-y_{i})^{2}" />

    > 相比之前推得的损失函数<img src="https://latex.codecogs.com/gif.latex?J(\theta)" title="J(\theta)" />多乘了<img src="https://latex.codecogs.com/gif.latex?\frac{1}{m}" title="\frac{1}{m}" />项 ==> 求全部 m 个样本共同的损失函数，将样本因素排除在外，更能说明损失函数和<img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />之间的关系

    <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J(\theta&space;_{0},\theta&space;_{1})}{\partial&space;\theta&space;_{0}}=\frac{1}{m}{\sum}_{i=1}^{m}(h_{\theta&space;}(x_{i})-y_{i})" title="\frac{\partial J(\theta _{0},\theta _{1})}{\partial \theta _{0}}=\frac{1}{m}{\sum}_{i=1}^{m}(h_{\theta }(x_{i})-y_{i})" />

    <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J(\theta&space;_{0},\theta&space;_{1})}{\partial&space;\theta&space;_{0}}=\frac{1}{m}{\sum}_{i=1}^{m}(h_{\theta&space;}(x_{i})-y_{i})" title="\frac{\partial J(\theta _{0},\theta _{1})}{\partial \theta _{0}}=\frac{1}{m}{\sum}_{i=1}^{m}(h_{\theta }(x_{i})-y_{i})" />

    <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J(\theta&space;_{0},\theta&space;_{1})}{\partial&space;\theta&space;_{1}}=\frac{1}{m}{\sum}_{i=1}^{m}(h_{\theta&space;}(x_{i})-y_{i})*x_{i}" title="\frac{\partial J(\theta&space;_{0},\theta _{1})}{\partial \theta _{1}}=\frac{1}{m}{\sum}_{i=1}^{m}(h_{\theta }(x_{i})-y_{i})*x_{i}" />

    更新

    <img src="https://latex.codecogs.com/gif.latex?\theta&space;_{1}:=\theta&space;_{1}-\alpha&space;*\frac{\partial&space;J(\theta&space;_{0},\theta&space;_{1})}{\theta&space;_{0}}" title="\theta _{1}:=\theta _{1}-\alpha *\frac{\partial J(\theta _{0},\theta _{1})}{\theta _{0}}" />

    <img src="https://latex.codecogs.com/gif.latex?\theta&space;_{0}:=\theta&space;_{0}-\alpha&space;*\frac{\partial&space;J(\theta&space;_{0},\theta&space;_{1})}{\theta&space;_{1}}" title="\theta _{0}:=\theta _{0}-\alpha *\frac{\partial J(\theta _{0},\theta _{1})}{\theta _{1}}" />

### 插曲

线性回归中假设的,两个模型之间符合的高斯分布的误差,最开始想当然认为了求得这个最小误差后,原始数据集减之就能求得预测模型。

钻了牛角尖，无脑认为只从误差这一个对象就可以找最大似然模型，只有一个原始数据集，没办法用最大似然理念。后续的步骤是对模型参数建模，求最大似然。

而且,单把误差拿出来没有意义。误差的含义是模型生成的某具体数据样本和原始数据中相对应的样本之间，差值的波动符合高斯分布，而不是和一个莫须有模型的整体上的差值。

但这么一个简单的思维漏洞，先是和 高远 扯了半天，后来又加入 纯宇，折腾到一两点。

回想起小学三四年级，晚自习上正和旁边人玩闹，数学老师冲进教室开始拧我耳朵，拿着我的试卷，上面我的计算 ** 30*40=12 **.

这种毛病一直都有，一直到大学还挺严重。研究生入学以来感触还不大，但事情一旦显现，这种思维盲区还是这么“惹人疼”。一旦不加思考认定之后，短时间内很难推翻自己的错误认知，即使和别人讨论中，也能“思维迸射”提出很多辩解，并试图用自己的观点去同化对方。

元认知管理？总之，这种情形，无脑为自己的理论辩护是好的，但等到讨论中，只剩下始终重复已经陈述的理由再无新意时，应当停停了，开始反思。
