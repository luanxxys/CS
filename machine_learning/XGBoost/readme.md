# XGBoost- eXtreme Gradient Boosting

- ## reference

    + 中文文档地址：http://xgboost.apachecn.org/cn/latest/
    + 英文文档地址：http://xgboost.apachecn.org/en/latest/
    + 中文文档 GitHub 地址：https://github.com/apachecn/xgboost-doc-zh
    + [Introduction to Boosted Trees-Tianqi Chen](BoostedTree.pdf)
    + [XGBoost: A Scalable Tree Boosting System](XGBoost.pdf)

- ## theory

    核心:决策树的集成,加树则效果变好 --> 提升方法

    样本* 权值

    目标函数:均方误差

    最优函数解 - 期望 --> 集成算法表示

    xgboost 损失函数:自身期望误差+正则化函数(惩罚项)

    泰勒展开近似目标

    样本遍历 --> 对叶子节点遍历,新目标函数

    带回到原目标函数

    库 - 特征重要程度

