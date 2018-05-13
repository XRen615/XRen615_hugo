+++
date = "2017-01-05T19:19:22+01:00"
author = "X.Ren"
comments = true
draft = false
image = ""
menu = ""
share = true
slug = "feature_reducing"
tags = ["data science","Feature selection"]
title = "A Simple Summary on Feature Selection"

+++

#### 介绍

数据工程项目往往严格遵循着riro (rubbish in, rubbish out) 的原则，所以我们经常说数据预处理是数据工程师或者数据科学家80%的工作，它保证了数据原材料的质量。而特征工程又至少占据了数据预处理的半壁江山，在实际的数据工程工作中，无论是出于解释数据或是防止过拟合的目的，特征选择都是很常见的工作。如何从成百上千个特征中发现其中哪些对结果最具影响，进而利用它们构建可靠的机器学习算法是特征选择工作的中心内容。在多次反复的工作后，结合书本，kaggle等线上资源以及与其他数据工程师的讨论，我决定写一篇简明的总结梳理特征选择工作的常见方法以及python实现。

总的来说，特征选择可以走两条路：

- 特征过滤（Filter methods）: 不需要结合特定的算法，简单快速，常用于预处理

- 包装筛选（Wrapper methods）: 将特征选择包装在某个算法内，常用于学习阶段

在scikit-learn环境中，特征选择拥有独立的包sklearn.feature_selection, 包含了在预处理和学习阶段不同层级的特征选择算法。  

***  

#### A. 特征过滤（Filter methods）
  
**(1) 方差阈（Variance Treshhold）**  

最为简单的特征选择方式之一，去除掉所有方差小于设定值的特征。  

在sklearn中实现：

	from sklearn.feature_selection import VarianceThreshold  

> VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features, i.e. features that have the same value in all samples.  
  


**(2) 单变量特征选择 (Univariate feature selection)**  

基于单变量假设检验的特征选择，比如卡方检验（[这里有一篇很好的博文用于回顾](https://segmentfault.com/a/1190000003719712)）是检测两变量是否相关的常用手段，那么就可以很自然的利用chi-square值来做降维，保留相关程度大的变量。

> Univariate feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator.  


  
	X_new = SelectKBest(chi2, k=2).fit_transform(X, y)  


#### B. 包装筛选（Wrapper methods）  

包装筛选往往利用一些在训练过程中可以计算各个特征对应权重的算法来达到选择特征的目的。在sklearn中有一个专门的模块 *SelectFromModel* 来帮助我们实现这个过程。

> SelectFromModel is a meta-transformer that can be used along with any estimator that has a coef_ or feature_importances_ attribute after fitting. The features are considered unimportant and removed, if the corresponding coef_ or feature_importances_ values are below the provided threshold parameter. Apart from specifying the threshold numerically, there are build-in heuristics for finding a threshold using a string argument. Available heuristics are “mean”, “median” and float multiples of these like “0.1*mean”.  

**（1）利用Lasso进行特征选择**  

在介绍利用Lasso进行特征选择之前，简要介绍一下什么是Lasso：  

对于一个线性回归问题  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/equation.png" align=center />  
</div>  

基本的任务是估计参数，使得

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/equation-3.png" align=center />  
</div>  

最小，这就是经典的 Ordinary Linear Square (OLS) 问题。  

但在实际的工作中，仅仅使用OLS进行回归计算很容易造成过拟合，噪声得到了过分的关注，训练数据的微小差异可能带来巨大的模型差异（主要是样本的共线性容易使矩阵成为对扰动敏感的病态阵，从而造成回归系数解析解的不稳定，要更详细的探究可以参考[这里](https://www.zhihu.com/question/38121173))。  

为了矫正过拟合，我们常使用带有正则项的cost function，其中使用L1正则的表达式则为Lasso方法：  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/equation-4.png" align=center />  
</div> 

Lasso方法下解出的参数常常具有稀疏的特征，即很多特征对应的参数会为零，这就使得特征选择成为可能：我们可以训练一个Lasso模型，然后将系数为零的特征去除。  

在实际的工作中，Lasso的参数lambda越大，参数的解越稀疏，选出的特征越少。那么如何确定使用多大的lambda？一个比较稳妥地方案是对于一系列lambda，用交叉验证计算模型的rmse，然后选择rmse的极小值点 (Kaggle上有一个很好的[例子](https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models))。

> Linear models penalized with the L1 norm have sparse solutions: many of their estimated coefficients are zero. When the goal is to reduce the dimensionality of the data to use with another classifier, they can be used along with feature_selection.SelectFromModel to select the non-zero coefficients. With Lasso, the higher the alpha parameter, the fewer features selected.  

在sk-learn中的实现参看[这里](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)。

**（2）基于决策树的特征选择**  

利用决策树中深度较浅的节点对应的特征提供信息较多（可以直观的理解为这个特征将更多的样本区分开）这一特性，许多基于决策树的算法，如[随机森林](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)也可以在结果中直接给出feature_importances属性。其主要思想是训练一系列不同的决策树模型，在每一棵树中使用特征集的某一个随机的子集（使用bootstrap等方法抽样），最后统计每个特征出现的次数，深度，分离的样本量以及模型的准确率等给出特征的权重值。设定一个阈值，我们便可以使用这类基于决策树的算法进行特征选择。

> Tree-based estimators (see the sklearn.tree module and forest of trees in the sklearn.ensemble module) can be used to compute feature importances, which in turn can be used to discard irrelevant features (when coupled with the sklearn.feature_selection.SelectFromModel meta-transformer).  

在sk-learn中的实现参看[这里](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)。

***  

#### 小结

这篇短文简明的介绍了部分常用的特征处理方法，应该提出的是，除了feature selection，feature transformation，包括PCA等降维方法也可以达到减少特征数量，抑制过拟合的目的。

#### 其他参考资料： 

[scikit-learn feature selection documentation](http://scikit-learn.org/stable/modules/feature_selection.html)  

[修正的LARS算法和lasso](http://cos.name/2011/04/modified-lars-and-lasso/)