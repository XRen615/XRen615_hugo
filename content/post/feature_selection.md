+++
date = "2016-03-06T19:19:22+01:00"
author = "X.Ren"
comments = true
draft = false
image = ""
menu = ""
share = true
slug = "feature_reducing"
tags = ["data science","Feature Engineering"]
title = "特征选择"

+++
### Feature reducing  

#### 缘起

处理Rosendal passive house的数据库时，我想找到影响某些用户行为（occupant behavior）的主导因素，因此需要进行某种形式的特征选择工作。在第一版的代码中，我选择了方便的tree-based method (random forest)，易于调校（不用调校，手动滑稽）并且自带了feature importance输出选项（应该是建立在不同sub-sampling的error reduction结果上）。后来与Enervalis的数据工程师Hussain Kazmi交谈的过程中，了解到了如此实际使用中RF的争议性（[这里](https://www.quora.com/When-would-one-use-Random-Forests-over-Gradient-Boosted-Machines-GBMs))，所以想系统的了解一下不同层级上的特征选择技术。  

#### 笔记开始  

在处理数据集时，尤其是在signal processing和bioinformatics等方向，容易出现单个observation feature多而sample size不足的情况。一方面许多无用的数据会干扰学习算法的准确性，另一方面容易造成过拟合。所以引进feature reducing的方法是实际操作中常见的步骤。  

目前feature reducing包括feature selection和feature transformation两个大方向，后者包括PCA等降维方法，由于在我的case里我并不想transform出新的feature而仅仅是做选择，所以另行讨论。本文将主要总结这几天看过的feature selection手段以及其在scikit-learn环境下的实现。  

总的来说，feature selection包含filter methods和wrapper methods两个类别：前者不需要结合特定的算法，简单快速，常用于预处理；后者是结合算法的特征选择，常用于学习阶段。在scikit-learn环境中，特征选择拥有独立的包sklearn.feature_selection, 包含了在preprocess或者学习阶段等不同层级的特征选择算法。以下方法的层级由低到高  
  
#### 1. Variance Treshhold（方差阈）    

	from sklearn.feature_selection import VarianceThreshold  

> VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features, i.e. features that have the same value in all samples.  

#### 2. Univariate feature selection (单变量特征选择)  

> Univariate feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator.  

比如[卡方检验](https://segmentfault.com/a/1190000003719712)是检测两变量是否相关的常用手段，那么就可以利用chi2的p-value进行选择：
  
	X_new = SelectKBest(chi2, k=2).fit_transform(X, y)  

#### 3. Recursive feature elimination （递归消除）  

> Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and weights are assigned to each one of them. Then, features whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

#### 4. Feature selection using SelectFromModel (Wrapper方法)  

> SelectFromModel is a meta-transformer that can be used along with any estimator that has a coef_ or feature_importances_ attribute after fitting. The features are considered unimportant and removed, if the corresponding coef_ or feature_importances_ values are below the provided threshold parameter. Apart from specifying the threshold numerically, there are build-in heuristics for finding a threshold using a string argument. Available heuristics are “mean”, “median” and float multiples of these like “0.1*mean”.  

##### 4.1 L1-based feature selection  

> Linear models penalized with the L1 norm have sparse solutions: many of their estimated coefficients are zero. When the goal is to reduce the dimensionality of the data to use with another classifier, they can be used along with feature_selection.SelectFromModel to select the non-zero coefficients. In particular, sparse estimators useful for this purpose are the **linear_model.Lasso for regression**, and of **linear_model.LogisticRegression and svm.LinearSVC for classification**.  

##### Penalty control

> With SVMs and logistic-regression, the parameter C controls the sparsity: the smaller C the fewer features selected. With Lasso, the higher the alpha parameter, the fewer features selected.  
> There is no general rule to select an alpha parameter for recovery of non-zero coefficients. It can by set by cross-validation (LassoCV or LassoLarsCV), though this may lead to under-penalized models: including a small number of non-relevant variables is not detrimental to prediction score. BIC (LassoLarsIC) tends, on the opposite, to set high values of alpha.  

##### Note: LASSO  
[LASSO (least absolute shrinkage and selection operator)](https://en.wikipedia.org/wiki/Lasso_(statistics\)) 是一种常用的regularization/feature selection方法，其目标函数为：  

<div  align="center">    
<img src="http://i593.photobucket.com/albums/tt11/RickRen/%202016-03-07%2012.29.14_zpsvxgzzhky.png" width = "300" height = "40" alt="Lasso" align=center />  
</div>  

简单来说，是在限制了∑|β|≤t, 的情况下，求使得残差平方和|y−Xβ|^2 达到最小的回归系数的估值。当解限制条件为等号时，回归方程可用lagrange乘子法求解。对于这种限制条件是不等号的情况，则可以利用计算机程序，对t从0开始，不断慢慢增加它的值，然后对每个t，求限制条件为等号时候的回归系数的估计，从而可以以t的值为横轴，作出一系列的回归系数向量的估计值，这一系列的回归系数的估计值就是LASSO estimation。  

LASSO estimate具有shrinkage和selection两种功能。[岭估计 (ridge regression)](http://blog.csdn.net/google19890102/article/details/27228279)会有shrinkage的功效，LASSO也同样。关于selection，Tibshirani提出，当t值小到一定程度的时候，LASSO estimate会使得某些回归系数的估值是0，这起到了变量选择的作用。当t不断增大时，选入回归模型的变量会逐渐增多，当t增大到某个值时，所有变量都入选了回归模型，这个时候得到的回归模型的系数是通常意义下的最小二乘估计。在这个过程中，variance逐渐增大，bias逐渐减小。从这个角度上来看，LASSO也可以看做是一种逐步回归的过程。  

与LASSO极为相似的[弹性网络 (elastic net)](https://en.wikipedia.org/wiki/Elastic_net_regularization)则被认为在features高度相关的数据集中具有更为优良的表现。 

##### 4.3. Tree-based feature selection  

> Tree-based estimators (see the sklearn.tree module and forest of trees in the sklearn.ensemble module) can be used to compute feature importances, which in turn can be used to discard irrelevant features (when coupled with the sklearn.feature_selection.SelectFromModel meta-transformer).  
 

#### 其他参考资料： 

[scikit-learn feature selection documentation](http://scikit-learn.org/stable/modules/feature_selection.html)  

[修正的LARS算法和lasso](http://cos.name/2011/04/modified-lars-and-lasso/)