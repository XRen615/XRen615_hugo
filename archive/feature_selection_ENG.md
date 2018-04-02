+++
author = "X.Ren"
comments = true
date = "2016-03-06T19:19:22+01:00"
draft = false
image = ""
menu = ""
share = true
slug = "Feature_reducing_ENG"
tags = ["Data science","Feature Engineering"]
title = "Feature reducing_ENG"
+++
### Feature reducing 

#### Background  

During my thesis study on the database of Roosendal passive houses, I was expected to find the dominant driving factors that influence the occupant behaviors, which means it is somehow a feature-selection task. At that time I selected a tree-based methodology (random forest) since it is easy to use with little tune needed (so-called *'out of shelf'*) and,  with inherent output attribute *'_feature_importance'*（should be based on the error reduction in different sub-sampling). Then I discussed it with Hussain Kazmi, a data scientist from Enervalis, found it is more or less controversial to implement RF in practice like this（[here](https://www.quora.com/When-would-one-use-Random-Forests-over-Gradient-Boosted-Machines-GBMs)). Also, during my meeting with Prof. Gert and msc. Christos from TU/e, they suggested it would be better if there is a more 'systematic' evaluation index to support the algorithm methodology I used. So this week I read some scientific references talking about systematic feature selection techniques in different level and summarized them into this note.

#### Note starts  

While dealing with datasets, especially data records from fields like signal processing or bioinformatics etc., it is common that the no. of features for one observation is large but the sample size is not really comparable. On one hand, many features that is not that 'informative' would influence the accuracy of the machine learning algorithm, on the other hand, too many features usually come with overfitting. Thus, feature reducing is a common step in practical machine learning studies.  

Currently there are mainly 2 categories of feature reducing methods: feature selection and feature transformation. Feature transformation tries to transform several features into a new feature and contains techniques such as principle component analysis (PCA), but it is not suitable for my case since I just want to choose rather than generate new features. So in this note I will concentrate on feature selection techniques and its realization in Python scikit-learn environment.  

In general, there are 2 types of feature selection techniques: filter methods and wrapper methods. Filter methods select the features based on statistical variants and do not depend on any specific learning algorithm, thus it is light and fast, suitable for quick pre-processing. While wrapper methods are the selection techniques with specific learning algorithms included (logistic regression, trees etc.).

In Python scikit-learn environment, feature selection is supported by the package *sklearn.feature_selection*, with different levels of techniques available.  From low level to higher ones:  
   
#### 1. Variance Threshold  

	from sklearn.feature_selection import VarianceThreshold  

> VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features, i.e. features that have the same value in all samples.  

#### 2. Univariate feature selection    

> Univariate feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator.  

E.g. [Chi-squared test](https://segmentfault.com/a/1190000003719712) is a widely-used technique to test to what extend two variables are correlated, then the p-value of chi2 test could be used for selection:   
 
	X_new = SelectKBest(chi2, k=2).fit_transform(X, y)  

#### 3. Recursive feature elimination  

> Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and weights are assigned to each one of them. Then, features whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

#### 4. Feature selection using SelectFromModel (Wrapper techniques)  

> SelectFromModel is a meta-transformer that can be used along with any estimator that has a coef_ or feature_importances_ attribute after fitting. The features are considered unimportant and removed, if the corresponding coef_ or feature_importances_ values are below the provided threshold parameter. Apart from specifying the threshold numerically, there are build-in heuristics for finding a threshold using a string argument. Available heuristics are “mean”, “median” and float multiples of these like “0.1*mean”.  

##### 4.1 L1-based feature selection  

> Linear models penalized with the L1 norm have sparse solutions: many of their estimated coefficients are zero. When the goal is to reduce the dimensionality of the data to use with another classifier, they can be used along with feature_selection.SelectFromModel to select the non-zero coefficients. In particular, sparse estimators useful for this purpose are the **linear_model.Lasso for regression**, and of **linear_model.LogisticRegression and svm.LinearSVC for classification**.  

###### Penalty control:  

> With SVMs and logistic-regression, the parameter C controls the sparsity: the smaller C the fewer features selected. With Lasso, the higher the alpha parameter, the fewer features selected.  
> There is no general rule to select an alpha parameter for recovery of non-zero coefficients. It can by set by cross-validation (LassoCV or LassoLarsCV), though this may lead to under-penalized models: including a small number of non-relevant variables is not detrimental to prediction score. BIC (LassoLarsIC) tends, on the opposite, to set high values of alpha.  

###### Note: LASSO  

[LASSO (least absolute shrinkage and selection operator)](https://en.wikipedia.org/wiki/Lasso_(statistics\)), a widely-used regularization/feature selection technique，with objective function like：  

<div  align="center">    
<img src="http://i593.photobucket.com/albums/tt11/RickRen/%202016-03-07%2012.29.14_zpsvxgzzhky.png" width = "300" height = "40" alt="Lasso" align=center />  
</div>  

In laymen's terms，with the constraint ∑|β|≤t, LASSO tries to find the estimation of regression coefficient combination that minimize |y−Xβ|^2. In practice, when the constraint is equality, the regression equation could be solved by Lagrange multiplier. In this case, with inequality constraint, the idea is like from 0, repeatedly increase the value of t then the coefficient estimation could be made for every t with the help of computer program. Then this series of estimation of coefficients is called 'LASSO estimation'  

Tibshirani pointed out that，while t is small enough，LASSO tends to estimate certain coefficients as 0，which could serve as selection criterion. The no. of feature selected will increase while t increasing and, at a certain point all features will be selected: which is commonly-called 'least-squared estimation'. During this process, the variance of model is increasing while the bias is decreasing, i.e. it is some kind of the 'trade-off' between bias and overfitting.

[Elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization), which looks very similar to LASSO, is considered outperform in highly inter-correlated datasets.  

##### 4.3. Tree-based feature selection  

> Tree-based estimators (see the sklearn.tree module and forest of trees in the sklearn.ensemble module) can be used to compute feature importances, which in turn can be used to discard irrelevant features (when coupled with the sklearn.feature_selection.SelectFromModel meta-transformer).  
 

#### Other references： 

[Scikit-learn feature selection documentation](http://scikit-learn.org/stable/modules/feature_selection.html)  

[Modified LARS and LASSO](http://cos.name/2011/04/modified-lars-and-lasso/)