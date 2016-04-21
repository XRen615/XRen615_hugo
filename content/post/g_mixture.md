+++
author = "X.REN"
comments = true
date = "2016-04-16T15:27:07+02:00"
draft = false
image = ""
menu = ""
share = true
slug = "gmixture"
tags = ["Gaussian Mixture Models", "Probabilistic classification"]
title = "Gaussian Mixture Models"

+++

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. Instead of having an output as cluster no. that a certain sample belongs to, Gaussian mixture model can additionally give an probability of this kind of clustering.  

In practice, expectation-maximization (EM) algorithm is often used for fitting mixture-of-Gaussian models.  

For better understand the theory of GMM this [link](http://blog.pluskid.org/?p=39) could be followed.  

For the algorithm realization in Python, follow the sklearn official documentary [here](http://scikit-learn.org/stable/modules/mixture.html). One remark: there are 4 different kinds of [covariance matrices](http://pinkyjie.com/2010/08/31/covariance/) that are supported by sklearn (diagonal, spherical, tied and full covariance matrices). Full covariance could get the best results however would significantly increase the calculation load in the same time.  

The **selection of best no. of components** could be done by calculating BIC（Bayesian information criterion）score. Schwarz's Bayesian Information Criterion (BIC) is a model selection tool. If a model is estimated on a particular data set (training set), BIC score gives an estimate of the model performance on a new, fresh data set (testing set). BIC is given by the formula: 

	BIC = -2 * loglikelihood + d * log(N), 


where N is the sample size of the training set and d is the total number of parameters. For better understanding of likelihood, refer to [link](https://sswater.wordpress.com/2012/06/04/似然函数最大似然估计/). The lower BIC score signals a better model. 

To use BIC for model selection, we simply chose the model giving smallest BIC over the whole set of candidates. BIC attempts to mitigate the risk of over-fitting by introducing the penalty term d * log(N), which grows with the number of parameters. This allows to filter out unnecessarily complicated models, which have too many parameters to be estimated accurately on a given data set of size N. BIC has preference for simpler models compared to Akaike Information Criterion (AIC). 