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