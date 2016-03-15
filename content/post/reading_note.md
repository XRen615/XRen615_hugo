+++
author = "X.Ren"
comments = true
date = "2016-03-13T20:12:06+01:00"
draft = false
image = ""
menu = ""
share = true
slug = ""
tags = ["Machine learning", "reading notes"]
title = "Reading Note: Building machine learning systems with python"

+++

( *I would like to take this oppurtunity to somehow review and systemize my knowledge on machine learning system.* )

“This book will give you a broad overview of what types of learning algorithms are currently most used in the diverse fields of machine learning, and where to watch out when applying them.”

typical workflow:  

- Reading in the data and cleaning it  
- Exploring and understanding the input data  (feature engineering: a simple algorithm with refined data generally outperforms a very sophisticated algorithm with raw data!)  
- Analyzing how best to present the data to the learning algorithm  
- Choosing the right model and learning algorithm  
- Measuring the performance correctly  

We see that only the fourth point is dealing with the fancy algorithms. Nevertheless, we hope that this book will convince you that the other four tasks are not simply chores, but can be equally exciting. Our hope is that by the end of the book, you will have truly fallen in love with data instead of learning algorithms.  

only blog we want to highlight right here (more in the Appendix) is http://blog.kaggle.com  

Python is an **interpreted language** (a highly optimized one, though) that is slow for many numerically heavy algorithms compared to C or FORTRAN. However in Python, it is very easy to **off-load** number crunching tasks to the lower layer in the form of **C or FORTRAN extensions**. And that is exactly what NumPy and SciPy do. In this tandem, **NumPy** provides the support of highly optimized multidimensional arrays, which are the basic data structure of most state-of-the-art algorithms. **SciPy** uses those arrays to provide a set of fast numerical recipes. Finally, **matplotlib** is probably the most convenient and feature-rich library to plot high-quality graphs using Python.  

##### Learning NumPy  

As we do not want to **pollute our namespace**, we certainly should **not** use the following code:  

	from numpy import *  
	
Because, for instance, numpy.array will potentially shadow the array package that is included in standard Python. Instead, we will use the following convenient shortcut:  

	import numpy as np  
	
Numpy **avoids copies** wherever possible, whenever you need a true copy, you can always perform:


	c = a.reshape((3,2)).copy()”

Another big advantage of NumPy arrays is that the **operations** are propagated to the **individual elements**.  

**Indexing**  

- allows you to use arrays themselves as indices by performing:  

		a[np.array([2,3,4])]  

- conditions are also propagated to individual elements  
 
		a[a>4]  
		
- clip function for trim the outliers, clipping the values at both ends of an interval with one function call  

		a.clip(0,4)  
		array([0, 1, 4, 3, 4, 4])”

- Handling nonexisting values  

		c = np.array([1, 2, np.NAN, 3, 4]) >>> c
		array([  1.,   2.,  nan,   3.,   4.])
		np.isnan(c)
		array([False, False,  True, False, False], dtype=bool)
		c[~np.isnan(c)]
		array([ 1.,  2.,  3.,  4.])
		np.mean(c[~np.isnan(c)])
		2.5

- Comparing the runtime: in every algorithm we are about to implement, we should always look how we can move loops over individual elements from Python to some of the highly optimized NumPy or SciPy extension functions. e.g. use a.dot(a) to calculate square sum instead of sum(a*a).  

##### Learning SciPy  

On top of the efficient data structures of NumPy, SciPy offers a magnitude of **algorithms** working on those arrays. Whatever numerical heavy algorithm you take from current books on numerical recipes, most likely you will find support for them in SciPy in one way or the other. Whether it is matrix manipulation, linear algebra, optimization, clustering, spatial operations, or even fast Fourier transformation, the toolbox is readily filled.  

SciPy's polyfit() function: Given data x and y and the desired order of the polynomial, it finds the model function that minimizes the error function.  

	fp1 = sp.polyfit(x, y, 1)
We then use poly1d() to create a model function from the model parameters:  

	f1 = sp.poly1d(fp1)  
	
