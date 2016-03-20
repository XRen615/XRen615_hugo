+++
date = "2016-03-08T19:19:22+01:00"
description = "master thesis progress record"
keywords = []
title = "Data Mining Project for Rosendal Passive House Database"
author = "X.Ren"
comments = true
draft = false
image = ""
menu = ""
share = true
slug = "meeting0309"
tags = ["master thesis"]

+++

#### Progress Mar. 3rd ~ Mar. 9th  

In this week I mainly finished:  

1. Systematic review regarding **feature selection techniques**: During last meeting Prof. Gert and msc. Christos suggested it would be better if there is a more ‘systematic’ evaluation methodology to support the algorithm used. So this week I reviewed some scientific references talking about the archetecture of feature selection techniques in different level and summerized them into this note on my blog [(find it here)](http://xren615.github.io/article/feature-reducing_eng/).
Benefit a lot from your suggestion, I am also planing to somehow re-construct part of the methodology of my thesis, to be more 'mathematic' and persuasive.  

2. Fixed the issues regarding the study of **ventilation setting position**: knew from Prof.Gert that the pulse I used as 'flowrate' is actually 'power consumption' and the boundaries should be different for type 505 and 506. I checked with Prof. Loomans's scripts and found that the boundaries to differentiate the 3 setting positions are  (5.5, 9) for type 505 and (4, 6) for type 506 respectively. Based on this, the code for the study of ventilation position setting is re-constructed. In summary, now:  

	- 'Pulse' bug fixed.    
	- The freqency of setting change is recorded.    
	- 2 years were seperated respectively.  
	- 2 house types were seperated respectively.


Check some of the plots here:  

![](http://7xro3y.com1.z0.glb.clouddn.com/freq.png)
![](http://7xro3y.com1.z0.glb.clouddn.com/yearly_avg.png)
![](http://7xro3y.com1.z0.glb.clouddn.com/2013.png)
![](http://7xro3y.com1.z0.glb.clouddn.com/scatter_2013.png)
![](http://7xro3y.com1.z0.glb.clouddn.com/2014.png)
![](http://7xro3y.com1.z0.glb.clouddn.com/scatter_2014.png)

#### Plan for next week  

- Re-determine the interval of position division.  
- Think about the thesis report scope.  
