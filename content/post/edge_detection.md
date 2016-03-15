+++
author = ""
comments = true
date = "2016-03-15T15:08:09+01:00"
draft = false
image = ""
menu = ""
share = true
slug = "post-title"
tags = ["Edge detection", "Signal processing","Master thesis"]
title = "Edge detection of energy consumption signal with Gaussian filter"

+++

##### Background  

There is a ventilation system (with heat recovery) in one passive house, of which the ventilation flow rate is controlled by a fan system, and adjustable by occpants. There are 3 availiable options (let's say, low, medium, high rate respectively)for the fan flowrate setting.  

The electricity consumption of the fan system is recorded by a smart meter in terms of pulse. Obviously, occupants' flowrate setting could put significant influence on the electricity consumption and we could calibrate when and how people adjust their ventilation system based on the electricity consumption.  

However, on the one hand, with the influence of back pressure, wind speed etc. the record is not something like a clear 3-stage square wave, instead it is quite noisy. On the other hand, we got many different houses (with similar structure but with different scales of records)  within our research. They made it is not really practical to calibrate the ventilation setting position by fixed intervals (like pulse < 3 == position 1; 3 < pulse < 5 == position 2 etc.). We need a new algorithmic method to do this job.  

This is a tiny piece of the elec. consumption record (day 185 in year 2014, house #9):   

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/pulsedemo.png" width = "620" height = "150"/>  
</div>  

##### Methodology  

Describe what we want to do in a few words: smooth the noise and detect the edge automatically, without any reset like boudary interval needed. This is actually a classic problem in *signal processing* or *computer vision* field. For this 1D signal the simplest solution maybe **Gaussian derivative filter**, for similar problems in 2D matrix (images) the **canny edge detector** could be effective. The figure may give you a vivid impression of what we are going to do:  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/canny1.jpg" width = "400" height = "150"/>  
</div>


###### Basic idea  

Tune a  Gaussian derivative filter to properly smooth the noise and take 1st deravative, then set an appropriate threshold to detect the edge.  

###### Terms  


**Gaussian fliter**  

For noise smoothing or "image blur". In layman's words, replace each point by the weighted average of its neighbors, the weights come from Gaussian distribution, then normalize the results.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/gaussian_formula.png" width = "200" height = "50"/>  
</div>  

![Gaussian](http://7xro3y.com1.z0.glb.clouddn.com/gaussian.png)  

(if you are dealing with 2d matrix (images), use 2-D Gaussian instead.)    

![2dG](http://7xro3y.com1.z0.glb.clouddn.com/bg2012110708.png)  

Effect:  
![](http://7xro3y.com1.z0.glb.clouddn.com/gblur.jpg)  

**Gaussian derivative filter**  

For noise smoothing and edge detection. In layman's words, replace each point by the weighted average of its neighbors, the weights come from ** the 1st detivative of Gaussian distribution**, then normalize the results. 

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/1st_der_Gaussian.png" width = "200" height = "50"/>  
</div>  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/Gaussian_curves.png" width = "600" height = "400"/>  
</div>

Effect:  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/canny1.jpg" width = "400" height = "150"/>  
</div>  

Advantages of Gaussian Kernal compared to other low-pass filter:   

- Being possible to derive from a small set of scale-space axioms.  
- Does not introduce new spurious structures at coarse scales that do not correspond to simplifications of corresponding structures at finer scales.  

**Scale Space**  

Representing an signal/image as a one-parameter family of smoothed signals/images, parametrized by the size of the smoothing kernel used for suppressing fine-scale structures. Specially for Gaussian kernals: t = sigma^2.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202016-03-15%20%E4%B8%8B%E5%8D%886.54.23.png" width = "600" height = "500"/>  
</div>  

##### Results 

Finished a demo of auto edge detection in our elec. consumption record, which contains a tuned Gaussian derivative filter, edge position detected, and scale space plot.  

**Original**  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/pulsedemo.png" width = "620" height = "150"/>  
</div>  

**1st derivative Gaussian filtered (sigma = 8)**  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/demo_edge_Gassuian.png" width = "620" height = "150"/>  
</div>  

**Edge position detected (threshold = 0.07 * global min/max)**  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/demo_edge_detected.png" width = "620" height = "150"/>  
</div> 

**Scale Space (sigma = range (1,9))**  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/edge_Gassuian_scale_space.png" width = "600" height = "400"/>  
</div> 


