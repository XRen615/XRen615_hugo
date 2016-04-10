+++
author = "X.Ren"
comments = true
date = "2016-03-15T15:08:09+01:00"
draft = false
image = ""
menu = ""
share = true
slug = "edge_detection"
tags = ["noise reduction", "edge detection","feature selection"]
title = "Discovery the motivation of occupants' behavior from electricity consumption signal"

+++

##### Key Points  

This report elaborates how to analyze the motivation of people's behavior from the electricity consumption signal and other data.  

- **System**: ventilation system in passive houses with adjustable flow rate option.  
- **Objective**: understand how, and why occupants interact with the system.   
- **Raw data**: electricity consumption signal; environment sensor records (temperature, humidity, CO2 etc.); 3-min interval, 2 years (2013-2015)  
- **Technique**: Noise reduction (Gaussian filter); Edge detection (1st derivative Gaussian filter), Feature selection (L1-penalized logistic regression)

Below **Figure 1** shows the overall work flow.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/flowchart.png" width = "630" height = "320"/>  
</div>  

(1) After preprocessing and cleaning, start with the system electricity consumption signal, e.g. **Figure 2** below. First thing to do is filtering the noise (caused by wind etc. or system itself) and "fake operation" (status change with too-short duration). 

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/pulsedemo_noted.png" width = "620" height = "150"/>  
</div>  

(2) After a finely-tuned **1st derivative Gaussian filter**, the "real operations" would be marked out, like shown in **Figure 3** below  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/finertune.png" width = "620" height = "150"/>  
</div>  

(3) Then the marked data would undergo an **undersampling** process since the dataset is  skewed (The no. of records with 'no operation' is far more than ones with 'operation: increase' or 'operation: decrease')  

(4) After undersampling, the training set would be **normalized** and then fed into a **L1-penalized logistic regression classifier**. Since linear models penalized with the L1 norm have sparse solutions i.e. many of their estimated coefficients are zero, it could be used for feature selection purpose. **Figure 4** below shows an example of the coefficients output in a certain experiment.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/L1.png" width = "600" height = "400"/>  
</div> 

Then the logistic regression runs repeatedly to make a **recursive feature elimination**. At last the most informative feature combination (judged by cross-validation accuracy) for this case could be determined, like below **Figure 5**  shows: these features implies the occupant's motivation for his/her behavior.

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/REFCV.png" width = "400" height = "280"/>  
</div>  

(5) The results from different occupants implies there are different kind of people: some of them with strong "time pattern" while others are more sensitive to indoor temperatures etc. A **K-Means clustering** could help us group the occupants into different user profiles.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/motivationdistribution.png" width = "400" height = "300"/>  
</div> 

---

#### Technical Details: Noise reduction & Edge detection 

##### Background  

There is a ventilation system (with heat recovery) in one passive house, of which the ventilation flow rate is controlled by a fan system, and adjustable by occupants. There are 3 available options (let's say, low, medium, high rate respectively)for the fan flow rate setting.  

The electricity consumption of the fan system is recorded by a smart meter in terms of pulse. Obviously, occupants' flow rate setting could put significant influence on the electricity consumption and we could calibrate when and how people adjust their ventilation system based on the electricity consumption.  

However, on the one hand, with the influence of back pressure, wind speed etc. the record is not something like a clear 3-stage square wave, instead it is quite noisy. On the other hand, we got many different houses (with similar structure but with different scales of records)  within our research. They made it is not really practical to calibrate the ventilation setting position by fixed intervals (like pulse < 3 == position 1; 3 < pulse < 5 == position 2 etc.). We need a new algorithmic method to do this job.  

This is a tiny piece of the elec. consumption record (day 185 in year 2014, house #9):   

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/pulsedemo.png" width = "620" height = "150"/>  
</div>  

##### Methodology  

Describe what we want to do in a few words: smooth the noise and detect the edge automatically, without any reset like boundary interval needed. This is actually a classic problem in *signal processing* or *computer vision* field. For this 1D signal the simplest solution maybe **Gaussian derivative filter**, for similar problems in 2D matrix (images) the **canny edge detector** could be effective. The figure may give you a vivid impression of what we are going to do:  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/canny1.jpg" width = "400" height = "150"/>  
</div>


###### Basic idea  

Tune a  Gaussian derivative filter to properly smooth the noise and take 1st derivative, then set an appropriate threshold to detect the edge.  

###### Terms  


**Gaussian filter**  

For noise smoothing or "image blur". In layman's words, replace each point by the weighted average of its neighbors, the weights come from Gaussian distribution, then normalize the results.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/gaussian_formula.png" width = "200" height = "50"/>  
</div>  

![Gaussian](http://7xro3y.com1.z0.glb.clouddn.com/gaussian.png)  

(if you are dealing with 2d matrix (images), use 2-D Gaussian instead.)  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/bg2012110708.png" width = "400" height = "300"/>  
</div>   


Effect:  
![](http://7xro3y.com1.z0.glb.clouddn.com/gblur.jpg)  

**Gaussian derivative filter**  

For noise smoothing and edge detection. In layman's words, replace each point by the weighted average of its neighbors, the weights come from **the 1st derivative of Gaussian distribution**, then normalize the results. 

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

Advantages of Gaussian Kernel compared to other low-pass filter:   

- Being possible to derive from a small set of scale-space axioms.  
- Does not introduce new spurious structures at coarse scales that do not correspond to simplifications of corresponding structures at finer scales.  

**Scale Space**  

Representing an signal/image as a one-parameter family of smoothed signals/images, parametrized by the size of the smoothing kernel used for suppressing fine-scale structures. Specially for Gaussian kernels: t = sigma^2.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202016-03-15%20%E4%B8%8B%E5%8D%886.54.23.png" width = "600" height = "500"/>  
</div>  

##### Results 

Finished a demo of auto edge detection in our elec. consumption record, which contains a tuned Gaussian derivative filter, edge position detected, and scale space plot.  

**Original**  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/pulsedemo.png" width = "620" height = "150"/>  
</div>  

**Gaussian filter smoothed (sigma = 8)**  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/smoothed.png" width = "620" height = "150"/>  
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

#### Finer Tuning  

In practice, it is usually needed to use different tailored tune strategies for the parameters to meet the specific requirements aroused by researchers. E.g. in a case the experts from built environment would like to filter out short-lived status (even they maybe quite steep in terms of pulse number). The strategies is carefully increase sigma (by which you are flattening the Gaussian curve, so the weights of center would be less significant so that the short peaks could be better wiped out by its flat neighbors) and also, properly increase the threshold would help (by which it would be more difficult for the derivatives of smoothed short peaks to pass the threshold and be recognized as one effective operation). Once the sigma and threshold reached an optimized combination, the results would be something like below for this case:  

**Edge position detected (Sigma = 10, threshold = 0.35 * global min/max)**  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/finertune.png" width = "620" height = "150"/>  
</div>  

**In a larger scale, see how does our finely-tuned lazy filter work to filter the fake operations out! (Sigma = 20, threshold = 0.5 * global min/max)**  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/largerscale.png" width = "620" height = "400"/>  
</div> 



#### Reference  

- [Scale Space wiki](https://en.wikipedia.org/wiki/Scale_space) 

- [Gaussian Blur Algorithm](http://www.ruanyifeng.com/blog/2012/11/gaussian_blur.html)  

- [OpenCV Canny Edge Detector](http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html)  

- [UNC Edge Detector 1D](http://www.cs.unc.edu/~nanowork/cismm/download/edgedetector/)  

- [scikit-image Canny edge detector](http://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html#example-edges-plot-canny-py)  

---

#### Technical Details: Feature selection




