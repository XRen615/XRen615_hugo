+++
author = "X.Ren"
comments = true
date = "2016-03-15T15:08:09+01:00"
draft = false
image = ""
menu = ""
share = true
slug = "edge_detection"
tags = ["Noise reduction", "Edge detection","Feature selection"]
title = "A data-based approach for behavior motivation digging"

+++

##### Key Points  

This chapter briefly elaborates how to analyze the motivation of people's operation on a system from the system electricity consumption signal and other data.  

- **Objective**: understand how, and why occupants interact with the system.
- **System**: ventilation system in passive houses with adjustable flow rate option.  
- **Raw data**: electricity consumption signal; environment sensor records (temperature, humidity, CO2 etc.); 3-min interval * 2 years (2013-2015): 325946 rows × 25 features.  
- **Technique**: Noise reduction (Gaussian filter); Edge detection (1st derivative Gaussian filter), Feature selection (L1-penalized logistic regression, recursive feature elimination)

Below **Figure 1** shows the overall pipeline I designed.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/flowchartbig.png" width = "800" height = "400"/>  
</div>
 
(1) After essential preprocessing and cleaning (NaNs are backfilled), start with a system electricity consumption signal like **Figure 2** below. A sudden change in the signal could imply the occupants' interaction with the system (e.g. once the occupant turn the flow rate into a higher option there should be a steep increasing edge on the electricity consumption signal). First thing to do is filtering out the noise (caused by wind etc. or system itself) and "fake operation" (status change with too-short duration). 

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/demo_original_noted%E5%89%AF%E6%9C%AC.jpg" />  
</div>  

In the previous research work before this study, researchers used to calibrate each position with fixed interval. E.g. in this case, positions with pulse no. fallen in [0,4] are assigned as ‘position 1’, while (4,8) for ‘position 2’ and pulse no. larger than 8 represents ‘position 3’. Follow this approach, the user operation frequency could be seriously over-estimated since both the noise (e.g. in circle 1) and ‘fake operations’ (e.g. in circle 2) are counted as effective user operation. In fact, in the previous report the researchers estimated this house with over 1,000 operations per year, which is apparently too much for a regular ventilation system controller. To make things worse, with the fixed-interval approach, for each house the intervals need to be decided case by case since the scope of the no. of pulse in different house may vary. In the next paragraphs, I will show how does the filter-based approach developed in this study solve all the issues mentioned above by automatically marking the effective operational edges and filtering out the noises and ‘fake operations’.  

(2) Through a finely-tuned **1st derivative Gaussian filter**, the noise and "fake operation" could be filtered out and the valid operations would be marked out, like shown in **Figure 3** below.   

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/combine2%E5%89%AF%E6%9C%AC.jpg"/>  
</div>  

In the figure above, (a) is the raw signal of fan electricity consumption, with noise and fake operations; (b) shows the signal after the Gaussian filter, with which the signal is smoothed and the noise is reduced; (c) is the 1st derivative signal of (b), each peak here could imply an edge in (b), with a proper threshold, we can filter out the real operation edge we want in a certain sensitivity. (d) is the original signal with operation edge marked out from (c), it could be observed that the finely-tuned algorithm could automatically ignore the noise and fake operation, only mark the real operation edge we want.  

With a lager scale in2 years, the operation detected could be presented in the **Figure 4** below, in which +1 represents increasing operation while -1 represents decreasing operation.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/ventpos%20setting%20operation.png"/>  
</div>  


(3) Then the marked data set would undergo an **undersampling** process since the dataset is now skewed (The no. of records marked with 'no operation' is far more than ones with operation, either increase or decrease). The undersampling process ensures the data set has balanced scales with each class, for the effectiveness of following classification algorithm.  

(4) After undersampling, the training set would be **normalized** and then fed into a **L1-penalized logistic regression classifier**. Since linear model penalized with L1 norm has sparse solutions i.e. many of its estimated coefficients would be zero, it could be used for feature selection purpose. **Figure 5** below shows an example of the coefficients output in a certain experiment.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/L1.png" width = "700" height = "430"/>  
</div> 

Then the logistic regression runs repeatedly to make a **recursive feature elimination** (first, the estimator is trained on the initial set of features and weights are assigned to each one of them. Then, features whose absolute weights are the smallest are pruned from the current set features). At last, the most informative feature combination (judged by cross-validation accuracy) in this case could be determined, like below **Figure 6**  shows: these features implies this occupant's motivation for his/her behavior.

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/REFCV.png" width = "500" height = "340"/>  
</div>  

(5) Repeat the process above for different occupants. The results imply there are different kinds of people since their "best feature combination" vary a lot: e.g. some of them are with strong "time pattern" while others may be more sensitive to indoor environment, like temperature etc. A **K-Means clustering** could help us demonstrate this by grouping the occupants into different user profiles.  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/motivationdistribution.png" width = "600" height = "450"/>  
</div> 

In the figure above, the horizontal axis represents the importance of indoor environment in determining occupants’ behavior, while the vertical axis represents the importance of time-related factors. Three different types of occupants could be observed:  
- Indoor environment sensitive occupants: house no. 2, 4, 6, 8  - Time sensitive occupants: house no.7, 9  - Mixed type occupants: houses no. 1, 3, 5, 10  
The complexity of occupants’ behavioral pattern is demonstrated by the data analysis result. The Indoor environment sensitive occupants are more likely to interact with their ventilation control panel when they feel unsatisfied about the indoor comfort, while the time sensitive occupants are more likely to have fixed timetables for their behavior (e.g., as soon as they wake up or come back from work etc.) and there are also some people in between, as mixed-type occupants their behaviors are effected considerably by both factors in the same time.

---  

**From here below is technical log regarding relevant theory and code to realize the whole process.**

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



##### Reference  

- [Scale Space wiki](https://en.wikipedia.org/wiki/Scale_space) 

- [Gaussian Blur Algorithm](http://www.ruanyifeng.com/blog/2012/11/gaussian_blur.html)  

- [OpenCV Canny Edge Detector](http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html)  

- [UNC Edge Detector 1D](http://www.cs.unc.edu/~nanowork/cismm/download/edgedetector/)  

- [scikit-image Canny edge detector](http://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html#example-edges-plot-canny-py)  

---

#### Technical Details: Feature selection  

Before feature selection I made an undersampling to the data set to ensure every class shares a balanced weight in the whole dataset (before which the ratio is something like 150,000 no operation, 400 increase, 400 decrease).  

The feature selection process is carried out in Python with scikit-learn. First each feature in the data set need to be standardized since the objective function of the l1 regularized linear model we use in this case assumes that all features are centered on zero and have variance in the same order. If a feature has a significantly lager scale or variance compared to others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected. In this case I used sklearn.preprocessing.scale() to standardize each feature to zero mean and unit variance.  

Then the standardized data set was fed into a recursive feature elimination with cross-validation (REFCV) loop with a L1-penalized logistic regression kernel since linear models penalized with the L1 norm have sparse solutions: many of their estimated coefficients are zero, which could be used for feature selection purpose.  

Below is the main part of the coding script for this session (ipynb format).  


##### Feature selection after Gaussian filter  

	import matplotlib.pyplot as plt
	%matplotlib inline
	import numpy as np
	import pandas as pd
	import seaborn as sns
	import math

**Data Loading**


```
ventpos = pd.read_csv("/Users/xinyuyangren/Desktop/1demo.csv")
ventpos = ventpos.fillna(method='backfill')
ventpos.head()
ventpos.op.value_counts()
```

**UnderSampling**


```
sample_size = math.ceil((sum(ventpos.op == 1) + sum(ventpos.op == -1))/2)
sample_size
noop_indices = ventpos[ventpos.op == 0].index
noop_indices
random_indices = np.random.choice(noop_indices, sample_size, replace=False)
random_indices
noop_sample = ventpos.loc[random_indices]
up_sample = ventpos[ventpos.op == 1]
down_sample = ventpos[ventpos.op == -1]
op_sample = pd.concat([up_sample,down_sample])
op_sample.head()
```

**Feature selection: up operation**


```
undersampled_up = pd.concat([up_sample,noop_sample])
undersampled_up.head()
#generate month/hour attribute from datetime string  
undersampled_up.dt = pd.to_datetime(undersampled_up.dt)
t = pd.DatetimeIndex(undersampled_up.dt)
hr = t.hour
undersampled_up['HourOfDay'] = hr
month = t.month
undersampled_up['Month'] = month
year = t.year
undersampled_up['Year'] = year
undersampled_up.head()
for col in undersampled_up:
    print col
```


```
def remap(x):
    if x == 't':
        x = 0
    else:
        x = 1
    return x

for col in ['wc_lr', 'wc_kitchen', 'wc_br3', 'wc_br2', 'wc_attic']:
    w = undersampled_up[col].apply(remap)
    undersampled_up[col] = w
undersampled_up.head()
openwin = undersampled_up.wc_attic + undersampled_up.wc_br2 + undersampled_up.wc_br3 + undersampled_up.wc_kitchen + undersampled_up.wc_lr
undersampled_up['openwin'] = openwin;
undersampled_up = undersampled_up.drop(['wc_lr', 'wc_kitchen', 'wc_br3', 'wc_br2', 'wc_attic','Year','dt','pulse_channel_ventilation_unit'],axis = 1)
undersampled_up.head()
for col in undersampled_up:
    print col
```

**Logistic Regression**


```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
```


```
#shuffle the order
undersampled_up = undersampled_up.reindex(np.random.permutation(undersampled_up.index))
undersampled_up.head()
```


```
y = undersampled_up.pop('op')
```


```
# Columnwise Normalizaion
from sklearn import preprocessing
X_scaled = pd.DataFrame()
for col in undersampled_up:
    X_scaled[col] = preprocessing.scale(undersampled_up[col])
X_scaled.head()
```


```
from sklearn import cross_validation
lg = LogisticRegression(penalty='l1',C = 0.1)
scores = cross_validation.cross_val_score(lg, X_scaled, y, cv=10)
#The mean score and the 95% confidence interval of the score estimate
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```


```
clf = lg.fit(X_scaled, y)
```


```
plt.figure(figsize=(12,9))
y_pos = np.arange(len(X_scaled.columns))
plt.barh(y_pos,abs(clf.coef_[0]))
plt.yticks(y_pos + 0.4,X_scaled.columns)
plt.title('Feature Importance from Logistic Regression')
```

**REFCV FEATURE OPTIMIZATIN**


```
from sklearn.feature_selection import RFECV
```


```
selector = RFECV(lg, step=1, cv=10)
selector = selector.fit(X_scaled, y)
mask = selector.support_ 
mask
```


```
selector.ranking_
```


```
X_scaled.keys()[mask]
```


```
selector.score(X_scaled, y)
```


```
X_selected = pd.DataFrame()
for col in X_scaled.keys()[mask]:
    X_selected[col] = X_scaled[col]
X_selected.head()
```


```
scores = cross_validation.cross_val_score(lg, X_selected, y, cv=10)
#The mean score and the 95% confidence interval of the score estimate
scores
```


```
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```


```
clf_final = lg.fit(X_selected,y)
```


```
y_pos = np.arange(len(X_selected.columns))
plt.barh(y_pos,abs(clf_final.coef_[0]))
plt.yticks(y_pos + 0.4,X_scaled.columns)
plt.title('Feature Importance After RFECV Logistic Regression')
```
  

##### Reference  

- [sklearn standardization](http://scikit-learn.org/stable/modules/preprocessing.html)  
- [undersampling](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)  
- [sklearn feature selection](http://scikit-learn.org/stable/modules/feature_selection.html)  
- [sklearn REFCV](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV)




