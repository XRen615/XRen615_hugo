+++
date = "2017-03-02T17:40:40+08:00"
title = "Python Data Engineering Cheat Sheet"
author = "X.Ren"
comments = true
draft = false
share = true
slug = "python_cheat_sheet"
tags = ["python","cheat sheet"]

+++

Some frequent needed utilities in Python data scripts —— good to have it by hand when facing puzzle.


### ETL  
---  


```
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
```

***Data Loading***  

```
import pandas as pd  

```

```  
# From CSV
df = pd.read_csv("path")

# From Excel
df = pd.read_excel('/path')

# From database (sqlite)
import sqlite3
conn = sqlite3.connect("foo.db")
cur = conn.cursor()
#Check how many tables are there in the database
cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
#SQL Query to pandas dataframe
df = pd.read_sql_query("select * from bar limit 5;", conn)
```  

***Indexing***  

```
# Set index
df = df.set_index('colName')
# loc works on labels in the index
s.loc[:3]
# iloc works on the positions in the index (so it only takes integers)
s.iloc[:3]
```

***Sorting***  


```
# First c1, then c2
df = df.sort(['c1','c2'], ascending=[False,True])
```

***Dropping***  

```
# Drop columns (axis=0: column-wise; axis=1: row-wise)
df = df.drop(['Cabin','Ticket'],axis = 1)
# Drop rows
df = df.drop(['label string']) # by index name
df.drop(df.index[[1,3]]) # by row number (0-based)
```  

***Slicing***  

```
# Filter the dataset by a certain condition
df = df[df.name != 'Tina']
```  

***Dealing with missings***  

```
# Drop them
df = df.dropna()
# Fill them
df.fillna(0) # fill by a number
df.fillna(method='ffill') # propagates last valid observation forward to next valid
df.fillna(method='bfill')
```  

***Sampling***  

```  
df_GM_sample = df_GM.sample(n=None, frac=None)
```  

***Apply function***

```
# axis=0: column-wise; axis=1: row-wise
df.apply(func,axis = )
# apply to every element
df.applymap(lambda x: )
```  

***Dealing with datetime***  

```
# string to datetime
df.dt = pd.to_datetime(df.dt, format='%Y%m%d')

# get datetime indexes
t = pd.DatetimeIndex(df.dt)
hr = t.hour
df['HourOfDay'] = hr
month = t.month
df['Month'] = month
year = t.year
df['Year'] = year

# resample time series
df = df.set_index('datetime')
weekly_summary['speed'] = df.speed.resample('W').mean()
weekly_summary['distance'] = df.distance.resample('W').sum()
weekly_summary['cumulative_distance'] = df.cumulative_distance.resample('W').last()

# generate given format string from datetime
df['DOB1'] = df['DOB'].dt.strftime('%m/%d/%Y')
```  

***Categorical to dummy***  

```
dummiesT = pd.get_dummies(test['Embarked'],prefix = 'Embarked')
test = pd.concat([test,dummiesT],axis = 1)
test = test.drop('Embarked',axis =1)
```

***concat & join***

```
# concat along rows
df_new = pd.concat([df_a, df_b])

# join
df = df1.join(df2, how='left', lsuffix='', rsuffix='', sort=False)
```  

***Groupby***  

```
df.groupby(by = 'Sex').mean()
```

***Differencing & Cumulation***

```
# Differencing
data['instantaneous'] = data.volume_out.diff()

# Cumulation
consum.loc[:,"group"] = consum["is_start_point"].cumsum()
```  

***Sliding Window Apply***  

```
df["is_lucky_than_previous"] =\
pd.rolling_apply(df.Survived, 2, lambda x: x[1] - x[0] == 1).fillna(1)
```


***Regular Expression***

```
 def volCalc(row):
    name = row['tbordername']
    try:
        vol = 0
        p = re.compile(r'(\d+)ml')
        sizes = p.findall(name)
        for size in sizes:
            p1 = re.compile(size + r'ml\D+(\d)\D+')
            amount = p1.findall(name)
            if amount:
                vol += int(size)*int(amount[0])
            else:
                vol += int(size)*1
        return vol
    
    except:
        return 'N/A'
```

### Descriptive Stats  
---  


***Numerical stats***  

```
df.describe()
```  

***Correlation***

```
corr = df.corr()
plt.matshow(df.corr())
```  

***Basic Charts***  

```  
# line chart
fig = plt.figure(figsize=(12,6))
plt.plot(data.dateTime,data.volume_out)
plt.title('title')

# hist: numerical feature distribution
df.Age.hist()
# categorical feature distribution  
df.Survived.value_counts().plot(kind = 'bar')
# Basic box plot
sns.boxplot(consum.instantaneous,orient='v')
plt.title('instantaneous consumption value distribution')
# Box plot with hue
sns.boxplot(x="Sex", y="Age",hue = 'Survived', data=df, palette="Set3")

# Scatter
plt.scatter(df.Fare,df.Survived)
plt.xlabel('Fare')
plt.ylabel('Survived?')

# Regression chart
sns.jointplot(x="duration", y="usage", kind = 'reg', data=filtered)
plt.title('title')


```   

### Feature Engineering  
---  

***Rescaling***  

```
# (0,1) scaling 
# (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
cols_to_norm = ['PassengerId','SibSp']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: scaler.fit_transform(x))

# Standardization: Zero mean and unit variance
from sklearn.preprocessing import scale
cols_to_norm = ['Age','SibSp']
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: scale(x))

# Normalization: scaling individual observation (row) to have unit norm.
# if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples, like KNN. 
from sklearn.preprocessing import normalize
df_normalized = pd.DataFrame(normalize(df._get_numeric_data(),norm = 'l2'),columns=df._get_numeric_data().columns,index=df._get_numeric_data().index)
df_normalized.apply(lambda x: np.sqrt(x.dot(x)), axis=1) # check results
```
***Feature Binarization***  

```
# thresholding numerical features to get boolean values
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=30)
df['Age'] = df['Age'].apply(lambda x: binarizer.fit_transform(x)[0][0])
```  

***Generating Polynomial Features***

```
# get features’ high-order and interaction terms
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
#  (X_1, X_2) to (1, X_1, X_2, X_1^2, X_1X_2, X_2^2)
X_poly = pd.DataFrame(poly.fit_transform(X))
```

### Feature Selection
---  

***Filter methods***

```
# Variance Treshhold
from sklearn.feature_selection import VarianceThreshold 

# Univariate feature selection 
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
```  


***Wrapper Methods*** 

```
# LASSO
class sklearn.linear_model.Lasso()

# Tree-based
class sklearn.ensemble.RandomForestClassifier()
```  

### Algorithm  
---  

[***Sk-Learn Official Cheat Sheet***](http://scikit-learn.org/stable/tutorial/machine_learning_map/)
<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/sklearncs.png" align=center width = "800" height = "500"/>  
</div>  

***Frequent Used Pieces***  

***Linear Regression***

```
from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept=True)
# Train the model using the training sets
regr.fit(df, y)
# The coefficients
print('Coefficients:', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(df) - y) ** 2))
# Explained variance score: 1 is perfect prediction
print "R Squared score:";regr.score(df, y)

# Coef_ check
plt.figure(figsize=(12,8))
plt.barh(range(len(regr.coef_)),regr.coef_,height=0.2,tick_label = df.columns)
plt.title('Regression Coefficients')

# Residuals Check
res = regr.predict(df) - y
plt.axhline(0)
plt.scatter(range(len(res)),res.values,color = 'r')
plt.title('Residual Plot')
```

***Kmeans***  


```
from sklearn.cluster import KMeans
estimator = KMeans(n_clusters=3)
estimator.fit(filtered_scaled)
labels = estimator.labels_
```  

***Random Forest***  

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
rf = RandomForestClassifier(n_estimators=8000,n_jobs=-1,oob_score=True)
rf.fit(train,res)
rf.oob_score_
# feature importance
feature_importances = pd.Series(rf.feature_importances_,index = train.columns)
feature_importances.sort(inplace = True)
feature_importances.plot(kind = 'barh')
```  

***[Recommender](https://turi.com)***

```
# item-based CF
import graphlab
train_data = graphlab.SFrame(df_CF)
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='Customer_id', item_id='item')
# Make Recommendations
item_sim_recomm = item_sim_model.recommend(users=['5208494361'],k=10)
item_sim_recomm.print_rows()
```

***Time Series***  

```
# DF Test
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=10)
    rolstd = pd.rolling_std(timeseries, window=10)

    # Plot rolling statistics:
    plt.figure(figsize=(14,8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries.SALES_IN_ML, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
```  

```  
# ARIMA
# Ordering: ACF and PACF plots
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(TS_log, lags=30)
plot_pacf(TS_log, lags=30)

# Ordering: AIC
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
# Smaller Better
arma_order_select_ic(TS_log, max_ar=4, max_ma=0, ic='aic')

# Modeling
model = ARIMA(TS_log, order=(1, 0, 0))  
results_AR = model.fit(disp= 1)  
plt.plot(TS_log)
plt.plot(results_AR.fittedvalues, color='red')
TS_fitted = pd.DataFrame(results_AR.fittedvalues,columns=['SALES_IN_ML'])

# Residual Series Check
residuals = TS_log - TS_fitted
test_stationarity(residuals)
```  



### Tuning & Validation  
---  

***Training/Test split***  

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
```  

***Cross Validation***

```
from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X.values, y[0].values, cv=5)
# 95% confidence interval of the score
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

***Exhaustive Grid Search***  

```
from sklearn.model_selection import GridSearchCV
svr = svm.SVC()
parameters = {'kernel':('linear','rbf'), 'C':[1, 10]}
clf = GridSearchCV(svr, parameters,n_jobs = -1,cv = 5)
clf.fit(X, y[0])
clf.cv_results_
clf.best_estimator_
```

