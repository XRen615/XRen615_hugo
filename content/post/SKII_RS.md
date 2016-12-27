+++
author = "X.Ren"
comments = true
date = "2016-12-19T13:21:09+08:00"
draft = false
image = ""
menu = ""
share = true
slug = "sk2_rs"
tags = ["recommender system"]
title = "SKII Recommender System Design"

+++

### Abstract

This prototype finished during P&G Data Science Hackthon, Nov. 2016 in Cincinati, OH.  

This document briefly elaborates the design ideology for SKII onsite recommender system, leverage the demographics data, skin scan result and purchase history.

Note. all the confidential data has been pre-processed.  

***  

### The Dataset  

The data used in this recommender system consists of the folling 3 parts:  

1. Demographics information (1 million rows * 10 columns), including gender, age, marital status etc.. Collected through the membership registration.  
2. Skin scan results (1 million rows * 21 columns), including varies parameters evaluate your pores, skin firmness etc.. Collected via the onsite [*Magic Ring*](http://www.sk-ii.com.sg/en/magic-ring.aspx) skin test. 
3. Purchasing history (1 million consumer * 131 SKU). Collected in the customer relationship management (CRM) system.  

***  

### Methodology  
Obviously there are 2 kinds of customers: **new customer** and **return customer**, between which, the difference need to be highlighted is the data availiability:

- For new comers, since they haven't purchased any item yet, the purchasing history is empty. Thus, the data availiable contains only demographic information and skin scan result. 

- For return customers, all of the 3 data categories mentioned above (demographic, skin scan results, purchase history) are availiable.  

Based on the different data schema, different strategy should be used for new comers and return customers.

#### New Comers

The basic idea is:

1. Instead of always recommending the general best sellers to you (unfortunately this is what they are doing right now), we listen more to people like you. 
2. Instead of what they bought, we care more about what makes them return, if applicable. 


So technically, I constructed a vector containing demographic info as well as skin scan result, find peers who are similar to you by calculating cosine similarity, then look at what makes them return, make a vote and recommend those stuff to you.


<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/method1.png" width = "530" height = "130"/>  
</div>  

#### Return Customers

Besides the demographic & skin info, which enbale the similarity based recommendation, we also hold the purchasing records for those return users. So how can we leverage those records to make the recommender system better?  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/purchasing.png" width = "120" height = "120"/>  
</div>

The basic idea is:

- Involve **collaborative filtering** for return customers, so the system could iterate and evolve: the more you buy, the better recommendation we can make.

So technically, considering the computing load, I constructed an item-based collaborative filtering using a thrid-party library [GraphLab](https://turi.com), make a recommendation and vote with the similarity-based results to get the final recommendation.


<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/method2.png" width = "530" height = "230"/>  
</div>  

***  

### Implementation

#### Data Preprocessing/ETL

Including data cleaning, normalization, mapping etc.


<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/pre1.png" width = "430" height = "330"/>  
</div>  

Implemented in [KNIME](https://www.knime.org)  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/pre3.png" width = "530" height = "300"/>  
</div>  

#### Similar People Based Recommendation

Implemented in Python (part of the code)

```  
# Calculating the similarity
# No fancy algorithm, we chooose cosine, sorry

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
 
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)
    
# Calculating the similarity between him/her with peers!   
scores = []
for index, row in df_GM_sample.iterrows():
    peer = row.values.tolist()
    score = cosine_similarity(new_list,peer)   
    scores.append(score)  
    
# The most look-alike guys we are looking for (top 100 out of 100000)
similar_users = df_GM_sample.index[0:100]
```  


What are their return-makers? we recommend those stuff

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/output_40_1.png" width = "420" height = "500"/>  
</div>   

#### Item Based Collaborative Filtering  

GraphLab provide a clean command to build collaborative filter once you organized your dataframe in their way.

```
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='Customer_id', item_id='item')
```  
Then we could make the following recommendations 

```
# Make Recommendations:
item_sim_recomm = item_sim_model.recommend(users=['5208494361'],k=10)
item_sim_recomm.print_rows()
```  
Then top 10 SKU to be recommended based on the collaborative filtering:  

<div  align="center">    
<img src="http://7xro3y.com1.z0.glb.clouddn.com/top10.png" width = "430" height = "200"/>  
</div>

#### Vote

The idea is simple: select the recommendations from people-similarity based method as well as collaborative filtering based method, make a vote as the final recommendation.  

***  

### Performance

On the test set we made a quick evaluation for the improvement. The precision of recommendation reached **61%** using under this methodology, in comparison with **30%-**, previous recommendation based on the overall best-sellers.  
  
***  

### Issues To Be Settled  

This is an early-stage demo finished during hackthon, (although it's the hackthon winner :p) before put into real business usage, some issues should be settled.

- The feature selection: over 150 features are treated as equal in this stage, which is apparently not realistic. So, feature selection based on corelation analysis/chi-squared testing or other more advanced methods such as PCA could be involved.
- Computing optimization: in order to reduce the computing time, in this stage I sampled the pool. Of course there are better ways to do so, optimize big O/play tricks on sparse matrix etc. could help.


