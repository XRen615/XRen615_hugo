+++
draft = false
tags = ["CTR prediction"]
date = "2018-07-16T01:36:01+08:00"
author = "X.Ren"
comments = true
share = true
slug = "pCTR"
title = "Tencent social ads algo. competition: wrap up"
+++

腾讯2018社交广告大赛。与朋友一起，借助FB的'Practical Lessons from Predicting Clicks on Ads at Facebook'一文的基本架构进入决赛。主要流程总结如下

![flow](http://7xro3y.com1.z0.glb.clouddn.com/tencent_ads_flow.png)   

```
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import time
from sklearn.metrics import roc_auc_score

print('Start loading Train, Test, Adfeature...')
# Data Loading
train = pd.read_csv("/home/riren/Documents/4fun/preliminary_contest_data/train.csv",sep=",")
test = pd.read_csv("/home/riren/Documents/4fun/preliminary_contest_data/test2.csv",sep=",")
adFeature = pd.read_csv("/home/riren/Documents/4fun/preliminary_contest_data/adFeature.csv",sep=",")

print('Train, Test, Adfeature Loaded')

# # sampling for debug
# print('Train, Test, Adfeature sampling')
# train = train.sample(200000)
# test = test.sample(100000)
# print('train, test sampled')

def parseChunk(chunk):
    userFeature_data = []
    for each_list in chunk[0].map(lambda x:x.split("|")):
        userFeature_dict = {}
        for element in each_list:
            pairs = element.split(' ')
            userFeature_dict[pairs[0]] = ' '.join(pairs[1:])
        userFeature_data.append(userFeature_dict)
    return userFeature_data


print('Loading userFeature...')
LARGE_FILE = "/home/riren/Documents/4fun/preliminary_contest_data/userFeature.data"
CHUNKSIZE = 100000 # processing 100k rows at a time

userFeature = pd.DataFrame()
reader = pd.read_table(LARGE_FILE, chunksize=CHUNKSIZE,header=None)
for chunk in reader:
    # process each data frame
    userFeature_data = parseChunk(chunk)
    tmp = pd.DataFrame(userFeature_data)
    userFeature = pd.concat([userFeature, tmp], ignore_index=True)
print('UserFeature Loaded')


# # sampling for debug
# print('Loading userFeature sampled...')
# LARGE_FILE = "/home/riren/Documents/4fun/preliminary_contest_data/userFeature.data"
# NROWS = 300000 # processing 100k rows at a time
# CHUNKSIZE = 100000

# userFeature = pd.DataFrame()
# reader = pd.read_table(LARGE_FILE, chunksize=CHUNKSIZE,nrows = NROWS, header=None)
# for chunk in reader:
#     # process each data frame
#     userFeature_data = parseChunk(chunk)
#     tmp = pd.DataFrame(userFeature_data)
#     userFeature = pd.concat([userFeature, tmp], ignore_index=True)
# print('UserFeature sample Loaded')

userFeature.uid = userFeature.uid.apply(lambda x : int(x))

# Feature Eng

train.loc[train['label']==-1,'label'] = 0
test['label'] = -1
data = pd.concat([train,test])
data=pd.merge(data,adFeature,on='aid',how='left')

# ad statistical feature
temp = data[['aid','label']][:len(train)]
aid_conv = temp.groupby('aid', as_index = False).mean()
aid_conv = aid_conv.rename(columns={"label": "aid_conv"})
# re-scale
aid_conv['aid_conv'] = (aid_conv['aid_conv'] - aid_conv['aid_conv'].min())/aid_conv['aid_conv'].max()
data = pd.merge(data,aid_conv,on='aid',how='left')

data = pd.merge(data,userFeature,on='uid',how='left')
print('data merge finished')

data=data.fillna('-1')

print('Feature encoding...')

label_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct',
                 'marriageStatus','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType']

vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4',
                'interest5','kw1','kw2','kw3','topic1','topic2','topic3']

# Label encoding
for feature in label_feature:
    print('Encoding ' + feature + '...')
    try:
        data[feature] = preprocessing.LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = preprocessing.LabelEncoder().fit_transform(data[feature])

cols = data.columns.tolist()
y = data.pop('label')
conv_cols = ['aid_conv']
base_cols = ['creativeSize'] + conv_cols + label_feature # aid uid hold-out
df_base_cols = data[base_cols]

# sparsify
df_sparse = scipy.sparse.csr_matrix(df_base_cols.values)

# # one-hot encoding
# for col in label_feature:
#     print ('one-hot encoding ' + col + '...')
#     enc = OneHotEncoder()
#     one_hot_encoded = enc.fit_transform(data[col].values.reshape(-1,1))
#     df_sparse = sparse.hstack([df_sparse, one_hot_encoded])

# Vectorizing
cv = CountVectorizer()
for col in vector_feature:
    print('Vectorzing ' + col + '...')
    try:
        cv.fit(data[col])
        cv_col = cv.transform(data[col])
        df_sparse = sparse.hstack([df_sparse, cv_col])
    except:
        print ('error passed')
        continue

del data

df_sparse = csr_matrix(df_sparse)

train_sparse = df_sparse[:len(train)]
test_sparse = df_sparse[len(train):]
del df_sparse

y_ = y[:len(train)]

print('Feature eng finished')

# feature selection

print('train_sparse shape is %s' %(str(train_sparse.shape)))

# print('L1 Norm LR training...')
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# lr_fs = LogisticRegression(penalty="l1",n_jobs = -1).fit(train_sparse, y_)

# print('L1 Norm LR Feature selection in progress...')
# model = SelectFromModel(lr_fs, threshold = 'mean',prefit=True)
# train_sparse_fs = model.transform(train_sparse)
# test_sparse_fs = model.transform(test_sparse)
# del train_sparse
# del test_sparse

print('GBDT for feature selection, training...')
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
clf_fs = lgb.LGBMClassifier(boosting_type='gbdt',n_estimators=1000, objective='binary',subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                         learning_rate=0.05, min_child_weight=50, random_state=522, n_jobs=-1).fit(train_sparse, y_)

print('GBDT feature selection in progress...')
model = SelectFromModel(clf_fs, threshold = '0.8*mean',prefit=True)
train_sparse_fs = model.transform(train_sparse)
test_sparse_fs = model.transform(test_sparse)
del train_sparse
del test_sparse


print('Feature selection finished')
print('train_sparse_fs shape is %s' %(str(train_sparse_fs.shape)))

# GBDT

print('gbdt preparing')
X_train, X_test, y_train, y_test = train_test_split(train_sparse_fs, y_, test_size=0.2,random_state = 522)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

param = {'num_leaves':31,'num_boost_round':4000, 'objective':'binary','metric':'auc','boosting':'goss',
        'num_threads':6}

bst = lgb.train(param, train_data, valid_sets = test_data, early_stopping_rounds = 500)
path = '/home/riren/Documents/4fun/models/gbdt_model_' + str(round(time.time())) + '.txt'
bst.save_model(path)
print('gbdt model saved at %s' %(path))

# ### Feature Encoding + LR
# print('leaf predicting...')
# train_leaf = bst.predict(X_train, num_iteration=bst.best_iteration,pred_leaf = True)
# test_leaf = bst.predict(X_test, num_iteration=bst.best_iteration,pred_leaf = True)
# pred_leaf = bst.predict(test_sparse_fs, num_iteration=bst.best_iteration,pred_leaf = True)

# print('leaf encoding...')
# enc = OneHotEncoder()
# X_train_leaf = enc.fit_transform(train_leaf)
# X_test_leaf = enc.transform(test_leaf)
# X_pred_leaf = enc.transform(pred_leaf)

# print('lr training...')
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(solver='saga')
# lr.fit(X_train_leaf,y_train)

# print('lr testing...')
# y_tesr_lr_pred = lr.predict_proba(X_test_leaf)[:,1]
# auc = roc_auc_score(y_test, y_tesr_lr_pred)
# print('roc: %d' %(auc))

# print('lr predicting...')
# res = test[['aid','uid']]
# ypred = lr.predict_proba(X_pred_leaf)[:,1]
# res['score'] = ypred

print('gbdt predicting...')
res = test[['aid','uid']]
ypred = bst.predict(test_sparse_fs)
res['score'] = ypred

print('results storing...')
outpath = '/home/riren/Documents/4fun/results/result_' + str(round(time.time())) + '.csv'
res.to_csv(outpath, index=False)
print('results stored at ' + outpath)

print('END')

```
