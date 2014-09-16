# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#%matplotlib inline

import numpy as np
import pandas as pd
from sklearn.linear_model import Lars,Ridge, Lasso, SGDClassifier,SGDRegressor,LogisticRegression,BayesianRidge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# <codecell>

import inspect
import os
import sys
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../kaggle-fire/python")

sys.path.append(code_path)

import xgboost as xgb

# <codecell>


# <codecell>



def weighted_gini(act,pred,weight):
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight})
    df = df.sort('pred',ascending=False)
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    n = df.shape[0]
    #df["gini"] = (df.lorentz - df.random) * df.weight
    #return df.gini.sum()
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

def normalized_weighted_gini(act,pred,weight):
    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)
def xgb_train_predict(data,label,test):
    xgmat = xgb.DMatrix( data, label=label,missing=-999)
    test_size = test.shape[0]
    param = {}
    param['objective'] = 'binary:logitraw'
    weight = data * float(test_size) / len(label)

    sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] != 0  )
    sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0  )
   # param['scale_pos_weight'] = sum_wneg/sum_wpos
   # param['booster_type']=1
    param['bst:eta'] = 0.2
    param['bst:max_depth'] = 2
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['nthread'] = 16
    plst = list(param.items())#+[('eval_metric', 'ams@0.15')]

    watchlist = [ (xgmat,'train') ]
    num_round = 55  # 48 is good
   # print ('loading data end, start to boost trees')
    bst = xgb.train( plst, xgmat, num_round, watchlist );
    # save out model
    #bst.save_model('higgs.model')
    #modelfile = 'higgs.model'
    
    xgmat = xgb.DMatrix(test,missing=-999)
    #bst = xgb.Booster({'nthread':16})
    #bst.load_model( modelfile )
    ypred = bst.predict( xgmat )
    return ypred
# good: 0.2,2,40 -> 0.32
# <codecell>

di='../../data/'
print 'start reading data'
train = pd.read_csv(di+'train.csv')


# <markdowncell>

# # clean data

# <codecell>

vvar=[]
for i in range(1,10):
    vvar.append('var'+str(i))
# get categorical feature
vv=train[vvar]

# <codecell>

allv={}
rc=[]
for i in range(1,10):
    col='var'+str(i)
    allv[col]= np.unique(vv[col])
    for j in allv[col]:
     
        xx=(vv[col][vv[col]==j].shape[0])*1.0/(vv.shape[0])
        # only keep feature that has less than 50% Z
        if xx<0.1 and j=='Z':
            print col,'percentage of Z',xx
            rc.append(col)

# <codecell>

vr=vv[rc] # just keep var 4,7,8,9

# <codecell>


# <codecell>

from sklearn import feature_extraction
def one_hot_dataframe(data, cols, replace=False):
    vec = feature_extraction.DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    #vecData = pd.DataFrame(vec.fit_transform(data[cols].to_dict(outtype='records')).toarray())
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData)

# <codecell>

# fast one-step encoder
vr, vr_n = one_hot_dataframe(vr,rc, replace=True)

# <codecell>

weather=[]
for i in range(1,237):
    weather.append('weatherVar'+str(i))
wvar=np.array(train[weather])
for i,c in enumerate(wvar.T):
    c[np.isnan(c)]=np.mean(c[~np.isnan(c)])
    wvar[:,i]=c

# <codecell>

geo=[]
for i in range(1,38):
    geo.append('geodemVar'+str(i))
gvar=np.array(train[geo])
for i,c in enumerate(gvar.T):
    c[np.isnan(c)]=np.mean(c[~np.isnan(c)])
    gvar[:,i]=c

# <codecell>

# get var 10 ~ var 17 and crimevar
var=[]
for i in range(8):
    var.append('var'+str(i+10))

crime=[]
for i in range(9):
    crime.append('crimeVar'+str(i+1))

t_var=np.array(train[var])
for i,c in enumerate(t_var.T):
    c[np.isnan(c)]=np.mean(c[~np.isnan(c)])
    t_var[:,i]=c


tc_var=np.array(train[crime])
for i,c in enumerate(tc_var.T):
    c[np.isnan(c)]=np.mean(c[~np.isnan(c)])
    tc_var[:,i]=c

# <codecell>

#Xtrain = np.concatenate([t_var[:,[0]+range(2,8)],vr,wvar[:,102:103],wvar[:,152:153],gvar[:,30:32]], axis = 1)
#Xtrain = np.concatenate([t_var[:,[0]+range(2,8)],vr,wvar[:,102:103],wvar[:,152:153],gvar[:,30:32]], axis = 1) # good sub-xgb-wg
Xtrain = np.concatenate([t_var[:,[0]+range(2,8)],vr,wvar[:,[102,152]],gvar[:,[30,31]]], axis = 1)
#Xtrain = np.concatenate([t_var[:,[0]+range(2,8)],vr], axis = 1)
Wtrain=t_var[:,1]

# <codecell>

Xtrain=np.hstack((Xtrain,np.matrix(Wtrain).T))
ytrain=train['target']


test = pd.read_csv(di+'test.csv')
sample = pd.read_csv(di+'sampleSubmission.csv')
ts_var=np.array(test[var])
for i,c in enumerate(ts_var.T):
    c[np.isnan(c)]=np.mean(c[~np.isnan(c)])
    ts_var[:,i]=c
tcs_var=np.array(test[crime])
for i,c in enumerate(tcs_var.T):
    c[np.isnan(c)]=np.mean(c[~np.isnan(c)])
    tcs_var[:,i]=c

wsvar=np.array(test[weather])
for i,c in enumerate(wsvar.T):
    c[np.isnan(c)]=np.mean(c[~np.isnan(c)])
    wsvar[:,i]=c

gsvar=np.array(test[geo])
for i,c in enumerate(gsvar.T):
    c[np.isnan(c)]=np.mean(c[~np.isnan(c)])
    gsvar[:,i]=c
    
vt=test[rc]
vt, vt_n = one_hot_dataframe(vt,rc, replace=True) 

#Xtest = np.concatenate([ts_var[:,[0]+range(2,8)],vt,wsvar[:,102:103],wsvar[:,152:153],gsvar[:,30:32],gsvar[:,0:1]], axis = 1)
Xtest = np.concatenate([ts_var[:,[0]+range(2,8)],vt,wsvar[:,[102,152]],gsvar[:,[30,31]]], axis = 1)
#the setting above is for submission-sk-xgb-wg
#Xtest = np.concatenate([ts_var[:,[0]+range(2,8)],vt,tcs_var[:,-2:-1],wsvar[:,102:103],wsvar[:,152:153],wsvar[:,91:92],wsvar[:,166:167],gsvar[:,30:32],gsvar[:,0:1]], axis = 1)
#Xtest = np.concatenate([ts_var[:,[0]+range(2,8)],vt,wsvar[:,102:103],wsvar[:,152:153]], axis = 1)
Wtest=ts_var[:,1]

#Xtest=scale(Xtest)
Xtest=np.hstack((Xtest,np.matrix(Wtest).T))


yp=xgb_train_predict(Xtrain[:,:-1],np.array(ytrain),Xtest[:,:-1])
sample['target'] = yp
print 'write back'

#sample.to_csv('submission-xgb-wg4-28-log-rounds140.csv', index = False)
sample.to_csv('submission-test.csv', index = False)



