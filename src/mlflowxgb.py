#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib as mpl

import mlflow 
import pandas as pd
import numpy as np

import argparse


import xgboost as xgb
import matplotlib as mpl

import mlflow
import mlflow.xgboost

import time
from sklearn.metrics import roc_auc_score



from constants import  *
from ..mlflow_env import *


# In[2]:


def plot_one_prediction(df_pred,label="",xl=[-10,10],yl=[-10,10]):#xl=[3.5,4.3],yl=[3.5,4.3]):
    plt.rc('font', size=8)
    sel=(df_pred.x<xl[1]) & (df_pred.x>xl[0])&(df_pred.y<yl[1]) & (df_pred.y>yl[0])  
    xx=df_pred[sel]["x"]
    yy=df_pred[sel]["y"]
    H, xedges, yedges = np.histogram2d(df_pred[sel]["x"],df_pred[sel]["y"], bins=500)
    H = H.T
    plt.pcolormesh(xedges, yedges, H, cmap=plt.cm.jet, norm=mpl.colors.LogNorm())
    #xl=[df_pred[sel]["x"].min(),df_pred[sel]["x"].max()]
    #yl=[df_pred[sel]["y"].min(),df_pred[sel]["y"].max()]

    plt.xlim(xl[0], xl[1])
    plt.ylim(yl[0], yl[1])
    score=round(r2_score(xx, yy,multioutput='variance_weighted'),2)
    plt.title(label+f"score={score} Np="+str(len(df_pred[sel]["x"])))
    plt.xlabel("y")
    plt.ylabel("$\^y$")
    plt.plot(xl,yl,color="black")
    plt.colorbar()
    fig_file=f"{label}.png"
    plt.savefig(fig_file)
    plt.show()
    plt.close()
    return fig_file

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return (rmse, mae, r2)


# TODO optimize for DASK
def fetch_data(fpath="../../gaiadr3/combined_trainingset2_with_xp_norm.ONE.parq"):
    #df=pd.read_csv(fpath).set_index("source_id")
    df=pd.read_parquet(fpath)
    return df

def okay_condition(trainingset, label="met50"):
    """
    Return a condition that defines an acceptable subset for training
    FA: nur metallicities mit guten uncertainties
    FA: für alle training data ausser VMP: ein [M/H] cut (wegen der grid boundary)
    FA: es heisst: keine white dwarfs
    """
    if label=="met50":
        return (trainingset["survey"] != b'WDs EDR3') & \
               ((trainingset["survey"] == b'VMPs') | (trainingset["met50"] > -2.)) & \
               (trainingset["met84"] - trainingset["met16"] < 0.5) & \
               (np.isfinite(trainingset["bp_coefficients_1"]))
    else:
        return (np.isfinite(trainingset["bp_coefficients_1"]))
    
def prepare_data(trainingset,label, random_state=0, frac=0.1, fillvalue=-9.99, 
                        cachepath="cache"):
    """
    Get the desired training and test data (given one label)
    """
    # Read the file
    # TODO make cache for each label 
    #trainingset = dd.read_parquet(trainfilepath)
    # Add the label if necessary:
    if label=='logdist50':
        trainingset['logdist50'] = np.log10(trainingset['dist50'])
    elif label=='logteff50':
        trainingset['logteff50'] = np.log10(trainingset['teff50'])
    else:
        pass
    # Determine the "okay" part of the training set
    ok  = okay_condition(trainingset, label)
    # Cut out the desired part of the training set and repartition
    # Currently no DASK
    trainingset = trainingset[ok][features + [label]].sample(frac=frac).fillna(fillvalue)#.persist()#repartition(npartitions=256)
    
    ## Make weights
    trainlabels   = trainingset[label]
    ## First the histogram
    hist, bin_edges = np.histogram(trainlabels, bins=30)
    bin_centres     = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ## Then convert this to a KDE
    kde                    = gaussian_kde(bin_centres, weights=hist, bw_method=bin_edges[1]-bin_edges[0])
    ## Then compute the weights
    trainingset["weights"] = 1.0/kde.evaluate(trainingset[label]) #trainingset[label].map_partitions(lambda x: 1. / kde.evaluate(x))
    
    #weights = trainingset["weights"]
  

    # Split dataframe into input and output columns
    X   = trainingset.drop(columns=[label])#.to_dask_array(lengths=True)
    #y   = trainingset[label]#.to_dask_array(lengths=True)
    # Splitting into train and test
    #train_X, test_X, train_y, test_y = train_test_split(X, y,
    #                  test_size = 0.2, random_state = 123)
    # Compute weights

   
    for icol in trainingset.columns:
        if trainingset[icol].dtype != np.dtype('float32'):
            print(icol,trainingset[icol].dtype)
            trainingset[icol]=trainingset[icol].astype("float32")
    
    return trainingset#train_X, test_X, train_y, test_y


def clean_data_by_nan(rawdata):
    sel=rawdata[features_mags[0]]> -9.0
    for col in features_mags[1:len(features_mags)]:
        sel=(rawdata[col]> -9.0) & sel
    return rawdata[sel]


# In[3]:

parser = argparse.ArgumentParser(description='Process SH values.')
parser.add_argument('integer', metavar='n', type=int, nargs='+',
        help="""
        an integer defining what to calculate: 
        0:logdist50,
        1:'av50',
        2:'logteff50',
        3:'logg50',
        4:'met50',
        5:'mass50'
""")

args = parser.parse_args()
#print(args.accumulate(args.integer))

ii=args.integer[0]
print(ii)

label = pred_vec[ii]
rawdata = fetch_data()
#rawdata=clean_data_by_nan(rawdata)
data=prepare_data(rawdata,label,random_state=42,frac=1.0)
print(len(data))
print(f"{round(data.memory_usage(index=True).sum()/1024/1024)}MB")
#del(rawdata)
print(f"{round(rawdata.memory_usage(index=True).sum()/1024/1024)}MB")
len(rawdata), len(rawdata[rawdata["survey"] == b'VMPs'])


# In[4]:


del(rawdata)


# In[6]:


# RUN:
# cd  D:\Users\arm2arm\Projects\jupyter\MLUtils\mlflow 
# mlflow ui
def train_sh23_model(data,params,label="target",mlflow_experiment_id=1):
    
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    weights=train["weights"]
    if label != "met50":
        weights=train["weights"]*0.0+1
        
    X_train = train.drop([label,"weights"], axis=1)
    X_test = test.drop([label,"weights"], axis=1)
    y_train = train[[label]]
    y_test = test[[label]]
    
    # enable auto logging
    mlflow.set_tracking_uri("http://192.168.111.203:5123")
    mlflow.xgboost.autolog()
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    dtest =  xgb.DMatrix(X_test, label=y_test)
   
   
    #xgbRegressor = xgb.XGBRegressor(
    #    **optimized_params
    #    )
    xgb.set_config(verbosity=1)
    with mlflow.start_run(experiment_id=mlflow_experiment_id,run_name=f"{label}_manual_squarederror_AKh"):
       
        model = xgb.train(params, dtrain, num_boost_round=1000,verbose_eval=True,
                #early_stopping_rounds=10,
                evals=[(dtrain, "train")])
        # Start prediction
        y_pred = model.predict(dtest)
        df_pred=pd.DataFrame({"x":y_test[label].values,"y":y_pred})
        dflims=df_pred.describe().T
        print(len(df_pred))
        img_fname=plot_one_prediction(df_pred,label)#,[dflims.loc["x"]["25%"]*0.2,dflims.loc["x"]["75%"]*1.8],[dflims.loc["y"]["25%"]*0.2,dflims.loc["y"]["75%"]*1.8])
        mlflow.log_artifact(img_fname)
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)
            
        y_pred = model.predict(dtrain)
        df_pred=pd.DataFrame({"x":y_train[label].values,"y":y_pred})
        dflims=df_pred.describe().T
        img_fname=plot_one_prediction(df_pred,label+"_train_")#,[dflims.loc["x"]["25%"]*0.2,dflims.loc["x"]["75%"]*1.8],[dflims.loc["y"]["25%"]*0.2,dflims.loc["y"]["75%"]*1.8])
        mlflow.log_artifact(img_fname)
    return model,X_train,X_test,y_train,y_test


# In[10]:


learning_rate=[0.1,0.05,0.005]
max_depth=[6,8,12]
min_child_weight=[4,6,8]

#params={
#"tree_method": "gpu_hist",
##'tree_method':'hist',
#"random_state":42,
#"learning_rate":0.05,
#"max_depth":8,
#"min_child_weight":4,
#"n_estimators":2000,
#"eval_metric":"rmsle",
#"objective":"reg:squarederror",
#"eval_metric":["mae","rmse"],
#"objective":"reg:pseudohubererror",
#"huber_slope":1.0,
#"subsample":0.8,
#"eta":0.5
#'num_boost_round':100
#'verbose_eval': False
#}

params={
        #"tree_method": "hist",
        "tree_method": "gpu_hist",
       # "tree_method": "exact",
        "random_state":42,
        'learning_rate': 0.01, 
        'max_depth': 16, 
        'min_child_weight': 1, 
        'subsample': 0.8,
        "objective":"reg:squarederror",
       # "objective":"reg:pseudohubererror",
        "eval_metric":["mae","rmse"],
       }

label = pred_vec[ii]
mlflow_experiment_id=654086611108235667
print(f"total datasets:{len(data)}")


# In[11]:


model,X_train,X_test,y_train,y_test=train_sh23_model(data,params,label, mlflow_experiment_id)
