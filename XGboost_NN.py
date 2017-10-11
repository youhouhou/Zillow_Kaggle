# Parameters
FUDGE_FACTOR = 1.1200  # Multiply forecasts by this

XGB_WEIGHT = 0.6200
BASELINE_WEIGHT = 0.0100
OLS_WEIGHT = 0.0620
NN_WEIGHT = 0.0800

XGB1_WEIGHT = 0.8000  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


#Import Lib#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as  xgb
import random
import lightgbm as lgb
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

#Load data#
load_s_t = time.time()

train_2017 = pd.read_csv('train_2017.csv',parse_dates = ["transactiondate"])
train_2016 = pd.read_csv('train_2016_v2.csv',parse_dates = ["transactiondate"])
frames = [train_2016,train_2017]
train = pd.concat(frames)
del train_2017
del train_2016
gc.collect()

properties = pd.read_csv('properties_2016.csv')

####preprocessing for properties###
for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)

for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))



print('loading data -- %s seconds--' % (time.time()- load_s_t))


#####feature selection!####
def change_date(df):
    df["transaction_month"] = df["transactiondate"].dt.month
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df

train = change_date(train)
train = train.merge(properties, how='left', on='parcelid')
#eliminate properties who have too much missing values
missing_perc_thresh = 0.97
exclude_missing = []
num_rows = train.shape[0]
for c in train.columns:
    num_missing = train[c].isnull().sum()
    missing_ratio = num_missing / num_rows
    if missing_ratio > missing_perc_thresh:
        exclude_missing.append(c)
#eliminate properties that has only one value
exclude_unique = []
for c in train.columns:
    num_unique = len(train[c].unique())
    if train[c].isnull().sum() != 0:
        num_unique -= 1 # eliminate NaN as a type
    if num_unique == 1:
        exclude_unique.append(c)
#define training feature
exclude_other = ['parcelid', 'logerror','propertyzoningdesc']
train_features = []
for c in train.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)



################
################
##  LightGBM  ##
################
################


print( "\nProcessing data for LightGBM ..." )

x_train = train.loc[:,train_features]
x_train.fillna(x_train.median(),inplace = True)
y_train = train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)


x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)

##### RUN LIGHTGBM
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

np.random.seed(0)
random.seed(0)

lgmb_train_st = time.time()
print("\nFitting LightGBM model ...")
clf = lgb.train(params, d_train, 430)
print('-- LightGBM training time %s seconds --' % (time.time()-lgmb_train_st))

del d_train; gc.collect()
del x_train; gc.collect()

print("\nPrepare for LightGBM prediction ...")
print("   Read sample file ...")
sample = pd.read_csv('sample_submission.csv')
print("   ...")
sample['parcelid'] = sample['ParcelId']
sample["transactiondate"] = pd.to_datetime('2016-11-15') # put dummy date in sample data
sample = change_date(sample)
print("   Merge with property data ...")
df_test = sample.merge(properties, on='parcelid', how='left')
print("   ...")
del sample; gc.collect()
print("   ...")
x_test = df_test[train_columns]
print("   ...")
del df_test; gc.collect()
print("   Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
print("   ...")
x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart LightGBM prediction ...")
lgbm_pridict_st = time.time()
p_test = clf.predict(x_test)
print('-- LightGBM predicting time %s seconds --' % (time.time()-lgmb_train_st))

del x_test; gc.collect()

print( "\nUnadjusted LightGBM predictions:" )
print( pd.DataFrame(p_test).head() )

################
################
##  XGBoost   ##
################
################

##### PROCESS DATA FOR XGBOOST

print( "\nProcessing data for XGBoost ...")
#remove out liers
train_df=train[train.logerror > -0.4]
train_df=train[train.logerror < 0.419]
x_train = train_df.loc[:,train_features]
y_train = train_df["logerror"].values
y_mean = np.mean(y_train)

train_columns = x_train.columns

sample = pd.read_csv('sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
sample["transactiondate"] = pd.to_datetime('2016-11-15')
sample = change_date(sample)
df_test = sample.merge(properties, on='parcelid', how='left')
del sample; gc.collect()

x_test = df_test[train_columns]

##### RUN XGBOOST

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 250
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
xgb_train_st = time.time()
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
print('--XGB first training time %s seconds--' % (time.time() - xgb_train_st))

print( "\nPredicting with XGBoost ...")

xgb_pred1 = model.predict(dtest)
xgb_predict_st = time.time()
print( "\nFirst XGBoost predictions:" )
print( pd.DataFrame(xgb_pred1).head() )
print('--XGB first predict time %s seconds--' % (time.time() - xgb_predict_st))

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

num_boost_rounds = 150
print("num_boost_rounds="+str(num_boost_rounds))

print( "\nTraining XGBoost again ...")
xgb_train_st = time.time()
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
print('--XGB second training time %s seconds--' % (time.time() - xgb_train_st))

print( "\nPredicting with XGBoost again ...")
xgb_predict_st = time.time()
xgb_pred2 = model.predict(dtest)
print('--XGB second predict time %s seconds--' % (time.time() - xgb_predict_st))

print( "\nSecond XGBoost predictions:" )
print( pd.DataFrame(xgb_pred2).head() )


##### COMBINE XGBOOST RESULTS
xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
#xgb_pred = xgb_pred1

print( "\nCombined XGBoost predictions:" )
print( pd.DataFrame(xgb_pred).head() )

del train_df
del x_train
del x_test
del dtest
del dtrain
del xgb_pred1
del xgb_pred2
gc.collect()

######################
######################
##  Neural Network  ##
######################
######################

# Read in data for neural network
print( "\n\nProcessing data for Neural Network ...")
print('\nLoading train, prop and sample data...')
sample = pd.read_csv('sample_submission.csv')
sample["transactiondate"] = pd.to_datetime('2016-11-15')
sample = change_date(sample)

print('Creating x_train and y_train from df_train...' )
x_train = train.loc[:,train_features]
y_train = train["logerror"]
y_mean = np.mean(y_train)
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)


print('Creating df_test...')
sample['parcelid'] = sample['ParcelId']
sample["transactiondate"] = pd.to_datetime('2016-11-15')
sample = change_date(sample)
print("Merging Sample with property data...")
df_test = sample.merge(properties, on='parcelid', how='left')
x_test = df_test[train_columns]

print('Shape of x_test:', x_test.shape)
print("Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)


## Preprocessing
print("\nPreprocessing neural network data...")
imputer= Imputer()
imputer.fit(x_train.iloc[:, :])
x_train = imputer.transform(x_train.iloc[:, :])
imputer.fit(x_test.iloc[:, :])
x_test = imputer.transform(x_test.iloc[:, :])

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

len_x=int(x_train.shape[1])
print("len_x is:",len_x)

print("\nSetting up neural network model...")
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units = 26, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))

print("\nFitting neural network model...")
nn_train_st = time.time()
nn.fit(np.array(x_train), np.array(y_train), batch_size = 32, epochs = 70, verbose=2)
print('--NN training time %s seconds' % (time.time() - nn_train_st))

print("\nPredicting with neural network model...")
#print("x_test.shape:",x_test.shape)
nn_pred_st = time.time()
y_pred_ann = nn.predict(x_test)
print('--NN predict time %s seconds' % (time.time() - nn_pred_st))
#transaction dateprint( "\nPreparing results for write..." )
nn_pred = y_pred_ann.flatten()
print( "Type of nn_pred is ", type(nn_pred) )
print( "Shape of nn_pred is ", nn_pred.shape )

# Cleanup
del sample
del x_train
del x_test
del df_train
del df_test
del y_pred_ann
gc.collect()

################
################
##    OLS     ##
################
################

np.random.seed(17)
random.seed(17)

print( "\n\nProcessing data for OLS ...")
sample = pd.read_csv("sample_submission.csv")
sample['parcelid'] = sample['ParcelId']
sample["transactiondate"] = pd.to_datetime('2016-11-15')
sample = change_date(sample)

x_train = trian[train_features]
y = train['logerror'].values

train_columns = train.columns

print("Merging Sample with property data...")
df_test = sample.merge(properties, on='parcelid', how='left')
x_test = df_test[train_columns]

def MAE(y, ypred):
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

properties = [] #memory

print("\nFitting OLS...")
reg = LinearRegression(n_jobs=-1)
reg.fit(x_train, y); print('fit...')
print(MAE(y, reg.predict(x_train)))
train = [];  y = [] #memory

del sample
del properties
del train
del x_train
del x_test
gc.collect()

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']


########################
########################
##  Combine and Save  ##
########################
########################

##### COMBINE PREDICTIONS

print( "\nCombining XGBoost, LightGBM, NN, and baseline predicitons ..." )
lgb_weight = 1 - XGB_WEIGHT - BASELINE_WEIGHT - NN_WEIGHT - OLS_WEIGHT
lgb_weight0 = lgb_weight / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
nn_weight0 = NN_WEIGHT / (1 - OLS_WEIGHT)
pred0 = 0
pred0 += xgb_weight0*xgb_pred
pred0 += baseline_weight0*BASELINE_PRED
pred0 += lgb_weight0*p_test
pred0 += nn_weight0*nn_pred

print( "\nPredicting with OLS and combining with XGB/LGB/NN/baseline predicitons: ..." )
for i in range(len(test_dates)):
    test['transactiondate'] = test_dates[i]
    pred = FUDGE_FACTOR * ( OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0 )
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

print( "\nWriting results to disk ..." )
submission.to_csv('1003_try.csv', index=False , float_format='%.4f')
print( "\nFinished ...")
