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
properties = pd.read_csv('properties_2016.csv')
print('loading data -- %s seconds--' % (time.time()- load_s_t))

################
################
##  LightGBM  ##
################
################


print( "\nProcessing data for LightGBM ..." )
for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)

df_train = train.merge(properties, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

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
print("   Merge with property data ...")
df_test = sample.merge(properties, on='parcelid', how='left')
print("   ...")
del sample, properties; gc.collect()
print("   ...")
#df_test['Ratio_1'] = df_test['taxvaluedollarcnt']/df_test['taxamount']
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

print( "\nRe-reading properties file ...")
properties = pd.read_csv('properties_2016.csv')

##### PROCESS DATA FOR XGBOOST

print( "\nProcessing data for XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
dup_entry = train_df['parcelid'].value_counts().reset_index() #parcelid that been traded twice
dup_id = dup_entry.loc[dup_entry.iloc[:,1]>1]
train_df = train_df.drop(['transactiondate'],axis = 1)
for id in dup_id.iloc[0]:
    train_df['parcelid'] = train_df.loc[train_df['parcelid']==id,'logerror'].sum()/(train_df['parcelid']==id).sum()
x_test = properties.drop(['parcelid'], axis=1)


# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
train_df = train_df.drop_duplicates()
x_train = train_df.drop(['parcelid', 'logerror'], axis=1)
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

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
del properties
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
train_2017 = pd.read_csv('train_2017.csv',parse_dates = ["transactiondate"])
train_2016 = pd.read_csv('train_2016_v2.csv',parse_dates = ["transactiondate"])
frames = [train_2016,train_2017]
train = pd.concat(frames)
prop = pd.read_csv('properties_2016.csv')
sample = pd.read_csv('sample_submission.csv')

print('Fitting Label Encoder on properties...')
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))


print('Creating training set...')
df_train = train.merge(prop, how='left', on='parcelid')

df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train["transactiondate_year"] = df_train["transactiondate"].dt.year
df_train["transactiondate_month"] = df_train["transactiondate"].dt.month
df_train['transactiondate_quarter'] = df_train['transactiondate'].dt.quarter
df_train["transactiondate"] = df_train["transactiondate"].dt.day


print('Filling NA/NaN values...' )
df_train.fillna(-1.0)

print('Creating x_train and y_train from df_train...' )
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train["logerror"]
y_mean = np.mean(y_train)
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)


print('Creating df_test...')
sample['parcelid'] = sample['ParcelId']

print("Merging Sample with property data...")
df_test = sample.merge(prop, on='parcelid', how='left')

df_test["transactiondate"] = pd.to_datetime('2016-11-15')  # placeholder value for preliminary version
df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
df_test["transactiondate"] = df_test["transactiondate"].dt.day
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
del train
del prop
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

train_2017 = pd.read_csv('train_2017.csv',parse_dates = ["transactiondate"])
train_2016 = pd.read_csv('train_2016_v2.csv',parse_dates = ["transactiondate"])
frames = [train_2016,train_2017]
train = pd.concat(frames)
properties = pd.read_csv("properties_2016.csv")
submission = pd.read_csv("sample_submission.csv")


def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

def MAE(y, ypred):
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)


train = pd.merge(train, properties, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
properties = [] #memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col])

print("\nFitting OLS...")
reg = LinearRegression(n_jobs=-1)
reg.fit(train, y); print('fit...')
print(MAE(y, reg.predict(train)))
train = [];  y = [] #memory

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