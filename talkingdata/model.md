

```python
# Bag of apps categories
# Bag of labels categories
# Include phone brand and model device
print("Initialize libraries")
import pandas as pd
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import log_loss
from sklearn.cluster import DBSCAN
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.decomposition import PCA
import os
import gc
from scipy import sparse
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

#------------------------------------------------- Write functions ----------------------------------------

def rstr(df): return df.dtypes, df.head(3) ,df.apply(lambda x: [x.unique()]), df.apply(lambda x: [len(x.unique())]),df.shape

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

```

# preprocessing the data


```python
seed = 700
np.random.seed(seed)
datadir = 'D:\\talkingdata\\data\\'



# Data - Events data
# Bag of apps
print("# Read app events")
app_events = pd.read_csv(os.path.join(datadir,'app_events.csv'), dtype={'device_id' : np.str})
app_events.head(5)
app_events.info()
#print(rstr(app_events))

# remove duplicates(app_id)
app_events= app_events.groupby("event_id")["app_id"].apply(
    lambda x: " ".join(set("app_id:" + str(s) for s in x)))
app_events.head(5)

print("# Read Events")
events = pd.read_csv(os.path.join(datadir,'events.csv'), dtype={'device_id': np.str})
events.head(5)
events["app_id"] = events["event_id"].map(app_events)
events = events.dropna()
del app_events

events = events[["device_id", "app_id"]]
events.info()
# 1Gb reduced to 34 Mb

# remove duplicates(app_id)
events.loc[:,"device_id"].value_counts(ascending=True)

events = events.groupby("device_id")["app_id"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events = events.reset_index(name="app_id")

# expand to multiple rows
events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events.iterrows()]).reset_index()
events.columns = ['app_id', 'device_id']
events.head(5)
f3 = events[["device_id", "app_id"]]    # app_id

##################
#   App labels
##################

app_labels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
label_cat = pd.read_csv(os.path.join(datadir,'label_categories.csv'))
app_labels.info()
label_cat.info()
label_cat=label_cat[['label_id','category']]

app_labels=app_labels.merge(label_cat,on='label_id',how='left')
app_labels.head(3)
events.head(3)
#app_labels = app_labels.loc[app_labels.smaller_cat != "unknown_unknown"]

#app_labels = app_labels.groupby("app_id")["category"].apply(
#    lambda x: ";".join(set("app_cat:" + str(s) for s in x)))
app_labels = app_labels.groupby(["app_id","category"]).agg('size').reset_index()
app_labels = app_labels[['app_id','category']]


# Remove "app_id:" from column
print("## Handling events data for merging with app lables")
events['app_id'] = events['app_id'].map(lambda x : x.lstrip('app_id:'))
events['app_id'] = events['app_id'].astype(str)
app_labels['app_id'] = app_labels['app_id'].astype(str)
app_labels.info()

print("## Merge")

events= pd.merge(events, app_labels, on = 'app_id',how='left').astype(str)
#events['smaller_cat'].unique()

# expand to multiple rows
print("#Expand to multiple rows")
#events= pd.concat([pd.Series(row['device_id'], row['category'].split(';'))
#                    for _, row in events.iterrows()]).reset_index()
#events.columns = ['app_cat', 'device_id']
#events.head(5)
#print(events.info())

events= events.groupby(["device_id","category"]).agg('size').reset_index()
events= events[['device_id','category']]
events.head(10)
print("# App labels done")

f5 = events[["device_id", "category"]]    # app_id
# Can % total share be included as well?
print("# App category part formed")

##################
#   Phone Brand
##################

pbd = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'),
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)

##################
#  Train and Test
##################


train = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                    dtype={'device_id': np.str})
train.drop(["age", "gender"], axis=1, inplace=True)

test = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                   dtype={'device_id': np.str})
test["group"] = np.nan

split_len = len(train)

# Group Labels
Y = train["group"]
lable_group = LabelEncoder()
Y = lable_group.fit_transform(Y)
device_id = test["device_id"]

```

# additional process


```python

# Concat
Df = pd.concat((train, test), axis=0, ignore_index=True)

print("### ----- PART 4 ----- ###")

Df = pd.merge(Df, pbd, how="left", on="device_id")
Df["phone_brand"] = Df["phone_brand"].apply(lambda x: "phone_brand:" + str(x))
Df["device_model"] = Df["device_model"].apply(
    lambda x: "device_model:" + str(x))


###################
#  Concat Feature
###################

print("# Concat all features")

f1 = Df[["device_id", "phone_brand"]]   # phone_brand
f2 = Df[["device_id", "device_model"]]  # device_model

events = None
Df = None

f1.columns.values[1] = "feature"
f2.columns.values[1] = "feature"
f5.columns.values[1] = "feature"
f3.columns.values[1] = "feature"

FLS = pd.concat((f1, f2, f3, f5), axis=0, ignore_index=True)

FLS.info()

###################
# User-Item Feature
###################
print("# User-Item-Feature")

device_ids = FLS["device_id"].unique()
feature_cs = FLS["feature"].unique()

data = np.ones(len(FLS))
len(data)

dec = LabelEncoder().fit(FLS["device_id"])
row = dec.transform(FLS["device_id"])
col = LabelEncoder().fit_transform(FLS["feature"])
sparse_matrix = sparse.csr_matrix(
    (data, (row, col)), shape=(len(device_ids), len(feature_cs)))
sparse_matrix.shape
sys.getsizeof(sparse_matrix)

sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]
print("# Sparse matrix done")

del FLS
del data
f1 = [1]
f5 = [1]
f2 = [1]
f3 = [1]

events = [1]

##################
#      Data
##################

print("# Split data")
train_row = dec.transform(train["device_id"])
train_sp = sparse_matrix[train_row, :]

test_row = dec.transform(test["device_id"])
test_sp = sparse_matrix[test_row, :]
"""

```

### 缺失值的数量


```python
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
def rstr(df): return df.dtypes, df.head(3) ,df.apply(lambda x: [x.unique()]), df.apply(lambda x: [len(x.unique())]),df.shape

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

```


```python
save_sparse_csr("train_sp",train_sp)
save_sparse_csr("test_sp",test_sp)
np.save("train_Y",Y)
np.save("device_id",device_id)
```

# load the data


```python
print("Initialize libraries")
import pandas as pd`a

import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import log_loss
from sklearn.cluster import DBSCAN
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import LabelEncoder
#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
from sklearn import ensemble
from sklearn.decomposition import PCA
import os
import gc
from scipy import sparse
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss 
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
train_sp=load_sparse_csr("train_sp.npz")
test_sp=load_sparse_csr("test_sp.npz")
device_id=np.load("device_id.npy")
Y=np.load("train_Y.npy")
```

    Initialize libraries


    Using Theano backend.


### 导入数据


```python
ax1_train=train_sp.getnnz(axis=1)
ax1_test=test_sp.getnnz(axis=1)
```

### 随机森林


```python
#稀疏
X_train,X_val,y_train,y_val=train_test_split(train_sp_3,Y_3,train_size=0.7,random_state=10)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_jobs=2,n_estimators=100)
clf.fit(X_train,y_train)
```

    D:\Anaconda2\lib\site-packages\ipykernel\__main__.py:5: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=2,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
import sklearn.metrics as sm
y_predpro=clf.predict_proba(X_val)
y_pre=clf.predict(X_val)
print sm.log_loss(y_val,y_predpro)
print sm.classification_report(y_val,y_pre)
```

    4.08939264153
                 precision    recall  f1-score   support
    
              0       0.16      0.13      0.15      1072
              1       0.09      0.02      0.03       908
              2       0.10      0.01      0.02       623
              3       0.14      0.02      0.04       992
              4       0.11      0.03      0.05      1100
              5       0.11      0.04      0.06       842
              6       0.15      0.27      0.20      1646
              7       0.16      0.35      0.22      2121
              8       0.11      0.02      0.03      1174
              9       0.12      0.06      0.08      1510
             10       0.14      0.30      0.19      1824
             11       0.17      0.13      0.14      1595
    
    avg / total       0.14      0.15      0.12     15407
    



```python
X_train,X_val,y_train,y_val=train_test_split(train_sp_3,Y_3,train_size=0.8,random_state=0)
import sklearn.metrics as sm
y_predpro=clf.predict_proba(X_val)
y_pre=clf.predict(X_val)
print sm.log_loss(y_val,y_predpro)
print sm.classification_report(y_val,y_pre)
```

    2.78715062958
                 precision    recall  f1-score   support
    
              0       0.22      0.19      0.21       726
              1       0.18      0.04      0.06       615
              2       0.36      0.03      0.06       433
              3       0.31      0.04      0.08       668
              4       0.20      0.05      0.08       757
              5       0.21      0.07      0.11       533
              6       0.19      0.32      0.24      1122
              7       0.18      0.41      0.25      1330
              8       0.29      0.05      0.08       791
              9       0.17      0.09      0.12       907
             10       0.18      0.37      0.24      1278
             11       0.22      0.17      0.19      1111
    
    avg / total       0.21      0.19      0.16     10271
    


### 逻辑回归


```python
traini=np.array(range(74645))/74645.0
testi=np.array(range(112071))/112071.0
```


```python
from scipy.sparse import hstack
mde=device_id_train.astype("int64").values.reshape(74645,1)
st_deid=sparse.csr_matrix(traini.reshape(74645,1))
testsp=sparse.csr_matrix(testi.reshape(112071,1))
train_sp_m=hstack([train_sp,st_deid])
test_sp_m=hstack([test_sp,testsp])
```


```python
from sklearn.linear_model import LogisticRegression
#from 
train_sp_3=train_sp[train_sp.getnnz(axis=1)>3]
Y_3=pd.DataFrame(Y)[train_sp.getnnz(axis=1)>3]
#sam_x,sam_y=stratified_sample(train_sp_3,Y_3,5000)
train_sp_x3=train_sp[train_sp.getnnz(axis=1)<=3]
Y_x3=pd.DataFrame(Y)[train_sp.getnnz(axis=1)<=3]
```


```python
traini_x3=pd.DataFrame(traini)[train_sp.getnnz(axis=1)<=3]
traini_d3=pd.DataFrame(traini)[train_sp.getnnz(axis=1)>3]
sp_traini_x3=sparse.csr_matrix(traini_x3.values)
sp_traini_d3=sparse.csr_matrix(traini_d3.values)
sp_train=sparse.csr_matrix(traini.reshape(74645,1))
sp_test=sparse.csr_matrix(testi.reshape(112071,1))
```


```python
#print sam_x.shape,sam_y.shape,train_sp_3.shape,Y_3.shape
np.random.seed(700)
X_train,X_val,y_train,y_val=train_test_split(train_sp_x3,Y_x3,train_size=0.8,random_state=8)
clf_lg=LogisticRegression(penalty='l2',n_jobs=2,C=0.12,solver='lbfgs',multi_class='multinomial',
                          random_state=700)
                          #class_weight='balanced')
#best:C=0.12
clf_lg.fit(X_train,y_train)

import sklearn.metrics as sm
y_predpro=clf_lg.predict_proba(X_val)
y_pre=clf_lg.predict(X_val)
print sm.log_loss(y_val,y_predpro)
print sm.classification_report(y_val,y_pre)
```


```python
y_predpro=clf_lg.predict_proba(X_val[X_val.getnnz(axis=1)<=3])
y_pre=clf_lg.predict(X_val[X_val.getnnz(axis=1)<=3])
Y_x3=pd.DataFrame(y_val)[X_val.getnnz(axis=1)<=3]
print sm.log_loss(Y_x3,y_predpro)
print sm.classification_report(Y_x3,y_pre)
```

    2.40447893253
                 precision    recall  f1-score   support
    
              0       0.16      0.16      0.16       743
              1       0.25      0.00      0.01       630
              2       0.00      0.00      0.00       425
              3       0.00      0.00      0.00       654
              4       0.13      0.01      0.02       760
              5       0.00      0.00      0.00       608
              6       0.15      0.16      0.15      1125
              7       0.16      0.42      0.23      1340
              8       0.20      0.00      0.01       775
              9       0.17      0.02      0.03       981
             10       0.14      0.30      0.19      1213
             11       0.15      0.29      0.19      1017
    
    avg / total       0.14      0.15      0.11     10271
    


### xgboost


```python
import xgboost as xgb
from scipy.sparse import hstack
X_train,X_val,y_train,y_val=train_test_split(train_sp_x3,Y_x3,train_size=0.99,random_state=7)
params = {}
params['booster'] = 'gblinear'
params['objective'] = "multi:softprob"
params['eval_metric'] = 'merror'
params['eta'] = 0.01
params['num_class'] = 12
params['lambda'] = 13
params['alpha'] = 0
params['max_depth']=4
d_train=xgb.DMatrix(X_train,label=y_train)
d_valid=xgb.DMatrix(X_val,label=y_val)

watchlist = [(d_train, 'train'), (d_valid, 'eval')]
clf = xgb.train(params, d_train,100, watchlist, early_stopping_rounds=5)
```


```python
train_sp_x3=X_val[X_val.getnnz(axis=1)<=3]
y_x3=pd.DataFrame(y_val)[X_val.getnnz(axis=1)<=3]
X_valdm=xgb.DMatrix(train_sp_x3)
y_predpro1=clf.predict(X_valdm)
print sm.log_loss(y_x3.values,y_predpro1)
#2.36592422048
```

    2.46414522999


### 神经网络


```python
train_sp_x3=train_sp[train_sp.getnnz(axis=1)<=3]
Y_x3=pd.DataFrame(Y)[train_sp.getnnz(axis=1)<=3].values
X_train,X_val,y_train,y_val=train_test_split(train_sp_x3,Y_x3,train_size=0.8,random_state=8)
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.6))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
model=baseline_model()
fit= model.fit_generator(generator=batch_generator(X_train, y_train, 300, True), nb_epoch=10,
                         samples_per_epoch=20000,validation_data=(X_val.todense(), y_val), verbose=2,nb_worker=2
                         )
# evaluate the model
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
print('logloss val {}'.format(log_loss(y_val, scores_val)))
```

    Epoch 1/10
    8s - loss: 2.4754 - acc: 0.1167 - val_loss: 2.4616 - val_acc: 0.1288
    Epoch 2/10
    8s - loss: 2.4457 - acc: 0.1252 - val_loss: 2.4360 - val_acc: 0.1305
    Epoch 3/10
    8s - loss: 2.4348 - acc: 0.1304 - val_loss: 2.4339 - val_acc: 0.1305
    Epoch 4/10
    8s - loss: 2.4307 - acc: 0.1343 - val_loss: 2.4325 - val_acc: 0.1305
    Epoch 5/10
    8s - loss: 2.4294 - acc: 0.1326 - val_loss: 2.4312 - val_acc: 0.1304
    Epoch 6/10
    8s - loss: 2.4278 - acc: 0.1381 - val_loss: 2.4291 - val_acc: 0.1335
    Epoch 7/10
    8s - loss: 2.4220 - acc: 0.1373 - val_loss: 2.4267 - val_acc: 0.1336
    Epoch 8/10
    8s - loss: 2.4231 - acc: 0.1386 - val_loss: 2.4243 - val_acc: 0.1363
    Epoch 9/10
    8s - loss: 2.4183 - acc: 0.1428 - val_loss: 2.4220 - val_acc: 0.1370
    Epoch 10/10
    8s - loss: 2.4188 - acc: 0.1412 - val_loss: 2.4204 - val_acc: 0.1378
    logloss val 2.42037132673



```python
X_train,X_val,y_train,y_val=train_test_split(train_sp_3,Y_3,train_size=0.8,random_state=0)
import sklearn.metrics as sm
y_predpro=model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
y_pre=model.predict_classes(X_val.todense())
print sm.log_loss(y_val,y_predpro)
print sm.classification_report(y_val,y_pre)

```

    10271/10271 [==============================] - 11s    
    2.40429836739
                 precision    recall  f1-score   support
    
              0       0.17      0.14      0.15       726
              1       0.00      0.00      0.00       615
              2       0.00      0.00      0.00       433
              3       0.00      0.00      0.00       668
              4       0.00      0.00      0.00       757
              5       0.00      0.00      0.00       533
              6       0.15      0.03      0.05      1122
              7       0.15      0.55      0.23      1330
              8       0.00      0.00      0.00       791
              9       0.00      0.00      0.00       907
             10       0.14      0.52      0.23      1278
             11       0.00      0.00      0.00      1111
    
    avg / total       0.07      0.15      0.07     10271
    



```python
X_train, X_val, y_train, y_val = train_test_split(train_sp, Y, train_size=0.999, random_state=10)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model

model=baseline_model()

fit= model.fit_generator(generator=batch_generator(X_train, y_train, 400, True),
                         nb_epoch=10,
                         samples_per_epoch=69984,class_weight=dics,
                         validation_data=(X_val.todense(), y_val), verbose=2
                         )


```

# evaluate the model


```python

scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
print('logloss val {}'.format(log_loss(y_val, scores_val)))

print("# Final prediction")
scores = model.predict_generator(generator=batch_generatorp(test_sp, 800, False), val_samples=test_sp.shape[0])
result = pd.DataFrame(scores , columns=lable_group.classes_)
result["device_id"] = device_id
print(result.head(1))
result = result.set_index("device_id")

#result.to_csv('./sub_bagofapps7_keras_10_50_pt2_10epoch.csv', index=True, index_label='device_id')
#Drop out 0.2
#Validation 2.3017
result.to_csv('sub_bagofapps7_keras_150_pt4_50_pt2_15epoch_prelu_softmax.csv', index=True, index_label='device_id')

print("Done")
```


```python
train_sp_3=train_sp[train_sp.getnnz(axis=1)>3]
Y_3=pd.DataFrame(Y)[train_sp.getnnz(axis=1)>3].values
train_sp_3.shape,Y_3.shape
```




    ((23290, 21425), (23290L, 1L))



### 逻辑回归


```python
from sklearn.linear_model import LogisticRegression
X_train,X_val,y_train,y_val=train_test_split(train_sp_3,Y_3,train_size=0.95,random_state=3)
clf_lg=LogisticRegression(penalty='l2',n_jobs=2,C=0.01,solver='lbfgs',multi_class='multinomial',
                          random_state=100)

clf_lg.fit(X_train,y_train)
import sklearn.metrics as sm
y_predpro=clf_lg.predict_proba(X_val)
y_pre=clf_lg.predict(X_val)
print sm.log_loss(y_val,y_predpro)
print sm.classification_report(y_val,y_pre)
```

    D:\Anaconda2\lib\site-packages\sklearn\utils\validation.py:515: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)


    1.98360140366
                 precision    recall  f1-score   support
    
              0       0.33      0.50      0.39        62
              1       0.13      0.05      0.07        62
              2       0.33      0.02      0.04        49
              3       0.18      0.14      0.16        63
              4       0.24      0.25      0.25        83
              5       0.15      0.06      0.09        67
              6       0.39      0.48      0.43        88
              7       0.33      0.42      0.37       153
              8       0.17      0.02      0.04        85
              9       0.17      0.10      0.12       122
             10       0.26      0.34      0.30       168
             11       0.35      0.61      0.44       163
    
    avg / total       0.26      0.30      0.26      1165
    



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
X_train,X_val,y_train,y_val=train_test_split(train_sp_3,Y_3,train_size=0.7,random_state=3)
clf_rf=RandomForestClassifier()
clf_rf.fit(X_train,y_train)
model=SelectFromModel(clf_rf,prefit=True)

X_train_new=model.transform(X_train)
X_val_new=model.transform(X_val)
X_train.shape,X_train_new.shape
```

    D:\Anaconda2\lib\site-packages\ipykernel\__main__.py:5: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().



```python
import xgboost as xgb
params = {}
params['booster'] = 'gblinear'
params['objective'] = "multi:softprob"
params['eval_metric'] = 'mlogloss'
params['eta'] = 0.02
params['num_class'] = 12
params['lambda'] = 5
params['alpha'] = 3

X_train,X_val,y_train,y_val=train_test_split(train_sp_d3,Y_d3,train_size=0.7,random_state=3)
print("# Feature Selection")

d_train=xgb.DMatrix(X_train,label=y_train)
d_valid=xgb.DMatrix(X_val,label=y_val)

watchlist = [(d_train, 'train'), (d_valid, 'eval')]
clf = xgb.train(params, d_train, 1, watchlist, early_stopping_rounds=5)

```

    # Feature Selection
    [0]	train-mlogloss:2.36098	eval-mlogloss:2.36545
    Multiple eval metrics have been passed: 'eval-mlogloss' will be used for early stopping.
    
    Will train until eval-mlogloss hasn't improved in 5 rounds.



```python
X_train,X_val,y_train,y_val=train_test_split(train_sp_3,Y_3,train_size=0.8,random_state=0)
import sklearn.metrics as sm
y_predpro=clf_lg.predict_proba(X_val)
y_pre=clf_lg.predict(X_val)
print sm.log_loss(y_val,y_predpro)
print sm.classification_report(y_val,y_pre)
```

    2.29669698746
                 precision    recall  f1-score   support
    
              0       0.23      0.19      0.21       726
              1       0.28      0.03      0.05       615
              2       0.56      0.02      0.04       433
              3       0.34      0.02      0.04       668
              4       0.22      0.06      0.09       757
              5       0.21      0.06      0.10       533
              6       0.21      0.32      0.25      1122
              7       0.19      0.41      0.26      1330
              8       0.35      0.04      0.07       791
              9       0.18      0.09      0.12       907
             10       0.18      0.38      0.24      1278
             11       0.20      0.25      0.22      1111
    
    avg / total       0.24      0.20      0.17     10271
    


### 神经网络


```python
train_sp_d3=train_sp[train_sp.getnnz(axis=1)>3]
Y_d3=pd.DataFrame(Y)[train_sp.getnnz(axis=1)>3].values
train_sp_d3.shape,Y_d3.shape
```




    ((23290, 21425), (23290L, 1L))




```python
from keras.models import load_model
model_all=load_model("all_datafit.h5")
#scores_val_all = model_all.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
#print('logloss val {}'.format(log_loss(y_val, scores_val_all)))
```


```python
traini_x3=pd.DataFrame(traini)[train_sp.getnnz(axis=1)<=3]
traini_d3=pd.DataFrame(traini)[train_sp.getnnz(axis=1)>3]
sp_traini_x3=sparse.csr_matrix(traini_x3.values)
sp_traini_d3=sparse.csr_matrix(traini_d3.values)
```


```python
np.random.seed(700)
from keras.regularizers import l2, activity_l2
from keras.layers import MaxoutDense
from keras import optimizers
X_train,X_val,y_train,y_val=train_test_split(train_sp_d3,Y_d3,train_size=0.6,random_state=10)
def baseline_model():
    """model=Sequential()
    model.add(Dense(output_dim=1000, input_dim=X_train.shape[1], init='lecun_uniform', W_regularizer=l2(0.000025))) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.5))    
    model.add(Dense(50, init='lecun_uniform', W_regularizer=l2(0.000025))) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.4))       
    model.add(Dense(12, init='lecun_uniform'))
    model.add(Activation('softmax'))    
    opt = optimizers.Adagrad(lr=0.0035)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model"""
    model = Sequential()
    model.add(Dense(1000, input_dim=X_train.shape[1], init='lecun_uniform', W_regularizer=l2(0.000025)))
    model.add(Activation('relu')) 
    model.add(Dropout(0.50))
    model.add(Dense(1000, init='lecun_uniform', W_regularizer=l2(0.000025)))
    model.add(Activation('relu')) 
    model.add(Dropout(0.4))
    model.add(MaxoutDense(12, init='lecun_uniform'))
    model.add(Activation('softmax'))    
    opt = optimizers.Adagrad(lr=0.0035)
   # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
  #  model.add(Dense(12, init='normal', W_regularizer=l2(0.1),activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model
model=baseline_model()
# evaluate the model
fit= model.fit_generator(generator=batch_generator(X_train, y_train, 32, True), nb_epoch=30,
                         samples_per_epoch=19984,validation_data=(X_val.todense(), y_val), verbose=2)


#scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
#print('logloss val {}'.format(log_loss(y_val, scores_val)))

"""
Epoch 1/16
7s - loss: 2.4024 - acc: 0.1395 - val_loss: 2.3346 - val_acc: 0.1796
Epoch 2/16
7s - loss: 2.2871 - acc: 0.1899 - val_loss: 2.1870 - val_acc: 0.2364
Epoch 3/16
7s - loss: 2.1601 - acc: 0.2403 - val_loss: 2.0821 - val_acc: 0.2770
Epoch 4/16
7s - loss: 2.0820 - acc: 0.2647 - val_loss: 2.0329 - val_acc: 0.2884
Epoch 5/16
7s - loss: 2.0291 - acc: 0.2816 - val_loss: 2.0026 - val_acc: 0.2969
Epoch 6/16"""

```


```python
import xgboost as xgb
from scipy.sparse import hstack
X_train,X_val,y_train,y_val=train_test_split(hstack([train_sp_d3,sp_traini_d3]),Y_d3,train_size=0.6,random_state=10)
params = {}
params['booster'] = 'gbtree'
params['objective'] = "multi:softprob"
params['eval_metric'] = 'mlogloss'
params['eta'] = 0.6
params['num_class'] = 12
params['lambda'] = 5
params['alpha'] = 2
params['max_depth']=4
params['']
d_train=xgb.DMatrix(X_train,label=y_train)
d_valid=xgb.DMatrix(X_val,label=y_val)

watchlist = [(d_train, 'train'), (d_valid, 'eval')]
clf = xgb.train(params, d_train, 100, watchlist, early_stopping_rounds=5)
```


```python
model.save("p99_350_better.h5")
```


```python
X_train,X_val,y_train,y_val=train_test_split(hstack([train_sp_3,sp_traini_d3]),Y_3,train_size=0.99,random_state=10)
y_predpro=model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
y_pre=model.predict_classes(X_val.todense())
print sm.log_loss(y_val,y_predpro)
print sm.classification_report(y_val,y_pre)
```

    224/233 [===========================>..] - ETA: 0s1.85624476167
                 precision    recall  f1-score   support
    
              0       0.45      0.62      0.53        16
              1       0.20      0.08      0.11        13
              2       0.00      0.00      0.00         5
              3       0.36      0.21      0.27        19
              4       0.29      0.45      0.35        22
              5       0.00      0.00      0.00         8
              6       0.42      0.50      0.46        16
              7       0.26      0.28      0.27        32
              8       0.00      0.00      0.00        16
              9       0.17      0.16      0.16        19
             10       0.27      0.36      0.31        33
             11       0.40      0.53      0.46        34
    
    avg / total       0.27      0.32      0.29       233
    


    D:\Anaconda2\lib\site-packages\ipykernel\__main__.py:30: RuntimeWarning: divide by zero encountered in double_scalars


## 对test进行测试


```python
device_id.shape,test_sp.shape
#device_id[test_sp.getnnz(axis=1)<=3]
```




    ((112071L,), (112071, 21425))




```python
test_sp_ls3=test_sp[test_sp.getnnz(axis=1)<=3]
test_sp_mo3=test_sp[test_sp.getnnz(axis=1)>3]
test_sp_ls3.shape,test_sp_mo3.shape


testi_x3=pd.DataFrame(testi)[test_sp.getnnz(axis=1)<=3]
testi_d3=pd.DataFrame(testi)[test_sp.getnnz(axis=1)>3]
sp_testi_x3=sparse.csr_matrix(testi_x3.values)
sp_testi_d3=sparse.csr_matrix(testi_d3.values)
```


```python
hstack([test_sp_mo3,sp_testi_d3]).shape
```




    (35172, 21426)




```python
scores1=clf_lg.predict_proba(hstack([test_sp_ls3,sp_testi_x3]))
result1= pd.DataFrame(scores1 , columns=lable_group.classes_)
result1["device_id"] = device_id[test_sp.getnnz(axis=1)<=3]
```


```python
#xgboost:
X_valdm=xgb.DMatrix(hstack([test_sp_ls3,sp_testi_x3]))
scores1=clf.predict(X_valdm)
result1= pd.DataFrame(scores1 , columns=lable_group.classes_)
result1["device_id"] = device_id[test_sp.getnnz(axis=1)<=3]
```


```python
result1.to_csv("d:\\talkingdata\\data\\xgboost_ep6_labmbda1.csv",index=False)
result1=pd.read_csv("d:\\talkingdata\\data\\result_merge_result_ls3_c0p12.csv")
```


```python
scores2= model.predict_generator(generator=batch_generatorp(merge_two.tocsr(), 800, False), val_samples=test_sp_mo3.shape[0])
result2 = pd.DataFrame(scores2 , columns=lable_group.classes_)
result2["device_id"] = device_id[test_sp.getnnz(axis=1)>3]
result=result1.append(result2)

result.to_csv("d:\\talkingdata_result\\data\\result_merge_three_adjust_keras_lg_v3_.csv",index=False)
```


```python
df_best=pd.read_csv("d:\\talkingdata_result\\data\\result_merge_three_adjust_keras_lg_v3_.csv")
df_best_set=df_best.set_index("device_id")
keras_result2=df_best_set.drop(result1['device_id'].astype("int64").values)

keras_lg=keras_result2.reset_index().append(result1)
keras_lg.to_csv("d:\\talkingdata_result\\data\\keras_xgboost_v2.csv",index=False)
```
