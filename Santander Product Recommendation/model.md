
# import libraries and functions


```python
import csv
import datetime
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, ensemble
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from getTrain import *
from getPredictData import *
from preprocessFunction import *
```

# data

# train
- ### get the train_all, train_null data and save them


```python
data_path="/media/chen/0009C4D1000728AE/Santander Product Recommendation/"
f=open(data_path+"train_ver2.csv")
null_x_1,null_y_1=get_train_null(f)
```


```python
data_path="/media/chen/0009C4D1000728AE/Santander Product Recommendation/"

f=open(data_path+"train_ver2.csv")
null_x_1,null_y_1=get_train_null(f)
np.savez("null_all.npz",(null_x_1,null_y_1))

f=open(data_path+"train_ver2.csv")
train_x_1,train_y_1=processData_train(f)
f.close()
np.savez("lessMore.npz",(train_x_1,train_y_1))
```

- ### load the train data


```python
daLoad=np.load("lessMore.npz")['arr_0']
train_x_1,train_y_1=daLoad[0],daLoad[1]
```

- ### split train_all data into two parts


```python
null_x,null_y,have_x,have_y=[],[],[],[]
for m,n in zip(train_x_1,train_y_1):
    if sum(m[18:])==0:
        for i,j in enumerate(n):
            if j>0:
                null_x.append(m)
                null_y.append(i)
    else:
        for i,j in enumerate(n):
            if j>0:
                have_x.append(m)
                have_y.append(i) 
```

- ###  preprocess the train data


```python
Tr_x,Tr_y=[],[]
for m,n in zip(train_x_1,train_y_1):
    for i,j in enumerate(n):
        if j>0:
            Tr_x.append(m)
            Tr_y.append(i)
```

# predict
- ### get the predict data and save


```python
f1=open(data_path+"train_ver2.csv")
cust_dict_5=processData_test_1(f1)
f1.close()
f2=open(data_path+"test_ver2.csv")
test_x_1,test_y_1=processData_test_2(f2,cust_dict_5)
f2.close()
np.savez("predict_0628.npz",(test_x_1,test_y_1))
```

- ### load the data


```python
arr=np.load("predict_0628.npz")
test_x_1,test_y_1=arr['arr_0'][0],arr['arr_0'][1]
```

- ### split predict_all data into two parts


```python
null_tx,null_ty,have_tx,have_ty=[],[],[],[]
for m,n in zip(test_x_1,test_y_1):
    if sum(m[18:])==0:
        for i,j in enumerate(n):
            if j>0:
                null_tx.append(m)
                null_ty.append(i)
    else:
        for i,j in enumerate(n):
            if j>0:
                have_tx.append(m)
                have_ty.append(i)        
```

# model
- ### xgboost of train_all


```python
def model_xgboost():
	param = {}
	param['booster']='gbtree'
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.05
	param['max_depth'] = 8
	param['silent'] = 1
	param['num_class'] = 24
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 2
	param['subsample'] = 1
	param['colsample_bytree'] = 0.9
	param['seed'] = 0
	param['nthread']=8    
	#param['alpha']=0.2
	#param['lambda']=0.5
	num_rounds = 110
	global target_cols
	plst = list(param.items())
	return plst
```

- ### cv to find the best n_estimators


```python
model=xgb.cv(model_xgboost(),xgb.DMatrix(Tr_x,label=Tr_y),num_boost_round=100,nfold=5,early_stopping_rounds=10,seed=0)
```

- ### gridsearch to find the best parameters


```python
param={'max_depth':range(3,10,2),'min_child_weight':range(1,5,2)}
scv=GridSearchCV(XGBClassifier(max_depth=6,min_child_weight=3,n_estimators=29,
                               learning_rate=0.3,subsample=1,colsample_bytree=0.9,objective='multi:softprob'),param_grid=param,
                 cv=5,n_jobs=4)
import datetime
begin=datetime.datetime.now()
scv.fit(np.array(Tr_x),np.array(Tr_y))
print datetime.datetime.now()-begin
scv.get_params,scv.grid_scores_
```


```python
xgtrain=xgb.DMatrix(null_x,label=null_y)
params=model_xgboost()
from sklearn.cross_validation import train_test_split
tr_x,te_x,tr_y,te_y=train_test_split(have_x,have_y,train_size=0.8)
xgtrain,xgtest=xgb.DMatrix(tr_x,label=tr_y),xgb.DMatrix(te_x,label=te_y)
model=xgb.train(params,xgtrain,210,evals=[(xgtrain,'train'),(xgtest,'test')],early_stopping_rounds=10)
```

# null and have


```python
null_tx,null_ty,have_tx,have_ty=[],[],[],[]
for m,n in zip(null_x_1,null_y_1):
        for i,j in enumerate(n):
            if j>0:
                null_tx.append(m)
                null_ty.append(i)
```

# xgboost of null and have
- # null


```python
def model_xgboost():
	param = {}
	param['booster']='gbtree'
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.1
	param['max_depth'] = 8
	param['silent'] = 1
	param['num_class'] = 24
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 2
	param['subsample'] = 0.9
	param['colsample_bytree'] = 0.9
	param['seed'] = 0
	param['nthread']=8    
	#param['alpha']=0.2
	#param['lambda']=0.8
	num_rounds = 110
	global target_cols
	plst = list(param.items())
	return plst
```


```python
tr_x,te_x,tr_y,te_y=cross_validation.train_test_split(null_tx,null_ty,train_size=0.8,random_state=0)
xgtrain=xgb.DMatrix(tr_x,label=tr_y)
xgtest=xgb.DMatrix(te_x,label=te_y)
model_null=xgb.train(model_xgboost(),xgtrain,150,evals=[(xgtrain,'train'),(xgtest,'test')],
                     evals_result={'eval_metric':'auc'},early_stopping_rounds=100)
```

#### grid_search to find the best number


```python
param={"max_depth":range(12,15,1)}
scv=GridSearchCV(XGBClassifier(max_depth=12,min_child_weight=1,learning_rate=0.1,n_estimators=60,subsample=1,
                               colsample_bytree=0.9,objective="multi:softprob"),param_grid=param,cv=5,n_jobs=-1)
```

- # have


```python
def model_xgboost():
	param = {}
	param['booster']='gbtree'
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.1
	param['max_depth'] = 6
	param['silent'] = 1
	param['num_class'] = 24
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 2
	param['subsample'] = 0.9
	param['colsample_bytree'] = 0.7
	param['seed'] = 0
	param['nthread']=8    
	#param['alpha']=0.2
	#param['lambda']=0.8
	num_rounds = 110
	global target_cols
	plst = list(param.items())
	return plst
```

tr_x,te_x,tr_y,te_y=cross_validation.train_test_split(have_x,have_y,train_size=0.8,random_state=0)
xgtrain=xgb.DMatrix(tr_x,label=tr_y)
xgtest=xgb.DMatrix(te_x,label=te_y)
model_have=xgb.train(model_xgboost(),xgtrain,150,evals=[(xgtrain,'train'),(xgtest,'test')],early_stopping_rounds=100)
model_have.save_model("model_have_1")


```python
# load the model
model_have=xgb.Booster(model_file='model_have_1')
```

- ### grid_search to find the best parameters


```python
param={'max_depth':range(3,9,2),'min_child_weight':range(1,5,2)}
scv=GridSearchCV(XGBClassifier(max_depth=6,min_child_weight=3,n_estimators=56,learning_rate=0.3,subsample=1,
                               colsample_bytree=0.9,objective='multi:softprob'),param_grid=param,cv=5,n_jobs=4)
```

# predict
- ## separate the null and have to predict the result and merge


```python
res=model_null.predict(xgb.DMatrix(np.array(test_x_1)[:,:18]))
preds=res
print("Getting the top products..")
target_cols = np.array(target_cols)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:8]

test_id = np.array(pd.read_csv(data_path+"test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
res_null=[]
for i in range(len(test_x_1)):
    if sum(test_x_1[i][18:])==0:
        res_null.append([test_id[i],final_preds[i]])
```

    Getting the top products..



```python
res=model_have.predict(xgb.DMatrix(test_x_1))
preds=res
print("Getting the top products..")
target_cols = np.array(target_cols)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:8]
test_id = np.array(pd.read_csv(data_path+"test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
res_have=[]
for i in range(len(test_x_1)):
    if sum(test_x_1[i][18:])!=0:
        res_have.append([test_id[i],final_preds[i]])
```

    Getting the top products..



```python
res_predict=res_null+res_have
res_fin=np.array(res_predict)
out_df = pd.DataFrame({'ncodpers':res_fin[:,0], 'added_products':res_fin[:,1]})
out_df.to_csv(data_path+'sub_xgb_lessMore_merge_null_2_v109.csv', index=False)
```

 - ## model_all to predict the result


```python
	res=model.predict(xgb.DMatrix(test_x_1))
	preds=res
	print("Getting the top products..")
	target_cols = np.array(target_cols)
	preds = np.argsort(preds, axis=1)
	preds = np.fliplr(preds)[:,:8]
	test_id = np.array(pd.read_csv(data_path+"test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
	final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
	out_df=pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
	out_df.to_csv(data_path+'sub_xgb_lessMore_merge_over11.csv', index=False)


```

    Getting the top products..



```python

```
