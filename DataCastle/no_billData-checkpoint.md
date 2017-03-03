
# Table of Contents
 <p><div class="lev1 toc-item"><a href="#数据准备" data-toc-modified-id="数据准备-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>数据准备</a></div><div class="lev1 toc-item"><a href="#自定义函数" data-toc-modified-id="自定义函数-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>自定义函数</a></div><div class="lev1 toc-item"><a href="#原始数据读取" data-toc-modified-id="原始数据读取-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>原始数据读取</a></div><div class="lev1 toc-item"><a href="#各表处理" data-toc-modified-id="各表处理-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>各表处理</a></div><div class="lev2 toc-item"><a href="#user-与overdue表" data-toc-modified-id="user-与overdue表-41"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>user 与overdue表</a></div><div class="lev2 toc-item"><a href="#browse" data-toc-modified-id="browse-42"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>browse</a></div><div class="lev3 toc-item"><a href="#browse日期范围分布" data-toc-modified-id="browse日期范围分布-421"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>browse日期范围分布</a></div><div class="lev2 toc-item"><a href="#bank" data-toc-modified-id="bank-43"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>bank</a></div><div class="lev2 toc-item"><a href="#bill" data-toc-modified-id="bill-44"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>bill</a></div><div class="lev3 toc-item"><a href="#bill时间与loan时间" data-toc-modified-id="bill时间与loan时间-441"><span class="toc-item-num">4.4.1&nbsp;&nbsp;</span>bill时间与loan时间</a></div><div class="lev1 toc-item"><a href="#各表合并" data-toc-modified-id="各表合并-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>各表合并</a></div><div class="lev2 toc-item"><a href="#prepare-data" data-toc-modified-id="prepare-data-51"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>prepare data</a></div><div class="lev2 toc-item"><a href="#统计查看逾期的数据表的bill，browse以及bank表信息" data-toc-modified-id="统计查看逾期的数据表的bill，browse以及bank表信息-52"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>统计查看逾期的数据表的bill，browse以及bank表信息</a></div><div class="lev2 toc-item"><a href="#strategy--start" data-toc-modified-id="strategy--start-53"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>strategy -start</a></div><div class="lev2 toc-item"><a href="#预测数据集整理" data-toc-modified-id="预测数据集整理-54"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>预测数据集整理</a></div><div class="lev2 toc-item"><a href="#填充缺失值" data-toc-modified-id="填充缺失值-55"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>填充缺失值</a></div><div class="lev1 toc-item"><a href="#train" data-toc-modified-id="train-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>train</a></div><div class="lev2 toc-item"><a href="#feature-selection" data-toc-modified-id="feature-selection-61"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>feature selection</a></div><div class="lev1 toc-item"><a href="#predict" data-toc-modified-id="predict-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>predict</a></div><div class="lev2 toc-item"><a href="#predict-data" data-toc-modified-id="predict-data-71"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>predict data</a></div><div class="lev1 toc-item"><a href="#summary" data-toc-modified-id="summary-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>summary</a></div>

# 数据准备


```python
import numpy as np
import pandas as pd
import os
from  sklearn import cross_validation
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as sm
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import datetime
random.seed(0)
from sklearn.cross_validation import KFold
% matplotlib inline
```

# 自定义函数


```python
def rules(y_val,y_pred_val):
    df_merge=pd.DataFrame(y_pred_val)
    df_merge[1]=y_val
    dfMergeSort=df_merge.round(8).sort(0)
    dfMergeSort['one']=1
    dfMergeSortGB=dfMergeSort.groupby([0,1]).sum()
    dfFinal=dfMergeSortGB.unstack().fillna(0).cumsum()['one']
    dfFinal[0]=dfFinal[0]/float(dfFinal[0].values[-1])
    dfFinal[1]=dfFinal[1]/float(dfFinal[1].values[-1])
    dfFinal['difference']=dfFinal[1]-dfFinal[0]
    return abs(dfFinal.difference).max()
def getBalanceData_1(X,Y,randomN,n=1):
    # binary classification
    #X,Y=df.ix[:,:-1],df.ix[:,-1]
    valCounts=pd.value_counts(Y)
    (index_max,val_max)=valCounts.argmax(),valCounts.max()
    (index_min,val_min)=valCounts.argmin(),valCounts.min()
#     print (index_max,val_max),(index_min,val_min)
#     n=val_max/val_min
    X_min,Y_min=X[Y==index_min],Y[Y==index_min]
    X_max,Y_max=X[Y==index_max],Y[Y==index_max]
    vn_=val_min*n
    if vn_>=Y_max.shape[0]:
        vn_=Y_max.shape[0]
    print (Y_max.shape[0])/val_min
    Y_max_new=Y_max.sample(int(vn_),random_state=randomN)
    X_max_new=X_max.sample(int(vn_),random_state=randomN)
    return X_min.append(X_max_new),Y_min.append(Y_max_new)
def getBalanceData_2(X,Y,randomN):
    # binary classification
    #X,Y=df.ix[:,:-1],df.ix[:,-1]
    valCounts=pd.value_counts(Y)
    (index_max,val_max)=valCounts.argmax(),valCounts.max()
    (index_min,val_min)=valCounts.argmin(),valCounts.min()
#     print (index_max,val_max),(index_min,val_min)
    n=val_max/val_min
    X_min,Y_min=X[Y==index_min],Y[Y==index_min]
    X_max,Y_max=X[Y==index_max],Y[Y==index_max]
    x_col=X_min.columns
    X_min_new=pd.DataFrame(np.repeat(X_min.values,n,axis=0),columns=x_col)
    Y_min_new=pd.Series(np.repeat(Y_min.values,n,axis=0))
    print X_min_new.shape,Y_min_new.shape,X_max.shape,Y_max.shape
    return X_max.append(X_min_new),Y_max.append(Y_min_new)
# convert time to str
def time5folds(a):
    res=[]
    ab=np.array(a)/20.0
    for i in range(len(ab)):
        if ab[i]<=0:
            res.append('_'+str(int(ab[i])))
        elif ab[i]>0:
            res.append(str(int(ab[i])))
        else:
            res.append("Na")
    return res
# bank' money to + - according to deal_type
def dealMoneyChange(arr):
    res=[]
    for i in arr:
        if i[0]==1:
            res.append(-1*i[1])
        elif i[0]==0:
            res.append(i[1])
        else:
            res.append(np.nan)
    return res

category_col = ['sex', 'occupation', 'education', 'marriage', 'residence']
def set_dummies(data, colname):
    for col in colname:
        data[col] = data[col].astype('category')
        dummy = pd.get_dummies(data[col])
        dummy = dummy.add_prefix('{}#'.format(col))
        data.drop(col,
                  axis = 1,
                  inplace = True)
        data = data.join(dummy)
    return data


# 提供无bill时间的user_id 的数据
def numofNan(df_bill_pre,df_loan_pre):
    billLoan_pre=pd.merge(df_bill_pre,df_loan_pre,on='user_id',how='outer')
    bgb=billLoan_pre.groupby(['user_id']).size().reset_index()
    b0gb=billLoan_pre[billLoan_pre['bill_time']==0].groupby(['user_id']).size().reset_index()
    gbbill=pd.merge(bgb,b0gb,on='user_id',how='outer')
    gbbill['ratio']=gbbill['0_y']/gbbill['0_x']
    return gbbill

gbbill=numofNan(df_bill,df_loan)
billNoDate=gbbill[gbbill['ratio']==1]

# 浏览记录的ifidf
def tdIdf(df):
    cv=CountVectorizer()
    tft=TfidfVectorizer()
    
    all_cols=list(df.columns.values)
    dfgb=df.groubby(all_cols[:-1])
    n=len(dfgb)
    resAll=[]

    dfs_list=list(dfgb)
    for i in range(n):
        res=dfs_list[i][1][all_col[-1]].values
        res=[str(i)+"_" for i in res]
        resAll.append(" ".join(res))
    return resAll

def getXy(billNDue,n=85):
    return billNDue.iloc[:,range(2,n)],billNDue.iloc[:,1]
```

# 原始数据读取


```python
bank_list=['user_id','time','deal_type','deal_money','is_salary']
user_list=['user_id','sex','occupation','education','marriage','residence']
browse_list=['user_id','browse_time','behaviors','behaviors_code']
bill_list=['user_id','bill_time','bank_id','last_bill_amount','last_repayment_amount','credit_limit','current_bill_balance','least_repayment_amount','consume_num','current_bill_amount','adjust_amount',
           'recycle_interest','avaiable_amount','cash_limit','repayment_status']
loan_list=['user_id','loan_time']
overdue_list=['user_id','is_overdue']
```


```python
path="E:\\data_learn\\rong360\\person_1108\\train\\"
df_bank=pd.read_csv(path+'bank_detail_train.txt',names=bank_list)
df_bill=pd.read_csv(path+'bill_detail_train.txt',names=bill_list)
df_browse=pd.read_csv(path+'browse_history_train.txt',names=browse_list)
df_loan=pd.read_csv(path+'loan_time_train.txt',names=loan_list)
df_overdue=pd.read_csv(path+'overdue_train.txt',names=overdue_list)
df_user_info=pd.read_csv(path+'user_info_train.txt',names=user_list)
```


```python
df_user_info=pd.read_csv(path+'user_info_train.txt',names=user_list)
user_list=['user_id','sex','occupation','education','marriage','residence']
path="E:\\data_learn\\rong360\\person_1108\\test\\"
df_bank_pre=pd.read_csv(path+'bank_detail_test.txt',names=bank_list)
df_bill_pre=pd.read_csv(path+'bill_detail_test.txt',names=bill_list)
df_browse_pre=pd.read_csv(path+'browse_history_test.txt',names=browse_list)
df_loan_pre=pd.read_csv(path+'loan_time_test.txt',names=loan_list)
# df_overdue=pd.read_csv(path+'overdue_test.txt',names=overdue_list[0])
df_user_info_pre=pd.read_csv(path+'user_info_test.txt',names=user_list)
```

# 各表处理

## user 与overdue表


```python
user_dummies=set_dummies(df_user_info,category_col)
userDue=pd.merge(df_overdue,user_dummies,on='user_id',how='outer')
user_dummies_pre=set_dummies(df_user_info_pre,category_col)
```

## browse

- 浏览总次数 -提高f1 0.1个点

- 日期都是贷款loan之前的数据
- groupby('user_id').sum()总数有提升 为有效特征
- predict分布的趋势与browse不同
- 每5天的browse 分数比按behaviors_code的特征好

- code数据比比例效果好，有code时比多一个sum的和效果好



```python
#merge browse and loan
def browseloan(df_browse,df_loan):
    df_browse['one']=1
    bankLoan=pd.merge(df_browse,df_loan,on='user_id',how='outer')
    bankLoan['diff']=(bankLoan['browse_time']-bankLoan['loan_time'])/86400.0

    bankLoan['diffStr']=time5folds(bankLoan['diff'].values)
    del bankLoan['diff']
    bankLoan['str_code']=bankLoan['diffStr'].astype(str)+"_"+bankLoan['behaviors_code'].astype(str)
    # fabricate feature of date
    bankUnstack=bankLoan.groupby(['user_id','str_code']).sum()['one'].unstack()
    bankUnstack=bankUnstack.add_prefix("browse_dateCode")
    bankBase=bankUnstack.reset_index()
    return bankBase

def browseCode(df_browse):
    df_browse['one']=1
    return df_browse.groupby(['user_id','behaviors_code']).sum()['one'].unstack().reset_index()

# feature
browLoan=browseloan(df_browse,df_loan)
browLoan.to_csv("feature_data\\browe_dateCode_sum.csv",index=None)
browLoan=browseloan(df_browse_pre,df_loan_pre)
browLoan.to_csv("feature_data\\browe_dateCode_sum_pre.csv",index=None)

# feature
df_browse['one']=1
broLoaCode=df_browse.groupby(['user_id','behaviors_code']).sum()['one'].unstack().add_prefix("browse_code_sum_")
broLoaCode['sumCode']=broLoaCode.sum(axis=1)
broLoaCode.to_csv("feature_data\\browse_Code_sum.csv",index=None)
df_browse_pre['one']=1
broLoaCode=df_browse_pre.groupby(['user_id','behaviors_code']).sum()['one'].unstack().add_prefix("browse_code_sum_")
broLoaCode['sumCode']=broLoaCode.sum(axis=1)
broLoaCode.to_csv("feature_data\\browse_Code_sum_pre.csv",index=None)

#feature
broTime=df_browse.ix[:,['user_id','browse_time']].groupby(['user_id']).median().add_prefix("browse_time_").reset_index()
```

### browse日期范围分布


```python
# 处理日期
brmin=broLoa.ix[:,['user_id','diff']].groupby('user_id').min()
broLoa_pre=pd.merge(df_browse_pre,df_loan_pre,on='user_id',how='outer')
broLoa_pre['diff']=(broLoa_pre['browse_time']-broLoa_pre['loan_time'])/86400
del broLoa_pre['browse_time'],broLoa_pre['loan_time']
#预测集处理日期
brmin_pre=broLoa_pre.ix[:,['user_id','diff']].groupby('user_id').min()
```


```python
br_1=brmin.iloc[:13899,:]
br_2=brmin.iloc[13899:13899*2,:]
br_3=brmin.iloc[13899*2:13899*3,:]
br_4=brmin.iloc[13899*3:13899*4,:]
```


```python
#作图表示
plt.subplot(221)
plt.hist(br_1[br_1.notnull().values].values,bins=40)
plt.subplot(222)
plt.hist(br_1[br_1.notnull().values].values,bins=40)
plt.subplot(223)
plt.hist(br_1[br_1.notnull().values].values,bins=40)
plt.subplot(224)
plt.hist(br_1[br_1.notnull().values].values,bins=40)

#hist图
plt.hist(brmin_pre[brmin_pre.notnull().values].values,bins=40)
```

## bank


```python
# convert bank_time to dummies variables according to the sum of 5 days
def bankloan(df_bank,df_loan):
    bankLoan=pd.merge(df_bank,df_loan,on='user_id',how='outer')
    bankLoan['diff']=(bankLoan['time']-bankLoan['loan_time'])/86400.0

    bankLoan['diffStr']=time5folds(bankLoan['diff'].values)
    bankLoan['diffStr01']=bankLoan['diffStr']+"_"+bankLoan['deal_type'].astype(str)
    del bankLoan['diff']

    # deal money to +/-
    rerr=bankLoan.ix[:,['deal_type','deal_money']].values
    bankLoan['dealMoney']=dealMoneyChange(rerr)
    bankLoan['dealMoney']=bankLoan['dealMoney'].astype(float)
    # fabricate feature of date
    bankUnstack=bankLoan.groupby(['user_id','diffStr']).mean()['dealMoney'].unstack()
    bankdiffstr=bankLoan.groupby(['user_id','diffStr01']).mean()['deal_money'].unstack()

    bankSum=bankUnstack.add_prefix("bank_").reset_index()
    bankSumMean=bankdiffstr.add_prefix("code_").reset_index()
    return bankSum,bankSumMean
```


```python
bankSum,bankSumMean=bankloan(df_bank,df_loan)
bankSum_pre,bankSumMean_pre=bankloan(df_bank_pre,df_loan_pre)
```


```python
df=pd.merge(df_overdue,bankSumMean.reindex_axis(bs[bs[0]<50000].diffStr.values,axis=1),on='user_id',how='outer')
```


```python
bs=bankSumMean.isnull().sum().reset_index()
df=pd.merge(df_overdue,bankSumMean.reindex_axis(bs[bs[0]<50000].diffStr01.values,axis=1),on='user_id',how='outer')
```

## bill

- important!


```python
def billloan(df_bill,df_loan):
#     bill_overdue=df_bill
#     bill_overdue['pay_amount']=bill_overdue['last_repayment_amount']-bill_overdue['last_bill_amount']
    df_bill['pay_amount']=df_bill['last_repayment_amount']-df_bill['last_bill_amount']
#     bill_overdue=df_bill.reset_index().drop('index',axis=1)
#     res=[1]
    """
    for i in range(1,df_bill.shape[0]):
        if df_bill.ix[i,'user_id']==df_bill.ix[i-1,'user_id'] and df_bill.ix[i,'bank_id']==df_bill.ix[i-1,'bank_id'] and df_bill.ix[i,'last_bill_amount']==df_bill.ix[i-1,'current_bill_balance'] and df_bill.ix[i-1,'pay_amount']<0 and df_bill.ix[i,'pay_amount']<0:
            res.append(-1)
        else:
            res.append(0)
    df_bill['delayedTwoDays']=res
    """

    #--------------------------------------above to deal with bill_overdue------------------------------------------#
    billLoan=pd.merge(df_bill,df_loan,on='user_id',how='outer')
    billLoan['diff']=(billLoan['bill_time']-billLoan['loan_time'])/86400
#     billLoan['last_owe']=billLoan[]
    billLoan['diff5']=time5folds(billLoan['diff'].values)
    
#     billLoan_=billLoan[billLoan['diff']>=0]
#     billLoanUnstack=billLoan.ix[:,['user_id','diff5','pay_amount','delayedTwoDays']].groupby(['user_id','diff5']).mean().ix[:,['pay_amount','delayedTwoDays']].unstack()
#     billLoaPay=billLoanUnstack['pay_amount'].add_prefix("pay_").reset_index()
#     billLoaDel=billLoanUnstack['delayedTwoDays'].add_prefix("delay_").reset_index()
    billRepay_last=billLoan.ix[:,['user_id','diff5','last_repayment_amount','pay_amount']].groupby(['user_id','diff5']).mean().unstack()['last_repayment_amount'].add_prefix("rep_").reset_index()
#     billBalance=billLoan.ix[:,['user_id','diff5','current_bill_balance']].groupby(['user_id','diff5']).mean().unstack()['current_bill_balance'].add_prefix("bal_").reset_index()
#     return billLoaDel,billLoaPay,billRepay,
#     billLoan_=billLoan[billLoan['diff']<0]
#     billRepay_old=billLoan_.ix[:,['user_id','diff5','last_repayment_amount']].groupby(['user_id','diff5']).mean().unstack()['last_repayment_amount'].add_prefix("rep_").reset_index()
    billRepay_payamount=billLoan.ix[:,['user_id','diff5','last_repayment_amount','pay_amount']].groupby(['user_id','diff5']).mean().unstack()['pay_amount'].add_prefix("pay_").reset_index()
    return billRepay_last,billRepay_payamount
```


```python
#bill 与loan结合函数
def getbillLoan(df_bill,df_loan):
    billLoan=pd.merge(df_bill,df_loan,on='user_id',how='outer')
    billLoan['timediff']=((billLoan['bill_time']-billLoan['loan_time'])/86400)
    billLoan['diffN']=time5folds(billLoan['timediff'],30)
    billN=billLoan.ix[:,['user_id','last_bill_amount','diffN']].groupby(['user_id','diffN']).sum()['last_bill_amount'].unstack().add_prefix("bill_lastBill_").reset_index()

    billNDue=pd.merge(df_overdue,billN,on='user_id',how='outer')
    return billNDue
```


```python
# feature
# time
```

### bill时间与loan时间

- bill的时间分loan的前后进行拆分


```python
billLoan=pd.merge(df_bill,df_loan,on='user_id',how='outer')
billLoan['time_diff']=(billLoan['bill_time']-billLoan['loan_time'])/86400
billLoan_old=billLoan[billLoan['time_diff']>=0]
billLoan_new=billLoan[billLoan['time_diff']<0]
```


```python
# generate training set
df_user_info=pd.read_csv("E:\\data_learn\\rong360\\person_1108\\train\\user_info_train.txt",names=user_list)
def gebillDue(df_bill,df_loan,df_user_info,boo):
    billRepay_last,billRepay_payamount=billloan(df_bill,df_loan)
    bill_ON=pd.merge(billRepay_last,billRepay_payamount,on='user_id',how='outer')
    if boo==1:        
    # bill_oldDue=pd.merge(df_overdue,billRepay_last,on='user_id',how='outer')
    # bill_newDue=pd.merge(df_overdue,billRepay_payamount,on='user_id',how='outer')
        billDue=pd.merge(df_overdue,bill_ON,on='user_id',how='outer')
    else:
        billDue=bill_ON
    user_dummies=set_dummies(df_user_info,category_col)
    billDue=pd.merge(billDue,user_dummies,on='user_id',how='outer')
    
    return billDue
```


```python
#bill 结合
billDue_pre=gebillDue(df_bill_pre,df_loan_pre,df_user_info_pre,boo=0)
billDue=gebillDue(df_bill,df_loan,df_user_info,boo=1)
#保存
billDue_.iloc[:,1:-24].to_csv("E:\\data_learn\\rong360\\bill_train.csv",index=None)
billDue_pre.iloc[:,:-24].to_csv("E:\\data_learn\\rong360\\bill_predict.csv",index=None)
billDue_=billDue.reindex_axis(['is_overdue']+billDue_pre.columns.values.tolist(),axis=1)
```


```python
from sklearn.cross_validation import KFold
x,y=billDue_.iloc[:,range(2,390)],billDue_.iloc[:,0]
x_pre=billDue_pre.iloc[:,range(1,389)]

x1,y1=x.iloc[:13899,:],y.iloc[:13899]
x2,y2=x.iloc[13899:13899*2,:],y.iloc[13899:13899*2]
x3,y3=x.iloc[13899*2:13899*3,:],y.iloc[13899*2:13899*3]
x4,y4=x.iloc[13899*3:13899*4,:],y.iloc[13899*3:13899*4]
```


```python
# to test bill's useful features
clf_xgb=xgb.XGBClassifier(max_depth=3,learning_rate=0.1,n_estimators=100,objective='binary:logistic',\
                          min_child_weight=2,subsample=1,colsample_bytree=1,seed=0)

for k in KFold(billDue_.shape[0],4):
#     print "this type is k[0][0]= ",k[1][0]
#     if k[1][0]==13899:
            x_,y_=x.iloc[k[0],],y.iloc[k[0]]
            x_['addition']=np.array(range(13899)*3)/13899.0
            clf_xgb.fit(x_,y_)
            x_p,y_p=x.iloc[k[1],:],y.iloc[k[1]]
            x_p['addition']=np.array(range(13899))/13899.0
            y_Trpred_proba_1=clf_xgb.predict_proba(x_p)
            y_Tr=clf_xgb.predict(x_p)

            print sm.log_loss(y_p,y_Trpred_proba_1[:,1])
            print sm.classification_report(y_p,y_Tr)
            print rules(y_p.values,y_Trpred_proba_1[:,1])
#             y_pre=clf_xgb.predict_proba(x_pre)[:,1]
#             df_res_predict=pd.DataFrame(y_pre,columns=['probability'])
#             df_res_predict['userid']=billDue_pre['user_id'].astype(int)
clf_xgb=xgb.XGBClassifier(max_depth=3,learning_rate=0.1,n_estimators=120,objective='binary:logistic',min_child_weight=2,subsample=1,colsample_bytree=1,seed=0)
clf_xgb.fit(x,y)
```


```python
y_pre=clf_xgb.predict_proba(x_pre)[:,1]
df_res_predict=pd.DataFrame(y_pre,columns=['probability'])
df_res_predict['userid']=billDue_pre['user_id'].astype(int)
#output 
df_res_predict.reindex_axis(['userid','probability'],axis=1).to_csv("data\\submissionAlltrain.csv",index=None)
```


```python
#模型融合
df_1=pd.read_csv("C:\\Users\\Administrator\\Desktop\\86f945e7-a278-4282-932a-0655b345a360.csv")
df_1.probability=(df_res_predict.probability*0.55+df_1.probability*0.45)
# 0.7:0.3时为：42872
#0.6：0.4时为：43166
#0.5：0.5时为：43080
#0.55:0.45时为：43220
#df_1为最好的四个时：43123
df_1.reindex_axis(['userid','probability'],axis=1).to_csv("data\\submissionAlltrainN2.csv",index=None)
```


```python

```

# 各表合并
- bill 表列选择


```python
```python
billLoaDelPay=pd.merge(billLoaDel,billLoaPay,on='user_id',how='outer')
billLoaDelPay_pre=pd.merge(billLoaDel_pre,billLoaPay_pre,on='user_id',how='outer')

bankBaseUser=pd.merge(bankBase,user_dummies,on='user_id',how='outer')
bankBaseUser_pre=pd.merge(bankBase_pre,user_dummies_pre,on='user_id',how='outer')

billbank=pd.merge(billLoaDelPay,bankBaseUser,on='user_id',how='outer')
billbank_pre=pd.merge(billLoaDelPay_pre,bankBaseUser_pre,on='user_id',how='outer')

billbankDue=pd.merge(df_overdue,billbank,on='user_id',how='outer')
billbankDue_=billbankDue.reindex_axis(['is_overdue']+billbank_pre.columns.values.tolist(),axis=1)

billbankDue.to_csv("data\\billbankDue.csv",index=None)
billbank_pre.to_csv("data\\billbank_pre.csv",index=None)
```

## prepare data


```python
billbankDue=pd.read_csv("data\\billbankDue.csv")
billbank_pre=pd.read_csv("data\\billbank_pre.csv")
billbankDue_=billbankDue.reindex_axis(['is_overdue']+billbank_pre.columns.values.tolist(),axis=1)

billbankbroDue_=pd.merge(billbankDue_,browseCodTim,on='user_id',how='outer')
billbankbro_pre=pd.merge(billbank_pre,browseCodTim_pre,on='user_id',how='outer')
```

## 统计查看逾期的数据表的bill，browse以及bank表信息


```python

loanDue=pd.merge(df_loan,df_overdue,on='user_id',how='outer')
billLoan=pd.merge(df_bill.ix[:,range(7)],loanDue,on='user_id',how='outer')

billLoan['diff']=(billLoan['bill_time']-billLoan['loan_time'])/86400
del billLoan['bill_time'],billLoan['loan_time']

rs=billLoan[billLoan['diff']>0].sort(['user_id','diff'])
```

------
## strategy -start

- 针对bill里无时间的id进行预测
- 利用browse表

------------------------------------------------------------------------------------------------------------------------



```python
# 提供无bill时间的user_id 的数据

def numofNan(df_bill_pre,df_loan_pre):
    billLoan_pre=pd.merge(df_bill_pre,df_loan_pre,on='user_id',how='outer')
    bgb=billLoan_pre.groupby(['user_id']).size().reset_index()
    b0gb=billLoan_pre[billLoan_pre['bill_time']==0].groupby(['user_id']).size().reset_index()
    gbbill=pd.merge(bgb,b0gb,on='user_id',how='outer')
    gbbill['ratio']=gbbill['0_y']/gbbill['0_x']
    return gbbill

gbbill=numofNan(df_bill,df_loan)
billNoDate=gbbill[gbbill['ratio']==1]
```


```python
# browse 与loan合并
browseLoan=browseloan(df_browse,df_loan)
```


```python
# 与是否逾期表合并
broLoaUse=pd.merge(browseLoan,billNoDate.ix[:,['user_id','ratio']],on='user_id',how='right')
broLoaUse.drop('ratio',axis=1,inplace=True)
useDue=pd.merge(user_dummies,df_overdue,on='user_id',how='outer')
broLoaUseDue=pd.merge(broLoaUse,useDue,on='user_id',how='left')
```


```python
# 查看结果

x,y=broLoaUseDue.iloc[:,range(2,1237)],broLoaUseDue.iloc[:,1237]
clf_xgb=xgb.XGBClassifier(max_depth=4,learning_rate=0.1,n_estimators=80,objective='binary:logistic',min_child_weight=2,subsample=1,colsample_bytree=1,seed=0)

x_,y_=getBalanceData_1(x,y,67)

tr_x,te_x,tr_y,te_y=cross_validation.train_test_split(x_,y_,train_size=0.75,random_state=60)
clf_xgb.fit(tr_x,tr_y)

y_Trpred_proba=clf_xgb.predict_proba(te_x)
y_Tr=clf_xgb.predict(te_x)

print sm.log_loss(te_y,y_Trpred_proba[:,1])
print sm.classification_report(te_y,y_Tr)
print rules(te_y.values,y_Trpred_proba[:,1])

```

    0.694192569931
                 precision    recall  f1-score   support
    
              0       0.53      0.67      0.59       135
              1       0.63      0.49      0.55       159
    
    avg / total       0.58      0.57      0.57       294
    
    0.177917540182
    

    D:\Anaconda2\lib\site-packages\ipykernel\__main__.py:4: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    


```python
# 针对 bank表
bankSum,bankSumMean=bankloan(df_bank,df_loan)
```


```python
bankLoaUse=pd.merge(bankSumMean,billNoDate.ix[:,['user_id','ratio']],on='user_id',how='right')
useDue=pd.merge(user_dummies,df_overdue,on='user_id',how='outer')
bankLoaUseDue=pd.merge(bankLoaUse,useDue,on='user_id',how='left')
```


```python
x,y=bankLoaUseDue.iloc[:,range(2,214)],bankLoaUseDue.iloc[:,214]
clf_xgb=xgb.XGBClassifier(max_depth=3,learning_rate=0.1,n_estimators=80,objective='binary:logistic',min_child_weight=2,subsample=1,colsample_bytree=1,seed=0)

x_,y_=getBalanceData_1(x,y,67)
tr_x,te_x,tr_y,te_y=cross_validation.train_test_split(x_,y_,train_size=0.75,random_state=60)
clf_xgb.fit(tr_x,tr_y)

y_Trpred_proba=clf_xgb.predict_proba(te_x)
y_Tr=clf_xgb.predict(te_x)

print sm.log_loss(te_y,y_Trpred_proba[:,1])
print sm.classification_report(te_y,y_Tr)
print rules(te_y.values,y_Trpred_proba[:,1])
```

    0.669782933152
                 precision    recall  f1-score   support
    
              0       0.51      0.67      0.58       135
              1       0.62      0.46      0.53       159
    
    avg / total       0.57      0.55      0.55       294
    
    0.1928721174
    

    D:\Anaconda2\lib\site-packages\ipykernel\__main__.py:4: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    


```python
#bank 与browse合并
bankbroUse=pd.merge(broLoaUse,bankLoaUse,on='user_id',how='outer')
bankbroDue=pd.merge(bankbroUse,useDue,on='user_id',how='left')
```


```python
bankbroDue_=bankbroDue.reindex_axis(['is_overdue']+bankbroDue_pre.columns.values.tolist(),axis=1)
```


```python
x,y=bankbroDue_.iloc[:,range(2,1204)],bankbroDue_.iloc[:,0]
clf_xgb=xgb.XGBClassifier(max_depth=3,learning_rate=0.1,n_estimators=80,objective='binary:logistic',min_child_weight=2,subsample=1,colsample_bytree=1,seed=0)
x_,y_=getBalanceData_1(x,y,9)

tr_x,te_x,tr_y,te_y=cross_validation.train_test_split(x_,y_,train_size=0.75,random_state=60)
clf_xgb.fit(tr_x,tr_y)

y_Trpred_proba=clf_xgb.predict_proba(te_x)
y_Tr=clf_xgb.predict(te_x)

print sm.log_loss(te_y,y_Trpred_proba[:,1])
print sm.classification_report(te_y,y_Tr)
print rules(te_y.values,y_Trpred_proba[:,1])
```

    0.690278479234
                 precision    recall  f1-score   support
    
              0       0.53      0.72      0.61       135
              1       0.65      0.45      0.54       159
    
    avg / total       0.60      0.57      0.57       294
    
    0.199720475192
    

    D:\Anaconda2\lib\site-packages\ipykernel\__main__.py:4: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    

## 预测数据集整理


```python
#无bill表的predict
gbbill_pre=numofNan(df_bill_pre,df_loan_pre)
billNoDate_pre=gbbill_pre[gbbill_pre['ratio']==1]

#获取初始browse and bank
browseLoan=browseloan(df_browse_pre,df_loan_pre)
bankSum,bankSumMean=bankloan(df_bank_pre,df_loan_pre)
#user
user_dummies_pre=set_dummies(df_user_info_pre,category_col)
# useDue=pd.merge(user_dummies,df_overdue,on='user_id',how='outer')

#bank
bankLoaUse_pre=pd.merge(bankSumMean,billNoDate_pre.ix[:,['user_id','ratio']],on='user_id',how='right')
#browse处理
broLoaUse_pre=pd.merge(browseLoan,billNoDate_pre.ix[:,['user_id','ratio']],on='user_id',how='right')

#bank 与browse合并
bankbroUse_pre=pd.merge(broLoaUse_pre,bankLoaUse_pre,on='user_id',how='outer')
bankbroDue_pre=pd.merge(bankbroUse_pre,user_dummies_pre,on='user_id',how='left')
```


```python
bankbroDue_pre.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>browse_0_1.0</th>
      <th>browse_0_10.0</th>
      <th>browse_0_3.0</th>
      <th>browse_0_4.0</th>
      <th>browse_0_5.0</th>
      <th>browse_0_6.0</th>
      <th>browse_0_7.0</th>
      <th>browse_0_8.0</th>
      <th>browse_0_9.0</th>
      <th>...</th>
      <th>marriage#1</th>
      <th>marriage#2</th>
      <th>marriage#3</th>
      <th>marriage#4</th>
      <th>marriage#5</th>
      <th>residence#0</th>
      <th>residence#1</th>
      <th>residence#2</th>
      <th>residence#3</th>
      <th>residence#4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>55805.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 1203 columns</p>
</div>




```python
clf_xgb=xgb.XGBClassifier(max_depth=3,learning_rate=1,n_estimators=80,objective='binary:logistic',min_child_weight=2,subsample=0.9,colsample_bytree=1,seed=0)

x,y=bankbroDue_.iloc[:,range(2,1204)],bankbroDue_.iloc[:,0]
x_pre=bankbroDue_pre.iloc[:,range(1,1203)]
res_predict=[]
for i in range(50):
    x_,y_=getBalanceData_1(x,y,i)
    clf_xgb.fit(x_,y_,eval_metric='logloss')
    y_Trpred_proba=clf_xgb.predict_proba(x_pre)[:,1]
    res_predict.append(y_Trpred_proba)
```


```python
df_res_predict=pd.DataFrame(res_predict).T
pre_mean=df_res_predict.mean(axis=1).values
df_res_predict=pd.DataFrame(pre_mean,columns=['probability'])
df_res_predict['userid']=bankbroDue_pre['user_id'].astype(int)
#导出结果
df_re=df_res_predict.reindex_axis(['userid','probability'],axis=1)
# .to_csv("data\\submissionC_6.csv",index=None)
```


```python
last_=pd.read_csv("C:\\Users\\Administrator\\Desktop\\86f945e7-a278-4282-932a-0655b345a360.csv")
use_=np.array(list(set(last_.userid.values)-set(df_re.userid.values)))
f1=last_.set_index('userid').reindex(use_).reset_index()
f=f1.append(df_re)
f.to_csv("data\\submissionMerge.csv",index=None)
# 分数下降
```


```python
def tdIdf(df):
    cv=CountVectorizer()
    tft=TfidfVectorizer()
    
    all_cols=list(df.columns.values)
    dfgb=df.groubby(all_cols[:-1])
    n=len(dfgb)
    resAll=[]

    dfs_list=list(dfgb)
    for i in range(n):
        res=dfs_list[i][1][all_col[-1]].values
        res=[str(i)+"_" for i in res]
        resAll.append(" ".join(res))
        
    return resAll
```


```python
clf_xgb=xgb.XGBClassifier(max_depth=3,learning_rate=0.12,n_estimators=60,objective='binary:logistic',min_child_weight=2,\
                          subsample=0.9,colsample_bytree=0.5,seed=0,gamma=0.1,reg_lambda=5,scale_pos_weight=3)
# clf_xgb=xgb.XGBClassifier(max_depth=6,learning_rate=0.1,n_estimators=150,scale_pos_weight=3,objective='binary:logistic',min_child_weight=2,subsample=1,colsample_bytree=1,seed=0)

for k in KFold(x_tr.shape[0],4):
    for i in range(1):   
            x_,y_=x_tr.iloc[k[0],],y_tr.iloc[k[0]]
            clf_xgb.fit(x_,y_)
            x_p,y_p=x_tr.iloc[k[1],:],y_tr.iloc[k[1]]
            y_Tr=clf_xgb.predict(x_p)
            print "logloss:",sm.log_loss(y_p,clf_xgb.predict_proba(x_p)[:,1]),"rules:",rules(y_p.values,clf_xgb.predict_proba(x_p))
```

    logloss: 0.389023879557 rules: 0.458263596359
    

    D:\Anaconda2\lib\site-packages\ipykernel\__main__.py:4: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
    

    logloss: 0.390433669072 rules: 0.432108039462
    logloss: 0.422794401421 rules: 0.418623692093
    logloss: 0.44140105129 rules: 0.423087259234
    

## 填充缺失值
- need long time


```python
for i in range(1):
    x_trF,y_trF=x_tr[x_tr.iloc[:,i].T.notnull()>0].iloc[:,range(i)+range(i+1,83)],x_tr[x_tr.iloc[:,i].T.notnull()>0].iloc[:,i].values
    clf_xgb=xgb.XGBClassifier(max_depth=3,learning_rate=0.12,n_estimators=60,objective='binary:logistic',min_child_weight=2,\
                          subsample=0.9,colsample_bytree=0.5,seed=0,gamma=0.1,reg_lambda=5,scale_pos_weight=3)
    x_teF=x_tr[x_tr.iloc[:,i].T.notnull()==0].iloc[:,range(i)+range(i+1,83)]
    clf_xgb.fit(x_trF,y_trF)
    y_pre=clf_xgb.predict_proba(x_teF)
```

# train

## feature selection


```python
#特征重要性选择
def feaSelect(clf_xgb,df,df_te,n):
    df_fi=pd.DataFrame(clf_xgb.feature_importances_)
    df_fi['col_names']=df.columns.values
    df_fi.sort([0],ascending=False,inplace=True)
    
    return df.reindex_axis(df_fi.iloc[0:n]['col_names'].values,axis=1),df_te.reindex_axis(df_fi.iloc[0:n]['col_names'].values,axis=1)

df_=feaSelect(clf_xgb,browLoanDue,50)

# 选取重要的特征
```


```python
def mergeF(df,df_):
    return pd.merge(df,df_,on='userid',how='outer')
def mergeAll(df):
    df_=df[0]
    for m in range(1,len(df)):
        df_=mergeF(df_,df[m])
    return df_
def getXy(bDue,n):
    return bDue.iloc[:,range(2,n)],bDue.iloc[:,1]

def mergeIn(df,df1):
    return pd.merge(df,df1,on='user_id',how='inner')

bank_list=['user_id','time','deal_type','deal_money','is_salary']
user_list=['user_id','sex','occupation','education','marriage','residence']
browse_list=['user_id','browse_time','behaviors','behaviors_code']
bill_list=['user_id','bill_time','bank_id','last_bill_amount','last_repayment_amount','credit_limit','current_bill_balance','least_repayment_amount','consume_num','current_bill_amount','adjust_amount',
           'recycle_interest','avaiable_amount','cash_limit','repayment_status']
loan_list=['user_id','loan_time']
overdue_list=['user_id','is_overdue']
##############
df_overdue=pd.read_csv('overdue_train.txt',names=overdue_list)
df=pd.read_csv("ks45.csv")
#############
df_=df.set_index('userid')

train=df_.iloc[:13899*3,:]
test=df_.iloc[13899*3:13899*4,:]
# to filter feature according to the feature's information of predict set
def featureSec(x,n=0):
    bs=x.notnull().sum().reset_index()
    features=bs[bs[0]>n]['index'].values
    res_bill=[]
    res_bank=[]
    res_browse=[]
    res_other=[]
    res_mc=[]
    for i in features:
        if type(i)!=long and len(i)>=4 and i[:4]=='bill':
            res_bill.append(i)    
    # for i in features:
        elif type(i)!=long and len(i)>=4 and i[:4]=='bank':
            res_bank.append(i) 
    # for i in features:
        elif type(i)!=long and len(i)>=4 and i[:4]=='brow':
            res_browse.append(i)
        elif type(i)!=long and i[:1]=='K':
            res_mc.append(i)
        else:
            res_other.append(i)                 
    res_bill_=list(set(res_bill)-set(['bill_dateOweMoney_Na', 'bill_dateOweMoney__-2288','bill_dateOweMoney__-2289','bill_dateLastPay_Na',\
                             'bill_dateLastPay__-2288', 'bill_dateLastPay__-2289','bill_delay2Days_Na', 'bill_delay2Days__-2281',\
                         'bill_delay2Days__-2282', 'bill_delay2Days__-2283', 'bill_delay2Days__-2284', 'bill_delay2Days__-2285',\
                         'bill_delay2Days__-2286', 'bill_delay2Days__-2287', 'bill_delay2Days__-2288']))

    res_bank_=list(set(res_bank)-set(['bank_dateMean_Na','bank_dateMean__-6865','bank_dateMean__-6866','bank_dateMean__-6867','bank_dateMean__-6868',\
                           'bank_date01Mean_Na_nan','bank_date01Mean__-6865_0.0', 'bank_date01Mean__-6866_0.0','bank_date01Mean__-6867_0.0',\
                            'bank_date01Mean__-6867_1.0','bank_date01Mean__-6868_0.0','bank_one_Na','bank_one__-6865', 'bank_one__-6866',\
                           'bank_one__-6867','bank_one__-6868']))
    
#     return res_mc,res_bill,res_bank,res_browse,res_other
#     return x.reindex_axis(res_other+res_bill_+res_bank_+res_browse,axis=1)
    return x.reindex_axis(features,axis=1)

def getTrain(train,test,m=1500,n=0):
    # the num userid of Na of train and test
    trainTNull=train.T.notnull().sum()
    testTNull=test.T.notnull().sum()

    #feature to split m
    Lnu=test[testTNull<m]
    Mnu=test[testTNull>m]

    #null to predict
    LnuFs=featureSec(Lnu,n)
#     MnuFs=featureSec(Lnu,n)
    #to choost featrue of train of LnuFS
    trainFsNu=train.reindex_axis(LnuFs.columns,axis=1)
    testFs=train.reindex_axis(Lnu)
    
    return trainFsNu
```


```python
# x,y=browLoanDue.iloc[:,range(2,670)],browLoanDue.iloc[:,1]
x,y=browLoanMeDue.iloc[:,range(2,1034)],browLoanMeDue.iloc[:,1]
# ,25)+range(670
clf_xgb=xgb.XGBClassifier(max_depth=3,learning_rate=0.1,n_estimators=100,objective='binary:logistic',min_child_weight=2,subsample=1,colsample_bytree=1,seed=0)

tr_x,tr_y=x.iloc[:-13899],y.iloc[:-13899]
te_x,te_y=x.iloc[-13899:],y.iloc[-13899:]

clf_xgb.fit(tr_x,tr_y)

tr_x,te_x=feaSelect(clf_xgb,tr_x,te_x,1032)
clf_xgb.fit(tr_x,tr_y)

y_Trpred_proba_1=clf_xgb.predict_proba(te_x)
y_Tr=clf_xgb.predict(te_x)

print sm.log_loss(te_y,y_Trpred_proba_1[:,1])
print sm.classification_report(te_y,y_Tr)
# y_Trpred_proba_1
print rules(te_y.values,y_Trpred_proba_1[:,1])
```


```python
rainNu=getTrain(train,test)
df_overdue.rename_axis({"user_id":'userid'},axis=1,inplace=True)

trainNuDue=mergeF(df_overdue,trainNu.reset_index())
trainDue=mergeF(df_overdue,train.reset_index())


x,y=getXy(trainDue,2723)
clf_xgb=xgb.XGBClassifier(max_depth=3,learning_rate=0.12,n_estimators=200,objective='binary:logistic',min_child_weight=2,\
                          subsample=0.9,colsample_bytree=0.5,seed=0,gamma=0.1,reg_lambda=5,scale_pos_weight=3)
for k in KFold(x.shape[0],4):
    for i in range(1):   
            x_,y_=x.iloc[k[0],],y.iloc[k[0]]
            clf_xgb.fit(x_,y_)
            x_p,y_p=x.iloc[k[1],:],y.iloc[k[1]]
            y_Tr=clf_xgb.predict(x_p)
            print "logloss:",sm.log_loss(y_p,clf_xgb.predict_proba(x_p)[:,1]),"rules:",rules(y_p.values,clf_xgb.predict_proba(x_p))
```


```python
#n-folds validation to fillter features
for k in KFold(x.shape[0],4):
        x_,y_=x.iloc[k[0],],y.iloc[k[0]]
        clf_xgb.fit(x_,y_)
        x_p,y_p=x.iloc[k[1],:],y.iloc[k[1]]

        y_Trpred_proba_1=clf_xgb.predict_proba(x_p)
        y_Tr=clf_xgb.predict(x_p)
#         print sm.log_loss(y_p,y_Trpred_proba_1[:,1])
#         print sm.classification_report(y_p,y_Tr)
        print rules(y_p.values,y_Trpred_proba_1[:,1])
```


```python
#按behaviors_code为特征
0.327823565711
0.316904859894
0.239341534338
0.185771054207
#按behaviors为特征
0.380620924022
0.387980339358
0.27439968311
0.23419158909
#按behaviors与code的联合为特征
0.385134948972
0.394074479969
0.27439968311
0.230750782691
# date与behaviors结合4070个特征，每30天
0.38325489336
0.400398292628
0.293119919007
0.236834810502
#按date与浏览数量结合为特征 每5天
0.328382317954
0.357284620721
0.264699074379
0.217536769025
#按date与浏览数量结合为特征 每10天
0.350186502216
0.352584927625
0.267909056989
0.220725102792
#按date与浏览数量结合为特征 每15天
0.357030984865
0.36769549522
0.27481273007
0.223671125635
#按date与浏览数量结合为特征 每20天
0.35055960505
0.383146173112
0.275715129525
0.225415581116
#按date与浏览数量结合为特征 每25天

0.36051136463
0.373360629329
0.273733781366
0.220035894813
#按date与浏览数量结合为特征 每30天
0.359364613254
0.369452535776
0.27031152764
0.222185335758
```

# predict

## predict data


```python
dfmc=pd.read_csv("D:\\Program Files (x86)\\qqq\\1533782617\\FileRecv\\ks45\\ks45.csv")
dfmcDue=pd.merge(df_overdue.rename_axis({"user_id":"userid"},axis=1),dfmc,on='userid',how='outer')
def getFea(dfmcDue):
    return dfmcDue.T[dfmcDue.notnull().sum()>0].T    
sdDue=getFea(dfmcDue)
```


```python
# x_tr,y_tr=sdDue.iloc[:13899*4,range(2,2545)],sdDue.iloc[:13899*4,1]
x_pre=sdDue.iloc[13899*4:13899*5,range(2,2545)]
clf_xgb=xgb.XGBClassifier(max_depth=6,learning_rate=0.1,n_estimators=350,objective='binary:logistic',min_child_weight=1.5,\
                          subsample=0.6,colsample_bytree=0.5,seed=0,colsample_bylevel= 0.5,reg_lambda=360)

clf_xgb.fit(x_tr.append(sdDue.iloc[13899*2:13899*4,range(2,2545)]),y_tr.append(sdDue.iloc[13899*2:13899*4,1]))
#predict
y_pre=clf_xgb.predict_proba(x_pre)
```


```python
clf_xgb=xgb.XGBClassifier(max_depth=6,learning_rate=0.1,n_estimators=350,objective='binary:logistic',min_child_weight=1.5,\
                          subsample=0.6,colsample_bytree=0.5,seed=0,colsample_bylevel= 0.5,reg_lambda=360,scale_pos_weight=3)

clf_xgb.fit(x_tr.append(sdDue.iloc[13899*2:13899*4,range(2,2545)]),y_tr.append(sdDue.iloc[13899*2:13899*4,1]))
y_pre1=clf_xgb.predict_proba(x_pre)
```


```python
#scale_pos_weight不同尺度下模型融合
# bad result
df_=sdDue.iloc[13899*4:,0].reset_index().astype(int)
df_['probability0']=y_pre[:,1]
df_['probability3']=y_pre1[:,1]
df_['probability']=df_.iloc[:,[2,3]].mean(axis=1)
df_.ix[:,['userid','probability']].to_csv("data\\last_v1.csv",index=None)
```


```python
# 选择null<1000的特征
# best result
clf_xgb=xgb.XGBClassifier(max_depth=6,learning_rate=0.1,n_estimators=350,objective='binary:logistic',min_child_weight=1.5,\
                          subsample=0.6,colsample_bytree=0.5,seed=0,colsample_bylevel= 0.5,reg_lambda=360)
x_preF=x_pre[x_pre.T.notnull().sum()<1000]
x_preFea=getFea(x_preF)
x_trFea=x_tr.reindex_axis(x_preFea.columns,axis=1)
x_preFeaNull=x_pre.reindex_axis(x_preFea.columns,axis=1)
clf_xgb.fit(x_trFea.append(x_trFea.iloc[13899*2:13899*4]),y_tr.append(sdDue.iloc[13899*2:13899*4,1]))
#predict
y_preFea_1=clf_xgb.predict_proba(x_preFeaNull)
clf_xgb.fit(x_trFea.append(x_trFea.iloc[13899*2:13899*4]),y_tr.append(sdDue.iloc[13899*2:13899*4,1]))
#predict
y_preFea_2=clf_xgb.predict_proba(x_preFeaNull)
```


```python
# 求预测特征少于1000时的训练特征数据
df_nu=sdDue.iloc[13899*4:,0].reset_index().astype(int)
df_nu['probability0']=y_preFea_1[:,1]
df_nu['probability3']=y_preFea_2[:,1]
df_nu['probability']=df_nu.iloc[:,[2,3]].mean(axis=1)
df_nu.ix[:,['userid','probability']].to_csv("data\\last_nuv1.csv",index=None)
```


```python
#null 的userid
x_preNuid=pd.DataFrame(sdDue.iloc[13899*4:][x_pre.T.notnull().sum()<1000]['userid'].values,columns=['userid'])
#null 的列
df_Nu=pd.merge(df_nu.ix[:,['userid','probability']],x_preNuid,on='userid',how='inner')
# 两模型结合
x_preHavd=pd.DataFrame(list(set(sdDue.iloc[13899*4:]['userid'].values)-set(x_preNuid.userid.values)),columns=['userid'])
df_have=pd.merge(df_.ix[:,['userid','probability']],x_preHavd,on='userid',how='inner')
df_all=df_Nu.append(df_have).sort_values(['userid'])
df_all.to_csv("data\\last_v2.csv",index=None)
```


```python
# best 为ks所有数据，同时再加上第四部分数据，
# xgboost参数为：
# (max_depth=6,learning_rate=0.1,n_estimators=300,objective='binary:logistic',min_child_weight=1.5,\
#                           subsample=0.6,colsample_bytree=0.5,seed=0,colsample_bylevel= 0.5,reg_lambda=360）
# 同时
# null 是特征为特征数少于1000的用户的特征，数据所有用户数据，并加第4部分数据
# xgboost参数
# (max_depth=6,learning_rate=0.1,n_estimators=300,objective='binary:logistic',min_child_weight=1.5,\
#                           subsample=0.6,colsample_bytree=0.5,seed=0,colsample_bylevel= 0.5,reg_lambda=360）
 

## analyse

# analyse the different of different model
# path="D:\\Anaconda2\\code\\rong360\\data\\"
# df_2=pd.read_csv(path+"result\\p458_1.csv")
# df_1=pd.read_csv(path+"result\\p458_1.csv")
# df_2=pd.read_csv(path+"result\\p458_1.csv")
# df_3=pd.read_csv(path+"last_v2.csv")
# df_2=pd.read_csv(path+"last_v1.csv")
# df_2=pd.read_csv(path+'allData_234_v1.csv')

# df_3=pd.read_csv(path+"allData_234_2509_250_seed0.csv")

# df_=pd.merge(df_2,df_3,on='userid',how='inner')
# dfT=df_.sort_values(['probability_x']).reset_index(drop=True)
# df_['sdiff']=df_['probability_x']-df_['probability_y']
```

- 对所有数据和bill_trian合并结果将最终结果与自己最后0.55:0.45模型融合 0.43123
- 训练数据同上，但选取后13899-13899*4之间的部分总体数据作为预测集,0.42
- 将训练非平衡数据平横后，负例为正例的2倍，效果较使用全部数据变差

- 特征的顺序也会影响预测概率的分布
- 对各个表获取的特征命名以各表，各类为区别，以便后续进行特征选择

# summary

1.一定要备份数据代码！！！

- in general

    > - 时间序列预测时，选择合适的训练数据而不是全部

    > - 调参很重要，各种参数都要用上

    > - 对于特定的预测集，分类进行预测效果更好

- for some details

    > - 先去重复数据
    > - 合适地填充数据
    > - 首先尽可能建立所有特征，并对每一类特征标记
    > - 特征选择时
       * 去除重复特征，如相关性为1，方差为0，与标签相关性为0等（无缺失值情况）filter
       * 存在缺失值时，慎重去除含缺失值的特征列
       * 根据重要性选择特征（可尽量不去特征，通过调参使之更conversation）
       * 正则化项选择特征embeded method
       * wrapper method 一个一个减去特征观察效果recursive featrue elimination
    > - 调参 xgboost 多尝试，cv简单查看参数，train来直接看出最佳次数，gridsearch 选最佳参数
    > - 详细记录每一次提交数据来源，处理过程以及预测参数和线上结果
    > - 时间序列预测时选测相应的前几天，同时选测合适的训练集


```python

```
