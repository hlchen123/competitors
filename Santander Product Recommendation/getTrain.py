import numpy as np
import pandas as pd
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
from preprocessFunction import *

def processData_train_all(in_file_name):
	x_vars_list = []
	y_vars_list = []
	cust_dict_1,cust_dict_2,cust_dict_3,cust_dict_4,cust_dict_5,cust_dict_6={},{},{},{},{},{}

	for row in csv.DictReader(in_file_name):
		# use only the four months as specified by breakfastpirate #
		if row['fecha_dato'] not in ['2015-06-28','2015-05-28','2015-04-28','2015-03-28','2015-02-28','2015-01-28']:
			#  ['2015-05-28', '2015-06-28','2015-04-28',]
			continue
            
		cust_id = int(row['ncodpers'])
		if row['fecha_dato']=='2015-01-28':#'2015-04-28'
			cust_dict_1[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2015-02-28':#'2015-04-28'
			cust_dict_2[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2015-03-28':#'2015-04-28'
			cust_dict_3[cust_id]=getTarget(row)[:]
			continue            
		if row['fecha_dato']=='2015-04-28':#'2015-04-28'
			cust_dict_4[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2015-05-28':#'2015-04-28'
			cust_dict_5[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2015-06-28':
			x_vars = []
			for col in cat_cols:
				x_vars.append( getIndex(row, col) )# to get the features id
			x_vars.append( getAge(row) ) #to normalize the age
			x_vars.append( getCustSeniority(row) )# to normalize the custmer seniority
			x_vars.append( getRent(row) )# to normalize the gross income of househodld
			target_1=cust_dict_1.get(cust_id,[0]*24)
			target_2=cust_dict_2.get(cust_id,[0]*24)
			target_3=cust_dict_3.get(cust_id,[0]*24)
			target_4=cust_dict_4.get(cust_id,[0]*24)
			target_5=cust_dict_5.get(cust_id,[0]*24)
			target_6=getTarget(row)[:]

			add_june=[max(x1-x2,0) for (x1,x2) in zip(target_6,target_5)]
			if sum(add_june)>0:
				x_vars_list.append(x_vars+target_1+target_2+target_3+target_4+target_5)
				y_vars_list.append(add_june)
	return x_vars_list, y_vars_list

def get_train_null(in_file_name):    
	x_vars_list = []
	y_vars_list = []
	cust_dict_1,cust_dict_2,cust_dict_3,cust_dict_4,cust_dict_5,cust_dict_6={},{},{},{},{},{}
	cust_dict_9,cust_dict_10,cust_dict_11,cust_dict_64,cust_dict_65={},{},{},{},{}
	for row in csv.DictReader(in_file_name):
		# use only the four months as specified by breakfastpirate #
		if row['fecha_dato'] not in ['2015-06-28','2015-05-28','2015-04-28',
                                     '2015-03-28','2015-02-28','2015-01-28',
                                    '2015-09-28','2015-10-28','2015-11-28','2016-04-28','2016-05-18']:
			#  ['2015-05-28', '2015-06-28','2015-04-28',]
			continue
            
		cust_id = int(row['ncodpers'])
		if row['fecha_dato']=='2015-01-28':#'2015-04-28'
			cust_dict_1[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2015-02-28':#'2015-04-28'
			cust_dict_2[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2015-03-28':#'2015-04-28'
			cust_dict_3[cust_id]=getTarget(row)[:]
			continue     
		if row['fecha_dato']=='2015-09-28':#'2015-04-28'
			cust_dict_9[cust_id]=getTarget(row)[:]
			continue    
		if row['fecha_dato']=='2016-04-28':#'2015-04-28'
			cust_dict_64[cust_id]=getTarget(row)[:]
			continue    
		if row['fecha_dato']=='2015-04-28':
			cust_dict_4[cust_id]=getTarget(row)[:]
			x_vars = []
			for col in cat_cols:
				x_vars.append( getIndex(row, col) )# to get the features id
			x_vars.append( getAge(row) ) #to normalize the age
			x_vars.append( getCustSeniority(row) )# to normalize the custmer seniority
			x_vars.append( getRent(row) )# to normalize the gross income of househodld
			target_2=cust_dict_2.get(cust_id,[0]*24)
			target_3=cust_dict_3.get(cust_id,[0]*24)
			target_4=cust_dict_4.get(cust_id,[0]*24)
			add_=[max(x1-x2,0) for (x1,x2) in zip(target_4,target_3)]
			if sum(add_)>0 and sum(target_3+target_2)==0:
				x_vars_list.append(x_vars)
				y_vars_list.append(add_)
		if row['fecha_dato']=='2015-05-28':
			cust_dict_5[cust_id]=getTarget(row)[:]
			x_vars = []
			for col in cat_cols:
				x_vars.append( getIndex(row, col) )# to get the features id
			x_vars.append( getAge(row) ) #to normalize the age
			x_vars.append( getCustSeniority(row) )# to normalize the custmer seniority
			x_vars.append( getRent(row) )# to normalize the gross income of househodld
			target_3=cust_dict_3.get(cust_id,[0]*24)
			target_4=cust_dict_4.get(cust_id,[0]*24)
			target_5=cust_dict_5.get(cust_id,[0]*24)
			add_=[max(x1-x2,0) for (x1,x2) in zip(target_5,target_4)]
			if sum(add_)>0 and sum(target_4+target_3)==0:
				x_vars_list.append(x_vars)
				y_vars_list.append(add_)
		if row['fecha_dato']=='2015-06-28':
			x_vars=[]
			for col in cat_cols:
				x_vars.append( getIndex(row, col) )# to get the features id
			x_vars.append( getAge(row) ) #to normalize the age
			x_vars.append( getCustSeniority(row) )# to normalize the custmer seniority
			x_vars.append( getRent(row) )# to normalize the gross income of househodld
			target_4=cust_dict_4.get(cust_id,[0]*24)
			target_5=cust_dict_5.get(cust_id,[0]*24)
			target_6=getTarget(row)[:]
			add_june=[max(x1-x2,0) for (x1,x2) in zip(target_6,target_5)]
			if sum(add_june)>0 and sum(target_5+target_4)==0:
				x_vars_list.append(x_vars)
				y_vars_list.append(add_june)
		if row['fecha_dato']=='2015-10-28':
			cust_dict_10[cust_id]=getTarget(row)[:]
			x_vars = []
			for col in cat_cols:
				x_vars.append( getIndex(row, col) )# to get the features id
			x_vars.append( getAge(row) ) #to normalize the age
			x_vars.append( getCustSeniority(row) )# to normalize the custmer seniority
			x_vars.append( getRent(row) )# to normalize the gross income of househodld
			target_9=cust_dict_9.get(cust_id,[0]*24)
			target_10=cust_dict_10.get(cust_id,[0]*24)            
			add_=[max(x1-x2,0) for (x1,x2) in zip(target_10,target_9)]
			if sum(add_)>0 and sum(target_9)==0:
				x_vars_list.append(x_vars)
				y_vars_list.append(add_)
		if row['fecha_dato']=='2015-11-28':
			cust_dict_11[cust_id]=getTarget(row)[:]
			x_vars = []
			for col in cat_cols:
				x_vars.append( getIndex(row, col) )# to get the features id
			x_vars.append( getAge(row) ) #to normalize the age
			x_vars.append( getCustSeniority(row) )# to normalize the custmer seniority
			x_vars.append( getRent(row) )# to normalize the gross income of househodld
			target_11=cust_dict_11.get(cust_id,[0]*24)
			target_10=cust_dict_10.get(cust_id,[0]*24)            
			add_=[max(x1-x2,0) for (x1,x2) in zip(target_11,target_10)]
			if sum(add_)>0 and sum(target_10)==0:
				x_vars_list.append(x_vars)
				y_vars_list.append(add_)
		if row['fecha_dato']=='2016-05-28':
			cust_dict_65[cust_id]=getTarget(row)[:]
			x_vars = []
			for col in cat_cols:
				x_vars.append( getIndex(row, col) )# to get the features id
			x_vars.append( getAge(row) ) #to normalize the age
			x_vars.append( getCustSeniority(row) )# to normalize the custmer seniority
			x_vars.append( getRent(row) )# to normalize the gross income of househodld
			target_65=cust_dict_65.get(cust_id,[0]*24)
			target_64=cust_dict_64.get(cust_id,[0]*24)            
			add_=[max(x1-x2,0) for (x1,x2) in zip(target_65,target_64)]
			if sum(add_)>0 and sum(target_64)==0:
				x_vars_list.append(x_vars)
				y_vars_list.append(add_)
	return x_vars_list, y_vars_list