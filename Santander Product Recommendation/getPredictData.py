def processData_test_1(in_file_name):
	x_vars_list = []
	y_vars_list = []
	cust_dict_1,cust_dict_2,cust_dict_3,cust_dict_4,cust_dict_5,cust_dict_6={},{},{},{},{},{}
	for row in csv.DictReader(in_file_name):
		# use only the four months as specified by breakfastpirate #
		if row['fecha_dato'] not in ['2016-05-28','2016-04-28','2016-03-28','2016-02-28','2016-01-28']:
			#  ['2015-05-28', '2015-06-28','2015-04-28',]
			continue            
		cust_id = int(row['ncodpers'])
		if row['fecha_dato']=='2016-01-28':#'2015-04-28'
			cust_dict_1[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2016-02-28':#'2015-04-28'
			cust_dict_2[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2016-03-28':#'2015-04-28'
			cust_dict_3[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2016-04-28':#'2015-04-28'
			cust_dict_4[cust_id]=getTarget(row)[:]
			continue
		if row['fecha_dato']=='2016-05-28':#'2015-04-28'
			cust_dict_5[cust_id]=cust_dict_1.get(cust_id,[0]*24)+cust_dict_2.get(cust_id,[0]*24)+cust_dict_3.get(cust_id,[0]*24)+cust_dict_4.get(cust_id,[0]*24)+getTarget(row)[:]
                
	return cust_dict_5
def processData_test_2(in_file_name,cust_dict):
	x_vars_list = []
	y_vars_list = []
	for row in csv.DictReader(in_file_name):
		# use only the four months as specified by breakfastpirate #
            
		cust_id = int(row['ncodpers'])
		x_vars = []
		for col in cat_cols:
			x_vars.append( getIndex(row, col) )# to get the features id
		x_vars.append( getAge(row) ) #to normalize the age
		x_vars.append( getCustSeniority(row) )# to normalize the custmer seniority
		x_vars.append( getRent(row) )# to normalize the gross income of househodld        
		x_vars_list.append(x_vars+cust_dict.get(cust_id,[0]*24))
        
	return x_vars_list,y_vars_list