import numpy as np
import csv

# 
file1name='train.csv'
file2name='test_X.csv'
file3name='Submi.csv'
eta=0.0000000001		# Learning rate
hoursfortrain = 7		# former 8 hours to predict PM2.5 of the 9th hour

# One month
traindata = np.zeros((18*hoursfortrain,(24*20-hoursfortrain)*12))	# reformat datafromfile to fit the format of input data of linear regression.
pm25data = np.zeros((24*20-hoursfortrain)*12)		# 
datafromfile = np.zeros((18,480*12))	# Data from test_X.csv. 18 Rows for 18 items related to PM2.5 density. we get 480 hours measurement data in one 12.
pm25datafromfile = np.zeros(24)	# Only 24 elements space is required.

day=0		# Count how many day has passed.
pm25data_index = 0

file1 = open(file1name,'r',encoding='utf-8', errors='ignore')
file1_num_of_row = -1
for row in csv.reader(file1):
	file1_num_of_row+=1
	
	# The first line describes names of every items, not required to store them.
	if(file1_num_of_row == 0):
		continue
	
	if(file1_num_of_row%18==10):		# PM2.5 -> answer
		pm25datafromfile = row[3:]
		datafromfile[file1_num_of_row-1,day*24:day*24+24] = row[3:]
	# Convert str2int to make it understandable for this program.
	elif(file1_num_of_row%18==11):		# RAINFALL -> If it doesn't rain, it takes NR as its value. (shuold be converted.)
		datafromstr2num=[0 if x=='NR' else x for x in row[3:]]
		datafromfile[file1_num_of_row-1,day*24:day*24+24] = datafromstr2num
	else:
		datafromfile[file1_num_of_row-1,day*24:day*24+24] = row[3:]
	
	# Finish a cycle of one day.
	if(file1_num_of_row%18 == 0):
		file1_num_of_row=0
		day+=1
		# dayi -> day:1
		if(day%20!=1):
			pm25data[pm25data_index:pm25data_index+24] = pm25datafromfile[0:]
			pm25data_index+=24
		# the first 9 hours' PM2.5 data should be abondoned.
		else:
			pm25data[pm25data_index:pm25data_index+(24-hoursfortrain)] = pm25datafromfile[hoursfortrain:]
			pm25data_index+=(24-hoursfortrain)
file1.close()
for frontiter in range((24*20-hoursfortrain)*12):
	traindata[:,frontiter]=np.concatenate((datafromfile[:,frontiter+(frontiter//(24*20-hoursfortrain))*hoursfortrain],datafromfile[:,frontiter+(frontiter//(24*20-hoursfortrain))*hoursfortrain+1],datafromfile[:,frontiter+(frontiter//(24*20-hoursfortrain))*hoursfortrain+2],datafromfile[:,frontiter+(frontiter//(24*20-hoursfortrain))*hoursfortrain+3],datafromfile[:,frontiter+(frontiter//(24*20-hoursfortrain))*hoursfortrain+4],datafromfile[:,frontiter+(frontiter//(24*20-hoursfortrain))*hoursfortrain+5],datafromfile[:,frontiter+(frontiter//(24*20-hoursfortrain))*hoursfortrain+6]))#,datafromfile[:,frontiter+(frontiter//(24*20-hoursfortrain))*hoursfortrain+7]))

"""
create a array storing all parameters.
"""
Weights = np.zeros(18*hoursfortrain)		# 18*hoursfortrain Weights.
par_Weights = np.zeros(18*hoursfortrain)		# 18*hoursfortrain partial derivatives of Weights.
bias = 1.0		# Bias.
par_bias = 0.0		# partial derivatives of bias.
minn = 30
for iternum in range(600000):
	K = ((np.dot(Weights,traindata)+np.ones((24*20-hoursfortrain)*12)*bias)*-1+pm25data)*-2
	par_Weights = np.dot(traindata,K)
	par_bias = np.dot(np.ones((24*20-hoursfortrain)*12),K)
	Weights -= eta*par_Weights
	bias -= eta*par_bias
	if(iternum==450000):
		eta*=0.1
	if(iternum==540000):
		eta*=0.1
	if(iternum==590000):
		eta*=0.1
	if(iternum==595000):
		eta*=0.1
	if(iternum==599000):
		eta*=0.01

	""""""

	if(iternum%10000==0):	
		result_K = K/-2
		result_total = np.sum(np.abs(result_K)) / np.shape(result_K)[0]
		print('At '+str(iternum)+'th iteration, the average error is '+str(result_total))
		if(result_total<minn):
			minn=result_total
			best_Weights = Weights
			best_bias = bias
			#break

print('The minimum value: '+str(minn))
Weights = best_Weights
bias = best_bias
	
# Test part
""""""
testdata = np.zeros((18*hoursfortrain,240))	# reformat testdatafromfile to fit the format of input data of linear regression.
testdatafromfile = np.zeros((18,9*240))	# Data from test_X.csv. 18 Rows for 18 items related to PM2.5 density. we get 240 data in total and 9 hours for one datum.

id_num=0		# Count the present id.

file2 = open(file2name,'r',encoding='utf-8', errors='ignore')
file2_num_of_row = 0
for row in csv.reader(file2):
	file2_num_of_row+=1
	
	# Convert str2int to make it understandable for this program.
	if(file2_num_of_row%18==11):		# RAINFALL -> If it doesn't rain, it takes NR as its value. (shuold be converted.)
		testdatafromstr2num=[0 if x=='NR' else x for x in row[2:]]
		testdatafromfile[file2_num_of_row-1,id_num*9:id_num*9+9] = testdatafromstr2num
	else:
		testdatafromfile[file2_num_of_row-1,id_num*9:id_num*9+9] = row[2:]
	
	# Finish a cycle of one day.
	if(file2_num_of_row%18 == 0):
		file2_num_of_row=0
		id_num+=1
file2.close()

for frontiter in range(240):
	testdata[:,frontiter]=np.concatenate((testdatafromfile[:,frontiter*9+2],testdatafromfile[:,frontiter*9+3],testdatafromfile[:,frontiter*9+4],testdatafromfile[:,frontiter*9+5],testdatafromfile[:,frontiter*9+6],testdatafromfile[:,frontiter*9+7],testdatafromfile[:,frontiter*9+8]))
	# testdatafromfile[:,frontiter*9],testdatafromfile[:,frontiter*9+1],
testoutcome = np.dot(Weights,testdata)

file3 = open(file3name,'w',encoding='utf-8', errors='ignore')
file3_w = csv.writer(file3,lineterminator='\n')
file3_w.writerow(['id','value'])

for iter in range(240):
	id = 'id_'+str(iter)
	file3_w.writerow([id,testoutcome[iter],])

file3.close()

