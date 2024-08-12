# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:44:28 2022

@author: User
"""


# importing required libraries
import numpy as np
import pandas as pd

from keras.layers import Dense # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.models import model_from_json # saving and loading trained model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report # for generating a classification report of model
import pickle # saving and loading trained model
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing library for plotting
import matplotlib.pyplot as plt
# representation of model layers
from keras.utils import plot_model
from random import random

from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout 
Temperature=100
alpha=0.98


def randomWalk(s1,s2,v1,v2):
    s1=s2
    v1=v2
def greedyAcceptance(s1,s2,v1,v2):
    if(s2>s1):
        s1=s2
        v1=v2
        
def simulatedAnnealing(s1,s2,v1,v2):
    u=random()
    delta=s1-s2
    if(s2>s1 or u<np.exp(-delta/Temperature) ):
        s1=s2
        v1=v2
    
#Destruction Methods##################################################
def randomRemoval(v,sizeTargetAttributes):
    dmin=1
    dmax=5
    d=np.random.randint(dmin,dmax)
    for i in range(0,d):
      n= np.random.randint(0,len(v)-sizeTargetAttributes)
      v[n]=False
      
def shuffleRemoval(v,sizeTargetAttributes):
    while True:
        n1= np.random.randint(0,len(v)-sizeTargetAttributes)
        n2= np.random.randint(0,len(v)-sizeTargetAttributes)
        if n1!=n2:
            break
    tmp=v[n1]
    v[n1]=v[n2]
    v[n2]=tmp
    
def wrostRemoval(v,corr,sizeNumericAttributes):
   
   for i in range(0,sizeNumericAttributes) :
      if(v[i]==True):
          min=corr[i]
          index=i
          break
   for i in range(0,sizeNumericAttributes) :
       if(v[i]==True and corr[i]<min):
           min=corr[i]
           index=i
   v[index]=False
   
   
#Repair Methodes#####################################################
def randomInsertion(v,sizeTargetAttributes):
    insert=False
    nbr_it=0
    while insert==False and nbr_it<100 :
        nbr_it=nbr_it+1
        n= np.random.randint(0,len(v)-sizeTargetAttributes)
        if(v[n]==False):
            v[n]=True
            insert=True
def greedyInsertion(v,corr,sizeNumericAttributes):
    index=0
    for i in range(0,sizeNumericAttributes) :
       if(v[i]==False):
           max=corr[i]
           index=i
           break
    for i in range(0,sizeNumericAttributes) :
        if(v[i]==False and corr[i]>max):
            max=corr[i]
            index=i
    v[index]=True     
     
acceptanceMethods=[[1/3,0,0,0,0],[1/3,0,0,0,0],[1/3,0,0,0,0]]
destructionMethods=[[1/3,0,0,0,0],[1/3,0,0,0,0],[1/3,0,0,0,0]]
repairMethods=[[1/2,0,0,0,0],[1/2,0,0,0,0]]
#0:prob 1: weight 2:nbrfoisutilisé 3:succes 4:nbrfoisnon utilisé(historique)
#Adjustement des poids#######################################################
def adjutWeights(nbrIteration,pu,tab,initial):
   
    max=tab[0][4]
    index=0
    for i in range(1,len(tab)):
        if max<tab[i][4] :
            max=tab[i][4]
            index=i
    p=tab[index][0]/initial
    if p>=1:
        ro=random()
    else:
        ro=1-np.exp(np.log(p)*pu/nbrIteration)
    for i in range(0,len(tab)):
        print(tab[i][2])
        if (tab[i][2]!=0):
            tab[i][1]=(1-ro)*tab[i][0]+ro*tab[i][3]/tab[i][2]
        else:
            tab[i][1]=(1-ro)*tab[i][0]
    s=0
    for i in range (0,len(tab)):
        s=s+tab[i][1]
        
    for i in range (0,len(tab)):
         tab[i][0]=tab[i][1]/s
        







col_names = ["Processor_DPC_Rate","Processor_pct_ Idle_Time","Processor_pct_ C3_Time",
"Processor_pct_ Interrupt_Time","Processor_pct_ C2_Time","Processor_pct_ User_Time",
"Processor_pct_ C1_Time","Processor_pct_ Processor_Time","Processor_C1_ransitions_sec",
"Processor_pct_ DPC_Time","Processor_C2_ransitions_sec","Processor_pct_ Privileged_Time",
"Processor_C3_ransitions_sec","Processor_DPCs_Queued_sec","Processor_Interrupts_sec",
"Process_Pool_Paged Bytes","Process_IO Read_Operations_sec","Process_Working_Set_ Private",
"Process_Working_Set_Peak","Process_IO_Write Operations_sec","Process_Page_File Bytes",
"Process_pct_ User_Time","Process_Virtual_Bytes Peak","Process_Page_File Bytes Peak",
"Process_IO_Other_Bytes_sec","Process_Private_Bytes","Process_IO_Write_Bytes_sec",
"Process_Elapsed_Time","Process_Virtual_Bytes","Process_pct_ Processor_Time",
"Process_Creating Process ID","Process_Pool Nonpaged Bytes","Process_Working Set",
"Process_Page Faults_sec","Process_ID Process","Process_IO Other Operations_sec",
"Process_IO Data Operations_sec","Process_Thread Count","Process_pct_ Privileged_Time",
"Process_IO Data Bytes_sec","Process_IO Read Bytes_sec","Process_Priority Base",
"Process_Handle Count","Network_I(Intel R _82574L_GNC)TCP_APS",
"Network_I(Intel R _82574L_GNC) Packets Received Unknown",
"Network_I(Intel R _82574L_GNC) Bytes Received sec",
"Network_I(Intel R _82574L_GNC) Bytes Sent sec",
"Network_I(Intel R _82574L_GNC) Packets Outbound Errors",
"Network_I(Intel R _82574L_GNC) Packets Received Discarded",
"Network_I(Intel R _82574L_GNC) Bytes Total sec",
"Network_I(Intel R _82574L_GNC) Packets Outbound Discarded",
"Network_I(Intel R _82574L_GNC) TCP RSC Exceptions sec",
"Network_I(Intel R _82574L_GNC) Packets Sent Unicast sec",
"Network_I(Intel R _82574L_GNC) Output Queue Length",
"Network_I(Intel R _82574L_GNC) Packets Received sec",
"Network_I(Intel R _82574L_GNC) Current Bandwidth",
"Network_I(Intel R _82574L_GNC) Packets sec",
"Network_I(Intel R _82574L_GNC) TCP Active RSC Connections",
"Network_I(Intel R _82574L_GNC) Packets Sent sec",
"Network_I(Intel R _82574L_GNC) Packets Received Unicast sec",
"Network_I(Intel R _82574L_GNC) Packets Sent Non-Unicast sec",
"Network_I(Intel R _82574L_GNC) Packets Received Non-Unicast sec",
"Network_I(Intel R _82574L_GNC) TCP RSC Coalesced Packets sec",
"Network_I(Intel R _82574L_GNC) Offloaded Connections",
"Network_I(Intel R _82574L_GNC) Packets Received Errors",
"Memory Pool Paged Bytes","Memory Free & Zero Page List Bytes",
"Memory Cache Bytes Peak","Memory System Code Resident Bytes",
"Memory Available Bytes","Memory Commit Limit","Memory Transition Pages RePurposed sec",
"Memory Pages Output sec","Memory Page Reads sec","Memory Demand Zero Faults sec",
"Memory Available KBytes","Memory Pages sec","Memory Cache Bytes","Memory Pool Nonpaged Bytes",
"Memory Page Faults sec","Memory Transition Faults sec","Memory System Cache Resident Bytes",
"Memory Long-Term Average Standby Cache Lifetime (s)","Memory Standby Cache Reserve Bytes",
"Memory Page Writes sec","Memory System Code Total Bytes","Memory Standby Cache Core Bytes",
"Memory System Driver Resident Bytes","Memory Standby Cache Normal Priority Bytes",
"Memory Pool Paged Allocs","Memory Pool Nonpaged Allocs","Memory pct_ Committed Bytes In Use",
"Memory Free System Page Table Entries","Memory Available MBytes","Memory Modified Page List Bytes",
"Memory Cache Faults sec","Memory Committed Bytes","Memory System Driver Total Bytes",
"Memory Pages Input sec","Memory Pool Paged Resident Bytes","Memory Write Copies sec",
"LogicalDisk(_Total) Avg  Disk Bytes Write","LogicalDisk(_Total) pct_ Idle Time",
"LogicalDisk(_Total) Disk Reads sec","LogicalDisk(_Total) pct_ Free Space",
"LogicalDisk(_Total) Disk Read Bytes sec","LogicalDisk(_Total) Avg  Disk sec Read",
"LogicalDisk(_Total) Disk Writes sec","LogicalDisk(_Total) Current Disk Queue Length",
"LogicalDisk(_Total) Split IO Sec","LogicalDisk(_Total) Free Megabytes",
"LogicalDisk(_Total) Avg  Disk sec Write","LogicalDisk(_Total) Disk Bytes sec",
"LogicalDisk(_Total) Avg  Disk Read Queue Length","LogicalDisk(_Total) pct_ Disk Time",
"LogicalDisk(_Total) Avg  Disk Bytes Read","LogicalDisk(_Total) Avg  Disk Write Queue Length",
"LogicalDisk(_Total) Avg  Disk Queue Length","LogicalDisk(_Total) pct_ Disk Read Time",
"LogicalDisk(_Total) Disk Write Bytes sec","LogicalDisk(_Total) Disk Transfers sec",
"LogicalDisk(_Total) Avg  Disk Bytes Transfer","LogicalDisk(_Total) pct_ Disk Write Time",
"LogicalDisk(_Total) Avg  Disk sec Transfer","label","type"]


# importing dataset
data = pd.read_csv('ToN-IoTwindows10.csv',header=None, names=col_names)
# print dataset
data

data.drop(['label'],axis=1,inplace=True)
data.shape
data.replace(' ', float('nan'), inplace=True)
data.dropna(inplace=True)


# descriptive statistics of dataset
data.describe()

# number of attack labels 
data['type'].value_counts()
data.replace([np.inf, -np.inf], np.nan, inplace=True)



# selecting numeric attributes columns from data
numeric_col = ["Processor_DPC_Rate","Processor_pct_ Idle_Time","Processor_pct_ C3_Time",
"Processor_pct_ Interrupt_Time","Processor_pct_ C2_Time","Processor_pct_ User_Time",
"Processor_pct_ C1_Time","Processor_pct_ Processor_Time","Processor_C1_ransitions_sec",
"Processor_pct_ DPC_Time","Processor_C2_ransitions_sec","Processor_pct_ Privileged_Time",
"Processor_C3_ransitions_sec","Processor_DPCs_Queued_sec","Processor_Interrupts_sec",
"Process_Pool_Paged Bytes","Process_IO Read_Operations_sec","Process_Working_Set_ Private",
"Process_Working_Set_Peak","Process_IO_Write Operations_sec","Process_Page_File Bytes",
"Process_pct_ User_Time","Process_Virtual_Bytes Peak","Process_Page_File Bytes Peak",
"Process_IO_Other_Bytes_sec","Process_Private_Bytes","Process_IO_Write_Bytes_sec",
"Process_Elapsed_Time","Process_Virtual_Bytes","Process_pct_ Processor_Time",
"Process_Creating Process ID","Process_Pool Nonpaged Bytes","Process_Working Set",
"Process_Page Faults_sec","Process_ID Process","Process_IO Other Operations_sec",
"Process_IO Data Operations_sec","Process_Thread Count","Process_pct_ Privileged_Time",
"Process_IO Data Bytes_sec","Process_IO Read Bytes_sec","Process_Priority Base",
"Process_Handle Count","Network_I(Intel R _82574L_GNC)TCP_APS",
"Network_I(Intel R _82574L_GNC) Packets Received Unknown",
"Network_I(Intel R _82574L_GNC) Bytes Received sec",
"Network_I(Intel R _82574L_GNC) Bytes Sent sec",
"Network_I(Intel R _82574L_GNC) Packets Outbound Errors",
"Network_I(Intel R _82574L_GNC) Packets Received Discarded",
"Network_I(Intel R _82574L_GNC) Bytes Total sec",
"Network_I(Intel R _82574L_GNC) Packets Outbound Discarded",
"Network_I(Intel R _82574L_GNC) TCP RSC Exceptions sec",
"Network_I(Intel R _82574L_GNC) Packets Sent Unicast sec",
"Network_I(Intel R _82574L_GNC) Output Queue Length",
"Network_I(Intel R _82574L_GNC) Packets Received sec",
"Network_I(Intel R _82574L_GNC) Current Bandwidth",
"Network_I(Intel R _82574L_GNC) Packets sec",
"Network_I(Intel R _82574L_GNC) TCP Active RSC Connections",
"Network_I(Intel R _82574L_GNC) Packets Sent sec",
"Network_I(Intel R _82574L_GNC) Packets Received Unicast sec",
"Network_I(Intel R _82574L_GNC) Packets Sent Non-Unicast sec",
"Network_I(Intel R _82574L_GNC) Packets Received Non-Unicast sec",
"Network_I(Intel R _82574L_GNC) TCP RSC Coalesced Packets sec",
"Network_I(Intel R _82574L_GNC) Offloaded Connections",
"Network_I(Intel R _82574L_GNC) Packets Received Errors",
"Memory Pool Paged Bytes","Memory Free & Zero Page List Bytes",
"Memory Cache Bytes Peak","Memory System Code Resident Bytes",
"Memory Available Bytes","Memory Commit Limit","Memory Transition Pages RePurposed sec",
"Memory Pages Output sec","Memory Page Reads sec","Memory Demand Zero Faults sec",
"Memory Available KBytes","Memory Pages sec","Memory Cache Bytes","Memory Pool Nonpaged Bytes",
"Memory Page Faults sec","Memory Transition Faults sec","Memory System Cache Resident Bytes",
"Memory Long-Term Average Standby Cache Lifetime (s)","Memory Standby Cache Reserve Bytes",
"Memory Page Writes sec","Memory System Code Total Bytes","Memory Standby Cache Core Bytes",
"Memory System Driver Resident Bytes","Memory Standby Cache Normal Priority Bytes",
"Memory Pool Paged Allocs","Memory Pool Nonpaged Allocs","Memory pct_ Committed Bytes In Use",
"Memory Free System Page Table Entries","Memory Available MBytes","Memory Modified Page List Bytes",
"Memory Cache Faults sec","Memory Committed Bytes","Memory System Driver Total Bytes",
"Memory Pages Input sec","Memory Pool Paged Resident Bytes","Memory Write Copies sec",
"LogicalDisk(_Total) Avg  Disk Bytes Write","LogicalDisk(_Total) pct_ Idle Time",
"LogicalDisk(_Total) Disk Reads sec","LogicalDisk(_Total) pct_ Free Space",
"LogicalDisk(_Total) Disk Read Bytes sec","LogicalDisk(_Total) Avg  Disk sec Read",
"LogicalDisk(_Total) Disk Writes sec","LogicalDisk(_Total) Current Disk Queue Length",
"LogicalDisk(_Total) Split IO Sec","LogicalDisk(_Total) Free Megabytes",
"LogicalDisk(_Total) Avg  Disk sec Write","LogicalDisk(_Total) Disk Bytes sec",
"LogicalDisk(_Total) Avg  Disk Read Queue Length","LogicalDisk(_Total) pct_ Disk Time",
"LogicalDisk(_Total) Avg  Disk Bytes Read","LogicalDisk(_Total) Avg  Disk Write Queue Length",
"LogicalDisk(_Total) Avg  Disk Queue Length","LogicalDisk(_Total) pct_ Disk Read Time",
"LogicalDisk(_Total) Disk Write Bytes sec","LogicalDisk(_Total) Disk Transfers sec",
"LogicalDisk(_Total) Avg  Disk Bytes Transfer","LogicalDisk(_Total) pct_ Disk Write Time",
"LogicalDisk(_Total) Avg  Disk sec Transfer"]
# using standard scaler for normalizing
std_scaler = StandardScaler()
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
  return df
# data before normalization
data.head()
# calling the normalization() function
data = normalization(data.copy(),numeric_col)
# data after normalization
data.head()

bin_label = pd.DataFrame(data.type.map(lambda x:'normal' if x=='normal' else 'anormal'))
# creating a dataframe with binary labels (normal,abnormal)
bin_data = data.copy()
bin_data['type'] = bin_label
# label encoding (0,1) binary labels (abnormal,normal)
le2 = preprocessing.LabelEncoder()
enc_label = bin_label.apply(le2.fit_transform)
bin_data['intrusion'] = enc_label
le2.classes_
np.save("labels/le1_classes.npy",le2.classes_,allow_pickle=True)
# dataset with binary labels and label encoded column
bin_data.head()


np.save("le2_classes.npy",le2.classes_,allow_pickle=True)
# one-hot-encoding attack label
bin_data = pd.get_dummies(bin_data,columns=['type'],prefix="",prefix_sep="") 
bin_data['type'] = bin_label
bin_data
# pie chart distribution of multi-class labels
plt.figure(figsize=(8,8))
plt.pie(bin_data.type.value_counts(),labels=bin_data.type.unique(),autopct='%0.2f%%')
plt.title('Pie chart distribution of multi-class labels')
plt.legend()
plt.savefig('plots/Pie_chart_multi.png')
plt.show()

bin_data.drop(labels= [ 'type'], axis=1, inplace=True)
bin_data






# creating a dataframe with only numeric attributes of multi-class dataset and encoded label attribute 
numeric_multi = bin_data[numeric_col]



# then joining encoded, one-hot-encoded, and original attack label attribute
bin_data = numeric_multi.join(bin_data[['intrusion','anormal','normal']])
bin_data= bin_data.apply(lambda col: col.fillna(col.mean()), axis=0)
# saving final dataset to disk
bin_data.to_csv('./datasets/bin_data.csv')

# final dataset for multi-class classification
bin_data




# from sklearn.feature_selection import mutual_info_classif


# # Function to calculate information gain
# def calculate_information_gain(data, target_column, exclude_columns=None):
#     if exclude_columns is None:
#         exclude_columns = []
    
#     # Extracting the features and target
#     X = data.drop(columns=[target_column] + exclude_columns)
#     y = data[target_column]
    
#     # Filling NaN values with the mean of the column
#     X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    
#     # Calculating mutual information
#     info_gain = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    
#     info_gain_vector=np.array(info_gain)
#     return info_gain_vector

# # Applying the function to the bin_data and excluding 'normal' and 'abnormal' attributes
# info_gain_vector = calculate_information_gain(bin_data, 'intrusion', exclude_columns=['normal', 'anormal'])

# print(info_gain_vector)
# # Function to calculate Pearson correlation
# def calculate_pearson_correlation(data, target_column, exclude_columns=None):
#     if exclude_columns is None:
#         exclude_columns = []
    
#     # Dropping the target column and any columns to exclude
#     X = data.drop(columns=[target_column] + exclude_columns)
#     y = data[target_column]
    
#     # Calculating Pearson correlation
#     correlation = X.apply(lambda x: x.corr(y))
#     correlation = correlation.fillna(-1)
    
#     # Returning as a numpy array
#     correlation_vector = correlation.to_numpy()
#     return correlation_vector

# # Applying the function to the bin_data and excluding 'normal' and 'abnormal' attributes
# correlation_vector = calculate_pearson_correlation(bin_data, 'intrusion', exclude_columns=['normal', 'anormal'])

# # Display the Pearson correlation DataFrame
# print(correlation_vector)




# bool_arr = np.random.choice([True, False],size=127)

# for i in range(124,127):
#     bool_arr[i]=False
    
# solution=bool_arr
# solutionFinalV=solution
# bestV=solution

# best=0
# nbr_itr=0
# pu=5
# solutionFinalAcc=0
# lamda1=0.75
# lamda2=0.5
# lamda3=0.25

# while(Temperature>0.001 and nbr_itr<1000 and best<1):
#     for i in range(0,3):
#         destructionMethods[i][4]=0
#     for i in range(0,2):
#         repairMethods[i][4]=0
#     for i in range(0,3):
#         acceptanceMethods[i][4]=0
    
#     for i in range(0,pu):
#         u=random()
#         f1=repairMethods[0][0]
#         f2=f1+repairMethods[1][0]
      
        
#         if(u<f1):
#             rSelect=0
#         else:
#             rSelect=1
           
#         u=random()
#         f1=destructionMethods[0][0]
#         f2=f1+destructionMethods[1][0]
#         f3=f2+destructionMethods[2][0]
       
        
#         alpha = np.random.rand()

#         # Calculate the score vector
#         score_vector = alpha * info_gain_vector + (1 - alpha) * correlation_vector
        
        
        
        
#         if(u<f1):
#               dSelect=0
             
#         else:
#               if u<f2:
#                   dSelect=1
#               else:
#                   dSelect=2
#         destructionMethods[dSelect][2]=destructionMethods[dSelect][2]+1
#         for i in range(0,3):
#             if i==dSelect:
#                 destructionMethods[i][4]=0
#             else:
#                 destructionMethods[i][4]=destructionMethods[i][4]+1
                
                   
#         if(dSelect==0):
#             randomRemoval(solution,3)
#         else:
#             if (dSelect==1):
                
#                 shuffleRemoval(solution,3)
#             else:
#                 wrostRemoval(solution,score_vector,124)
                
#         if rSelect==0:
#             randomInsertion(solution,3)
#         else:
#             greedyInsertion(solution,score_vector,124)
#         repairMethods[rSelect][2]=repairMethods[rSelect][2]+1
#         for i in range(0,2):
#             if i==rSelect:
#                 repairMethods[i][4]=0
#             else:
#                 repairMethods[i][4]=repairMethods[i][4]+1   
       
       
    
#         print("entrain de commencer l'entrainement")
#         X = bin_data.iloc[:,solution]  # dataset excluding target attribute (encoded, one-hot-encoded,original)
#         Y = bin_data[['intrusion']] # target attributes
#         # splitting the dataset 75% for training and 25% testing
       
#         from sklearn.preprocessing import LabelBinarizer

#         Y = LabelBinarizer().fit_transform(Y)
#         Y
#         X=np.array(X)
#         Y=np.array(Y)
#         X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=42)
#         X_train2=pd.DataFrame(X_train)
#         y_train2=pd.DataFrame(y_train)
#         X_test2=pd.DataFrame(X_test)
#         y_test2=pd.DataFrame(y_test)
#         X_train2.to_csv('./datasets/X_train.csv')
#         X_test2.to_csv('./datasets/X_test.csv')
#         y_train2.to_csv('./datasets/y_train.csv')
#         y_test2.to_csv('./datasets/y_test.csv')

#         X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#         print(X_train.shape)
#         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#         print(X_test.shape)

  
#         model = Sequential() # initializing model
#         # input layer and first layer with 50 neurons
#         model.add(Conv1D(16, 3, padding="same",input_shape = (X_train.shape[1], 1), activation='relu'))
#         model.add(MaxPool1D(pool_size=(2)))  
#         #model.add(Dropout(0.2))
#         model.add(Conv1D(32, 3, padding="same", activation='relu'))
#         model.add(MaxPool1D(pool_size=(2)))
        
#         model.add(Conv1D(64, 3, padding="same", activation='relu'))
#         model.add(MaxPool1D(pool_size=(2)))  
  
#         #model.add(Dropout(0.2))
#         model.add(Flatten())
#         model.add(Dense(units=50))
#         # output layer with softmax activation
#         model.add(Dense(units=1,activation='sigmoid')) 
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         model.summary()
#         history = model.fit(X_train, y_train, epochs=5, batch_size=5000,validation_split=0.2)          
#         # defining loss function, optimizer, metrics and then compiling model
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         # predicting target attribute on testing dataset
#         test_results = model.evaluate(X_test, y_test, verbose=1)
#         print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')  
          

#         u=random()
#         f1=acceptanceMethods[0][0]
#         f2=f1+acceptanceMethods[1][0]
#         f3=f2+acceptanceMethods[2][0]
#         if(u<f1):
#             aSelected=0
#         else:
#             if u<f2:
#                 aSelected=1
#             else:
#                 aSelected=2

#         if aSelected==0:
#             randomWalk(solutionFinalAcc,test_results[1]*100 ,solution, solutionFinalV)
#         else:
#             if aSelected==1:
#                 greedyAcceptance(solutionFinalAcc,test_results[1]*100 ,solution, solutionFinalV)
#             else:
#                 simulatedAnnealing(solutionFinalAcc,test_results[1]*100 ,solution, solutionFinalV)
#                 Temperature=Temperature*0.99
#         acceptanceMethods[aSelected][2]=acceptanceMethods[aSelected][2]+1
#         for i in range(0,3):
#             if i==aSelected:
#                 acceptanceMethods[i][4]=0
#             else:
#                 acceptanceMethods[i][4]=acceptanceMethods[i][4]+1    
#         if test_results[1]>best:
#             best=test_results[1]
#             bestV=solution
#             destructionMethods[dSelect][3]=destructionMethods[dSelect][3]+lamda1
#             repairMethods[rSelect][3]=repairMethods[rSelect][3]+lamda1
#             acceptanceMethods[aSelected][3]=acceptanceMethods[aSelected][3]+lamda1
    
#         else:
#             if test_results[1]>solutionFinalAcc:
#                 destructionMethods[dSelect][3]=destructionMethods[dSelect][3]+lamda2
#                 repairMethods[rSelect][3]=repairMethods[rSelect][3]+lamda2
#                 acceptanceMethods[aSelected][3]=acceptanceMethods[aSelected][3]+lamda2
#             else:
#                 destructionMethods[dSelect][3]=destructionMethods[dSelect][3]+lamda3
#                 repairMethods[rSelect][3]=repairMethods[rSelect][3]+lamda3
#                 acceptanceMethods[aSelected][3]=acceptanceMethods[aSelected][3]+lamda3
    
    
#     adjutWeights(1000, 10, acceptanceMethods, 1/3)
#     adjutWeights(1000, 10, destructionMethods, 1/3)
#     adjutWeights(1000, 10, repairMethods, 1/2)
#     nbr_itr=nbr_itr+1
#     print("########################################################it ",nbr_itr)    
  





bestV=[True, True, False, False, False, True, True, True, False, True, False, True, False, False, True, True, False, False, False, False, True, True, True, False, False, False, False, False, True, True, False, False, True, False, False, True, False, False, True, False, False, False, True, True, False, False, True, False, False, False, False, False, True, False, False, False, True, False, True, True, False, False, False, False, False, False, True, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, True, True, False, False, False, False, True, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False,False]


X = bin_data.iloc[:,bestV]  # dataset excluding target attribute (encoded, one-hot-encoded,original)
Y = bin_data[['intrusion']] # target attributes
# splitting the dataset 75% for training and 25% testing

from sklearn.preprocessing import LabelBinarizer

Y = LabelBinarizer().fit_transform(Y)
Y
X=np.array(X)
Y=np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=42)
X_train2=pd.DataFrame(X_train)
y_train2=pd.DataFrame(y_train)
X_test2=pd.DataFrame(X_test)
y_test2=pd.DataFrame(y_test)
X_train2.to_csv('./datasets/X_train.csv')
X_test2.to_csv('./datasets/X_test.csv')
y_train2.to_csv('./datasets/y_train.csv')
y_test2.to_csv('./datasets/y_test.csv')

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)


# model = Sequential() # initializing model
# # input layer and first layer with 50 neurons
# model.add(Conv1D(16, 3, padding="same",input_shape = (X_train.shape[1], 1), activation='relu'))
# model.add(MaxPool1D(pool_size=(2)))  
# #model.add(Dropout(0.2))
# model.add(Conv1D(32, 3, padding="same", activation='relu'))
# model.add(MaxPool1D(pool_size=(2)))
 
# model.add(Conv1D(64, 3, padding="same", activation='relu'))
# model.add(MaxPool1D(pool_size=(2)))  

# #model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(units=50))
# # output layer with softmax activation
# model.add(Dense(units=1,activation='sigmoid')) 
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
# history = model.fit(X_train, y_train, epochs=1000, batch_size=5000,validation_split=0.2)
# filepath = './models/cnn_multi.json'
# weightspath = './weights/cnn_multi.h5'
# if (not path.isfile(filepath)):
#   # serialize model to JSON
#   cnn_json = model.to_json()
#   with open(filepath, "w") as json_file:
#     json_file.write(cnn_json)

#   # serialize weights to HDF5
#   model.save_weights(weightspath)
#   print("Saved model to disk")
 
filepath = './models/Binary/ToN-IoTwindows10/cnn_multi.json'
weightspath = './weights/Binary/ToN-IoTwindows10/cnn_multi.h5'  
# load json and create model
json_file = open(filepath, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(weightspath)
print("Loaded model from disk")

# defining loss function, optimizer, metrics and then compiling model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# predicting target attribute on testing dataset
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')  
  
 
plot_model(model, to_file='plots/cnn_binary.png', show_shapes=True,)
            
y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = auc(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('plots/cnn_binary_roc.png')
plt.show()
            
pred = model.predict(X_test)
y_classes = (model.predict(X_test)>0.5).astype('int32')
print("Recall Score - ",recall_score(y_test,y_classes))
print("F1 Score - ",f1_score(y_test,y_classes))
print("Precision Score - ",precision_score(y_test,y_classes))















