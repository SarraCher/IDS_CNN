# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:38:55 2022

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
   print("entre wrost removal")
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
   print("sortir wrostremoval")
   
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
    print("dans adjust***************************************")
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
        


#algorithme général#########################################################

col_names = ["frame.time","ip.src_host","ip.dst_host","arp.dst.proto_ipv4",
              "arp.opcode","arp.hw.size","arp.src.proto_ipv4","icmp.checksum","icmp.seq_le",
              "icmp.transmit_timestamp","icmp.unused","http.file_data","http.content_length",
              "http.request.uri.query","http.request.method","http.referer","http.request.full_uri",
              "http.request.version","http.response","http.tls_port","tcp.ack","tcp.ack_raw",
              "tcp.checksum","tcp.connection.fin","tcp.connection.rst","tcp.connection.syn",
              "tcp.connection.synack","tcp.dstport","tcp.flags","tcp.flags.ack","tcp.len",
              "tcp.options","tcp.payload","tcp.seq","tcp.srcport","udp.port","udp.stream",
              "udp.time_delta","dns.qry.name","dns.qry.name.len","dns.qry.qu","dns.qry.type",
              "dns.retransmission","dns.retransmit_request","dns.retransmit_request_in",
              "mqtt.conack.flags","mqtt.conflag.cleansess","mqtt.conflags","mqtt.hdrflags",
              "mqtt.len","mqtt.msg_decoded_as","mqtt.msg","mqtt.msgtype","mqtt.proto_len",
              "mqtt.protoname","mqtt.topic","mqtt.topic_len","mqtt.ver","mbtcp.len",
              "mbtcp.trans_id","mbtcp.unit_id","Attack_label","Attack_type"]





# importing dataset
data = pd.read_csv('Edge-IIoTset.csv',header=None, names=col_names)
#print dataset
data


data.drop(0, axis=0, inplace=True)






data.describe()

data.head(5)

print(data['Attack_type'].value_counts())

from sklearn.utils import shuffle

drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4", 

          "http.file_data","http.request.full_uri","icmp.transmit_timestamp",

          "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",

          "tcp.dstport", "udp.port", "mqtt.msg"]

data.drop(drop_columns, axis=1, inplace=True)
data.dropna(axis=0, how='any', inplace=True)

data.drop_duplicates(subset=None, keep="first", inplace=True)

#data = shuffle(data)

data.isna().sum()

print(data['Attack_type'].value_counts())
def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)

encode_text_dummy(data,'http.request.method')

encode_text_dummy(data,'http.referer')

encode_text_dummy(data,"http.request.version")

encode_text_dummy(data,"dns.qry.name.len")

encode_text_dummy(data,"mqtt.conack.flags")

encode_text_dummy(data,"mqtt.protoname")

encode_text_dummy(data,"mqtt.topic")
    
data              
data.to_csv('preprocessed_Data.csv', encoding='utf-8', index=False)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_csv('./preprocessed_Data.csv', low_memory=False) 
#df


bin_label = pd.DataFrame(data.Attack_type.map(lambda x:'normal' if x=='Normal' else 'anormal'))
# creating a dataframe with binary labels (normal,abnormal)
bin_data = data.copy()
bin_data['Attack_type'] = bin_label
# label encoding (0,1) binary labels (abnormal,normal)
le2 = preprocessing.LabelEncoder()
enc_label = bin_label.apply(le2.fit_transform)
bin_data['intrusion'] = enc_label
le2.classes_
np.save("labels/le1_classes.npy",le2.classes_,allow_pickle=True)
# dataset with binary labels and label encoded column
bin_data.head()


# pie chart distribution of multi-class labels
plt.figure(figsize=(8,8))
plt.pie(bin_data.Attack_type.value_counts(),labels=bin_data.Attack_type.unique(),autopct='%0.2f%%')
plt.title('Pie chart distribution of multi-class labels')
plt.legend()
plt.savefig('plots/Pie_chart_multi.png')
plt.show()



feat_cols = list(bin_data.columns)

label_col="Attack_type"
feat_cols.remove(label_col)
#feat_cols
empty_cols = [col for col in bin_data.columns if bin_data[col].isnull().all()]
empty_cols
skip_list = ["icmp.unused", "http.tls_port", "dns.qry.type", "mqtt.msg_decoded_as"]
bin_data[skip_list[3]].value_counts()

X = bin_data.drop(label_col, axis=1)
y = bin_data['intrusion']




 




std_scaler = StandardScaler()
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
   
  return df
# data before normalization

# calling the normalization() function


X= normalization(X.copy(),X.columns)
X_data = X.copy()





# from sklearn.feature_selection import mutual_info_classif


# # Function to calculate information gain
# def calculate_information_gain(X, y):
#     # Filling NaN values with the mean of the column
#     X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    
#     # Calculating mutual information
#     info_gain = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    
#     info_gain_vector = np.array(info_gain)
#     return info_gain_vector



# info_gain_vector = calculate_information_gain(X, y)
# print(info_gain_vector)



# # Function to calculate Pearson correlation
# def calculate_pearson_correlation(X, y):
#     # Calculating Pearson correlation
#     correlation = X.apply(lambda x: x.corr(y))
#     correlation = correlation.fillna(-1)
    
#     # Returning as a numpy array
#     correlation_vector = correlation.to_numpy()
#     return correlation_vector

# # Example usage with bin_data DataFrame
# correlation_vector = calculate_pearson_correlation(X, y)
# print(correlation_vector)



# bool_arr = np.random.choice([True, False],size=97)


# bool_arr[96]=False
    
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

# while(Temperature>0.001 and nbr_itr<100 and best<1):
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
#             randomRemoval(solution,2)
#         else:
#             if (dSelect==1):
                
#                 shuffleRemoval(solution,2)
#             else:
#                 wrostRemoval(solution,score_vector,96)
                
#         if rSelect==0:
#             randomInsertion(solution,2)
#         else:
#             greedyInsertion(solution,score_vector,96)
#         repairMethods[rSelect][2]=repairMethods[rSelect][2]+1
#         for i in range(0,2):
#             if i==rSelect:
#                 repairMethods[i][4]=0
#             else:
#                 repairMethods[i][4]=repairMethods[i][4]+1   
       
       
    
#         print("entrain de commencer l'entrainement")
#         X = X_data.iloc[:,solution]  
#         if 'Attack_label' in X.columns:
#             X = X.drop('Attack_label', axis=1)
#         Y = bin_data[['intrusion']] 
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
#         history = model.fit(X_train, y_train, epochs=1, batch_size=5000,validation_split=0.2)          
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
    
    
#     adjutWeights(100, 10, acceptanceMethods, 1/3)
#     adjutWeights(100, 10, destructionMethods, 1/3)
#     adjutWeights(100, 10, repairMethods, 1/2)
#     nbr_itr=nbr_itr+1
#     print("########################################################it ",nbr_itr)    
  













bestV=[True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, True, False, True, True, False, True, True, True, True, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]

X= X_data.iloc[:,bestV]  
if 'Attack_label' in X.columns:
    X = X.drop('Attack_label', axis=1)

y = bin_data[['intrusion']] 
print("longueyr",len(X.columns))

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train2=pd.DataFrame(X_train)
y_train2=pd.DataFrame(y_train)
X_test2=pd.DataFrame(X_test)
y_test2=pd.DataFrame(y_test)
X_train2.to_csv('./datasets/X_train.csv')
X_test2.to_csv('./datasets/X_test.csv')
y_train2.to_csv('./datasets/y_train.csv')
y_test2.to_csv('./datasets/y_test.csv')

# from sklearn.preprocessing import LabelEncoder


# label_encoder = LabelEncoder()
# y_train =  label_encoder.fit_transform(y_train)
# y_test = label_encoder.transform(y_test)
# label_encoder.classes_

import numpy as np

# assuming X_train and X_test are DataFrames
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
input_shape = X_train.shape[1:]
print(X_train.shape)
print(X_test.shape)
num_classes = len(np.unique(y_train))
num_classes

# from  tensorflow.keras.utils import to_categorical 

# y_train = to_categorical(y_train, num_classes=num_classes)
# y_test = to_categorical(y_test, num_classes=num_classes)
print(y_train.shape, y_test.shape)




X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

# model = Sequential() # initializing model
# # input layer and first layer with 50 neurons
# model.add(Conv1D(16, 3, padding="same",input_shape = input_shape, activation='relu'))
# model.add(MaxPool1D(pool_size=(2)))  
# #model.add(Dropout(0.2))
# model.add(Conv1D(32, 3, padding="same", activation='relu'))
# model.add(MaxPool1D(pool_size=(2)))
 
# model.add(Conv1D(64, 3, padding="same", activation='relu'))
# model.add(MaxPool1D(pool_size=(2)))  
# model.add(Flatten())
# #model.add(Dropout(0.2))
# model.add(Dense(units=50))
# # output layer with softmax activation
# model.add(Dense(units=1,activation='sigmoid')) 
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
# history = model.fit(X_train, y_train, epochs=100, batch_size=5000,validation_split=0.2)
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
 
  
filepath = './models/Binary/Edge-IIoTset/cnn_multi.json'
weightspath = './weights/Binary/Edge-IIoTset/cnn_multi.h5' 
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




















