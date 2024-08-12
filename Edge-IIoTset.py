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
def accuracyCalcul(matri,classes):
    sumAccuracy=0
    taille=np.size(classes)
    for i in range(0,taille):
        TP=matri[i][i]
        FN=0
        for j in range(0,taille):
            if(i!=j):
                FN=FN+matri[i][j]
        FP=0        
        for k in range(0,taille):
            if(k!=i):
                FP=FP+matri[k][i]
        TN=0
        for n in range(0,taille):
            for m in range(0,taille):
                if(n!=i and m!=i):
                  TN=TN+matri[n][m]
        print("TP",TP)
        print("FP",FP)
        print("TN",TN)
        print("FN",FN)
        Accuracy=((TP+TN)/(TP+TN+FP+FN))*100
        FauxAlerte=(FP/(FP+TN))*100
        sumAccuracy=sumAccuracy+Accuracy
      
        print(classes[i]," accuracy is ",Accuracy)
      #  print(classes[i]," faux alerte is ",FauxAlerte)
        print("***********************************")

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


data2 = data.copy()
multi_label = pd.DataFrame(data2.Attack_type)
# label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
le2 = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
data2['intrusion'] = enc_label



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






feat_cols = list(data.columns)

label_col="Attack_type"
feat_cols.remove(label_col)
#feat_cols
empty_cols = [col for col in data.columns if data[col].isnull().all()]
empty_cols
skip_list = ["icmp.unused", "http.tls_port", "dns.qry.type", "mqtt.msg_decoded_as"]
data[skip_list[3]].value_counts()
X = data.drop(label_col, axis=1)
y = data[label_col]


y.value_counts()
print("Number of samples in X:", X.shape[0])
print("Number of samples in y:", y.shape[0])
#data balancing with randomover  sampling
 

from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore")
minority_classes = ['Port_Scanning', 'XSS', 'Ransomware', 'Fingerprinting', 'MITM']
desired_samples = {
    'Port_Scanning': 20000,
    'XSS': 20000,
    'Ransomware': 20000,
    'Fingerprinting': 20000,
    'MITM': 20000
}
mask = data[label_col].isin(minority_classes)
minority_mask = data[label_col].isin(minority_classes)

X_minority = X[minority_mask]
y_minority = y[minority_mask]

oversample = RandomOverSampler(sampling_strategy={k: desired_samples[k] if k in desired_samples else 'auto' for k in minority_classes},
                                random_state=42)
X_oversampled, y_oversampled = oversample.fit_resample(X[mask], y[mask])

# Concatenate the oversampled data with the original data
X_balanced = pd.concat([X, X_oversampled])
y_balanced = pd.concat([y, y_oversampled])
y.value_counts()
y_balanced.value_counts()


std_scaler = StandardScaler()
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
   
  return df
# data before normalization

# calling the normalization() function


X_balanced = normalization(X_balanced.copy(),X_balanced.columns)

y_balanced = pd.get_dummies(y_balanced,columns=[label_col])
y_balanced.head()
print(X_balanced.columns)


#these values are calculated in Edge_IIoTsetBinary.py

# info_gain_vect = [
#     0.00161666, 0.0012152, 0.0918817, 0.0999622, 0, 0.0411834, 0.0698899, 
#     0.000109006, 0.359178, 0.374717, 0.140031, 0.0922167, 0.04815, 0.100129, 
#     0.050185, 0.237712, 0.196408, 0.211578, 0.310615, 0.129455, 0.0119398, 
#     0.0305872, 0.00352749, 0, 0.00631651, 0.00148206, 0.00157742, 0.06149, 
#     0.0620217, 0.130807, 0.121323, 0, 0.130674, 0.0620978, 0.0622906, 
#     0.0617593, 0.00288509, 0.00155292, 0.00135953, 0.797758, 0.0636401, 
#     0.240004, 0.0676067, 0.00029949, 0.00620599, 0.000463721, 0, 
#     0.000534005, 0.00332507, 0.0543415, 0.00226059, 0.10545, 0.0026326, 
#     0.000123798, 0.0636376, 0.000670659, 0.000540309, 0.000240172, 
#     0.000290483, 0.239547, 5.99672e-06, 0.000565493, 0.0137159, 0.0433005, 
#     0.000182992, 0.000100079, 0.000225696, 0.12895, 0.0153648, 0.00286489, 
#     0.00270731, 0.00314932, 0.00297522, 0.00283819, 0.000928237, 0.000255978, 
#     0.00132183, 0.0153482, 0.656835, 0.0616703, 0.000659001, 0.000980402, 
#     0.000960051, 0, 0.000423195, 0.000703661, 0.000165407, 0.000208431, 
#     0.000780413, 0.000603803, 0.0148119, 0.65642, 0.0618855, 0.0150003, 
#     0.656675, 0.0613725
# ]
# info_gain_vector=np.array(info_gain_vect)

# correlation_vect= [
#     -0.00477262, -0.00498575, -0.263178, -0.28476, -1, -0.0850881, -0.208552, 
#     -1, -0.180578, 0.283569, 0.249135, 0.108279, -0.0737566, -0.161896, 
#     0.0723836, 0.417327, 0.456126, 0.0738063, 0.167464, -0.336906, 0.0174929, 
#     -0.10559, -0.0133185, -1, 0.0140408, 0.00316944, 0.00286361, 0.133358, 
#     0.133358, 0.187912, 0.175727, -1, 0.187912, 0.133358, 0.133356, 0.133358, 
#     0.00511449, 0.00482602, 0.00461207, -1, -0.195534, -0.526648, -0.201511, 
#     -0.00280517, -0.0398814, -0.00229041, -0.00161956, -0.00161956, 
#     -0.0225311, -0.137312, -0.0155772, -0.332417, -0.0224143, -0.00161956, 
#     -0.194467, -0.00229041, -0.00198355, -0.00161956, -0.00161956, -0.527197, 
#     -0.00161956, -0.0011452, -0.0666577, -0.195404, -0.00229041, -0.00161956, 
#     -0.00198355, 0.225065, -0.0829322, 0.00960586, -0.0172552, 0.00953083, 
#     0.00954159, 0.00958448, 0.00110906, 0.00110906, 0.0027911, -0.0847179, 
#     0.903269, 0.133357, 0.000640316, 0.000640316, 0.000905544, 0.000905544, 
#     0.000784224, 0.000784224, 0.000640316, 0.000640316, 0.00135832, 
#     0.00135832, -0.0847179, 0.903311, 0.133358, -0.0847179, 0.903314, 
#     0.133356
# ]
# correlation_vector= np.array(correlation_vect)
# bool_arr = np.random.choice([True, False],size=96)

    
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
#             randomRemoval(solution,0)
#         else:
#             if (dSelect==1):
                
#                 shuffleRemoval(solution,0)
#             else:
#                 wrostRemoval(solution,score_vector,96)
                
#         if rSelect==0:
#             randomInsertion(solution,0)
#         else:
#             greedyInsertion(solution,score_vector,96)
#         repairMethods[rSelect][2]=repairMethods[rSelect][2]+1
#         for i in range(0,2):
#             if i==rSelect:
#                 repairMethods[i][4]=0
#             else:
#                 repairMethods[i][4]=repairMethods[i][4]+1   
       
       
    
#         print("entrain de commencer l'entrainement")
#         X= X_balanced.iloc[:,solution]  
#         X= X_balanced.drop('Attack_label', axis=1)
#         print("longueyr",len(X_balanced.columns))




#         X_train, X_test, y_train, y_test = train_test_split(X, y_balanced, test_size=0.25, random_state=42)
#         X_train2=pd.DataFrame(X_train)
#         y_train2=pd.DataFrame(y_train)
#         X_test2=pd.DataFrame(X_test)
#         y_test2=pd.DataFrame(y_test)
#         X_train2.to_csv('./datasets/X_train.csv')
#         X_test2.to_csv('./datasets/X_test.csv')
#         y_train2.to_csv('./datasets/y_train.csv')
#         y_test2.to_csv('./datasets/y_test.csv')

       

#         import numpy as np

#         # assuming X_train and X_test are DataFrames
#         X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
#         X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
#         input_shape = X_train.shape[1:]
#         print(X_train.shape)
#         print(X_test.shape)
#         num_classes = len(np.unique(y_train))
#         num_classes

#         # from  tensorflow.keras.utils import to_categorical 

#         # y_train = to_categorical(y_train, num_classes=num_classes)
#         # y_test = to_categorical(y_test, num_classes=num_classes)
#         print(y_train.shape, y_test.shape)




#         X_train = np.asarray(X_train).astype('float32')
#         y_train = np.asarray(y_train).astype('float32')
#         X_test = np.asarray(X_test).astype('float32')
#         y_test = np.asarray(y_test).astype('float32')

#         model = Sequential() # initializing model
#         # input layer and first layer with 50 neurons
#         model.add(Conv1D(16, 3, padding="same",input_shape = input_shape, activation='relu'))
#         model.add(MaxPool1D(pool_size=(2)))  
#         #model.add(Dropout(0.2))
#         model.add(Conv1D(32, 3, padding="same", activation='relu'))
#         model.add(MaxPool1D(pool_size=(2)))
         
#         model.add(Conv1D(64, 3, padding="same", activation='relu'))
#         model.add(MaxPool1D(pool_size=(2)))  
#         model.add(Flatten())
#         #model.add(Dropout(0.2))
#         model.add(Dense(units=50))
#         # output layer with softmax activation
#         model.add(Dense(units=15,activation='softmax')) 
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    
    
#     adjutWeights(100, 10, acceptanceMethods, 1/3)
#     adjutWeights(100, 10, destructionMethods, 1/3)
#     adjutWeights(100, 10, repairMethods, 1/2)
#     nbr_itr=nbr_itr+1
#     print("########################################################it ",nbr_itr)    
  














bestV=[True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, True, False, True, True, False, True, True, True, True, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]

X_balanced = X_balanced.iloc[:,bestV]  
X_balanced = X_balanced.drop('Attack_label', axis=1)





X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.25, random_state=42)
X_train2=pd.DataFrame(X_train)
y_train2=pd.DataFrame(y_train)
X_test2=pd.DataFrame(X_test)
y_test2=pd.DataFrame(y_test)
X_train2.to_csv('./datasets/X_train.csv')
X_test2.to_csv('./datasets/X_test.csv')
y_train2.to_csv('./datasets/y_train.csv')
y_test2.to_csv('./datasets/y_test.csv')



import numpy as np

# assuming X_train and X_test are DataFrames
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
input_shape = X_train.shape[1:]
print(X_train.shape)
print(X_test.shape)
num_classes = len(np.unique(y_train))
num_classes


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
# model.add(Dense(units=15,activation='softmax')) 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
  
   
  
  
filepath = './models/Multiclass/Edge-IIoTset/cnn_multi.json'
weightspath = './weights/Multiclass/Edge-IIoTset/cnn_multi.h5'
json_file = open(filepath, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(weightspath)
print("Loaded model from disk")
  
# defining loss function, optimizer, metrics and then compiling model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# predicting target attribute on testing dataset
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')  



# representation of model layers
plot_model(model, to_file='plots/cnn_multi.png', show_shapes=True,)  
n_classes = 15
n_classes
y_pred = model.predict(X_test)
fpr_cnn = dict()
tpr_cnn = dict()
roc_auc_cnn = dict()
for i in range(n_classes):
    fpr_cnn[i], tpr_cnn[i], _ = roc_curve(y_test2.iloc[:, i], y_pred[:, i])
    roc_auc_cnn[i] = auc(fpr_cnn[i], tpr_cnn[i])
for i in range(n_classes):
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr_cnn[i], tpr_cnn[i], label='Keras (area = {:.3f})'.format(roc_auc_cnn[i]))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig('plots/cnn_multi_roc'+str(i)+'.png')
  plt.show()
  
pred = model.predict(X_test)
for j in range(0,pred.shape[1]):
  for i in range(0,pred.shape[0]):
    pred[i][j] = int(round(pred[i][j]))
pred_df = pd.DataFrame(pred,columns=y_test2.columns)
print("Recall Score - ",recall_score(y_test,pred_df.astype('uint8'),average='micro'))
print("F1 Score - ",f1_score(y_test,pred_df.astype('uint8'),average='micro'))
print("Precision Score - ",precision_score(y_test,pred_df.astype('uint8'),average='micro'))

matrix = confusion_matrix(y_test2.values.argmax(axis=1),pred_df.values.argmax(axis=1))

print(matrix)

classes=['Normal','DDoS_UDP','DDoS_ICMP','SQL_injection','Password',
          'Vulnerability_scanner','DDoS_TCP','DDoS_HTTP',
          'Uploading','Backdoor','Port_Scanning','XSS',
          'Ransomware','MITM','Fingerprinting']
accuracyCalcul(matrix, classes)




















