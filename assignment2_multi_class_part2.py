#!/usr/bin/env python
# coding: utf-8

# # Name:Saad,Mohammad
# 
# # Student no :300267006  
# 
# # Assignment(2)-Part(2)
# 
# # ELG7186 – AI for Cybersecurity Applications

# Some important library imports

# In[1]:


import numpy as np 
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay
from io import BytesIO,StringIO
import ast
import pickle


# Reading the dataset for training the static classifier :<br>

# In[2]:


dataset=pd.read_csv('iot_static_data.csv')
y=np.array(dataset.pop('Class'))
X=dataset.copy()


# The data in the column 'Source' are nominal , so we are applying label encoding.

# In[3]:


label_encoder = LabelEncoder()
X['Source']= label_encoder.fit_transform(X['Source'])


# splitting the given dataset for training and testing :

# In[4]:


X1=np.nan_to_num(X,nan=0,posinf=0,neginf=0)
X_train, X_test, y_train, y_test = train_test_split(X1,y,stratify=y, test_size=0.33, random_state=42)


# Training the static classfier and testing using the test split: 

# In[5]:


clf=RandomForestClassifier(random_state=0).fit(X_train,y_train)
y_prdict=clf.predict(X_test)


# In[6]:


print(classification_report(y_test, y_prdict))
cm_test=confusion_matrix(y_test, y_prdict)
ConfusionMatrixDisplay(confusion_matrix=cm_test).plot()


# To save the model using pickle library:<br>

# In[7]:


filename = 'static_model_multi.sav'
pickle.dump(clf, open(filename, 'wb'))  
# load the model from disk
clf = pickle.load(open(filename, 'rb'))


# Using the kafka server data stream 1000 packet were read and then concatenated in a pd dataframe to be used in testing the two models the static and dynamic.After testing ,the 1000 exampel will be added to training data and another 1000 from the old data will be removed in order to be adaptive for any change:  

# In[8]:


import time

# Import the python Consumer Client for Kafka
from kafka import KafkaConsumer

# instantiate the KafkaConsumer Class using the arguments mentioned.
# do not change any arguments other than the first positional argument.
#to load the static model
filename = 'static_model_multi.sav'
clf_static = pickle.load(open(filename, 'rb'))


consumer = KafkaConsumer(
    'task2',   # change this to "task2" for the IOT Botnet Detection  ---- important ----
    bootstrap_servers="34.130.121.39:9092",
    sasl_plain_username="student",
    sasl_plain_password="uottawa",
    security_protocol="SASL_PLAINTEXT",
    sasl_mechanism="PLAIN",
    auto_offset_reset='earliest',
    enable_auto_commit=False
)

# Data Stream flowing in.
X_sliding=X.copy()
y_sliding=np.copy(y)
accuracy_static=[]
accuracy_dynamic=[]
f1_static=[]
f1_dynamic=[]
for k in range(100):
    i = 0
    line=[]
    for message in consumer:
        try:
                print(f"Consuming the {i+1}th data packet!and {k+1} round")
                data_packet = message.value
                dict_str = data_packet.decode("UTF-8")
                my_data = ast.literal_eval(dict_str)
                line.append(pd.DataFrame(my_data,index=[(dataset.shape[0]-1000+i)+(k)*1000]))
        except:
            pass
    
        if len(line) ==1000:#to collect 1000 packet
            break
        i += 1
    for j in range(1,len(line)):#to concatenate the packets in one data_frame
        line[0]=pd.concat([line[0],line[j]])
    data=line[0].copy()
    y_dynamic=np.array(data.pop('Class'))#exrtacting the labels
    X_dynamic=data.copy()
    X_dynamic['Source']= label_encoder.fit_transform(X_dynamic['Source'])
    #matching columns names for correct concatenation
    X_sliding.columns=list(X_dynamic.columns.values)
    #combining the two data frames [moving the sliding window by 1000 exampel]
    frames = [X_sliding.iloc[1000:,:],X_dynamic]
    result = pd.concat(frames)
    #to remove nan elements
    X_test=np.nan_to_num(X_dynamic,nan=0,posinf=0,neginf=0)
 
    #testing the static model only on the collected 1000 packet
    y_predict_static=clf_static.predict(X_test)
    print(classification_report(y_dynamic, y_predict_static))
    #appending the accuracy over iterations
    accuracy_static.append(accuracy_score(y_dynamic, y_predict_static))
    f1_static.append(f1_score(y_dynamic, y_predict_static,average='macro'))

    #########dynamic_classifier
    if k ==0 :#at the beginning we have only one model, so results are the same for static and dynamic
        print(classification_report(y_dynamic, y_predict_static))
        accuracy_dynamic.append(accuracy_score(y_dynamic, y_predict_static))
        f1_dynamic.append(f1_score(y_dynamic, y_predict_static,average='macro'))
    else:#testing the dynamic model
        y_predict_dynamic=clf_dynamic.predict(X_test)
        print(classification_report(y_dynamic, y_predict_dynamic))
        accuracy_dynamic.append(accuracy_score(y_dynamic, y_predict_dynamic))
        f1_dynamic.append(f1_score(y_dynamic, y_predict_dynamic,average='macro'))
    y_sliding=np.hstack((y_sliding[1000:],y_dynamic))
    X_sliding=result.copy()
    X_new=np.nan_to_num(X_sliding,nan=0,posinf=0,neginf=0)
    #trainging with fixed time sliding window
    clf_dynamic=RandomForestClassifier(random_state=0).fit(X_new,y_sliding)
    
    
    
    


# Plotting  accuracies through iterations and computing the average :

# In[17]:


plt.plot(range(k+1),accuracy_static,accuracy_dynamic)
plt.legend(['Accuracy_static','Accuracy_dynamic'])
plt.savefig('acc_multi')
static_accuray_avg=np.sum(np.array(accuracy_static))/100
print(static_accuray_avg)
dynamic_accuracy_avg=np.sum(np.array(accuracy_dynamic))/100
print(dynamic_accuracy_avg)


# Plotting  f1-scores through iterations and computing the average :

# In[18]:


plt.plot(range(k+1),f1_static,f1_dynamic)
plt.legend(['F1_static','F1_dynamic'])
plt.savefig('f1_multi')
static_f1_avg=np.sum(np.array(f1_static))/100
print(static_f1_avg)
dynamic_f1_avg=np.sum(np.array(f1_dynamic))/100
print(dynamic_f1_avg)

