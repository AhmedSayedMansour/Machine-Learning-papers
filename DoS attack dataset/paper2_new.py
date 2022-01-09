import matplotlib.pyplot as plt;
import pandas as pd
import numpy as np
from sklearn import preprocessing
from kneed import DataGenerator, KneeLocator
from sklearn.metrics import accuracy_score
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import BernoulliNB

trainSize = 400000
testSize = 50000
size = trainSize+testSize # 400000 for taining and 50000 for testing
data = pd.read_csv("mixed.csv", nrows=size)
data = data.drop(columns="No.")
columns_titles = ["Time", "Source", "Destination", "Length", "Info", "Protocol"]
data = data.reindex(columns=columns_titles)
label_names = np.unique(data.iloc[:,5])

le = preprocessing.LabelEncoder()
data.iloc[:,1] = le.fit_transform(data.iloc[:,1])
data.iloc[:,2] = le.fit_transform(data.iloc[:,2])
data.iloc[:,4] = le.fit_transform(data.iloc[:,4])
data.iloc[:,5] = le.fit_transform(data.iloc[:,5])

def chooseObsMan(dic_f):
    chosenIDs=np.array(())
    for k,d in dic_f.items():
        temp=np.where(y_train==k)[0]
        if temp.size < dic_f[k]:
            temp = dic_f[k]
        z=np.random.choice(temp,dic_f[k],replace=False)
        chosenIDs=np.concatenate((chosenIDs,z))
    return np.int64(chosenIDs)
def label_index(row,threshold=0):
    
    if threshold==0:
        threshold=np.max(row)
    temp_i=np.where(row>=threshold)[0]
    if temp_i.shape[0]>0:
        return temp_i[0]+1
    else:
        return -1

y_train = data.iloc[:trainSize,5]
x_train = data.iloc[:trainSize,:-1]
#x_train = preprocessing.normalize(data.iloc[:trainSize,:-1])

y_test = data.iloc[trainSize:,5]
x_test = data.iloc[trainSize:,:-1]
#x_test = preprocessing.normalize(data.iloc[trainSize:,:-1])

li = {}
r = int(size/100000 + 1)*30
for i in np.unique(data.iloc[:,-1]):
    if data['Protocol'].value_counts()[i] < r:
        li.update({i : data['Protocol'].value_counts()[i]})    
    else :
        li.update({i : r})

labelled=np.zeros(x_train.shape[0])
labelled[:]=-1
labelledIndex=chooseObsMan(li)
labelled[labelledIndex]=y_train[labelledIndex]

clf = BernoulliNB()
clf.fit(x_train, y_train)

predicted = clf.predict(x_train)

print((predicted == y_train).mean())

print("------------------------------------Paper 2---------------------------------------")

print('\n*********Training*********\n')

matrices = metrics.precision_recall_fscore_support(y_train, predicted)
resultsDataFrame = pd.DataFrame()
for i in range(0,len(matrices)):
    temp = pd.DataFrame([matrices[i]],columns=label_names)
    resultsDataFrame = resultsDataFrame.append(temp, ignore_index=True)
resultsDataFrame.insert(0,'  ',['Precision','Recall','Fscore','Support'])
print(resultsDataFrame)

correctLabelled=metrics.accuracy_score(y_train, predicted,normalize=False)
incorrectLabelled=sum(resultsDataFrame.iloc[3,1:]) - correctLabelled
print([correctLabelled,incorrectLabelled])

numbers=resultsDataFrame.iloc[3,1:]
correctLabelledForEachclass=resultsDataFrame.iloc[3,1:] *resultsDataFrame.iloc[1,1:]
incorrectLabelledForEachclass=resultsDataFrame.iloc[3,1:]-correctLabelledForEachclass
results1 = pd.DataFrame(columns = ['Total','Correct', 'Incorrect'])
results1['Total'] = numbers
results1['Correct'] = correctLabelledForEachclass
results1['Incorrect'] = incorrectLabelledForEachclass
print(results1)
