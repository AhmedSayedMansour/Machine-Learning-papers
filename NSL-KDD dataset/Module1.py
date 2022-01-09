# -*- coding: utf-8 -*-

"""
Created on Fri Sep 11 19:59:39 2020

@author: bsom2
"""
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt;
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans 
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
import math
from collections import Counter
from sklearn.model_selection import GridSearchCV



def chooseObsMan(dic_f):
    chosenIDs=np.array(())
    for k,d in dic_f.items():
        temp=np.where(y_train==k)[0]
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

path=r''
traData = pd.read_csv(path+'trainingGroup.csv',index_col=None, header=None)
#traData = pd.read_csv('D:\\drive\\publications_2017\\datasets\\NSL-KDD\\NSL-KDD\\testingData.csv'  , index_col = None , header = None)
tesData = pd.read_csv(path+'testingGroup.csv',index_col=None, header=None)
target = 41




y_train = traData[[target]].values.reshape(len(traData)) 
x_train=preprocessing.normalize(traData.iloc[:,:-1])

y_test = tesData[[target]].values.reshape(len(tesData)) 
x_test=preprocessing.normalize(tesData.iloc[:,:-1])


labelled=np.zeros(x_train.shape[0])
labelled[:]=-1
li={1:60,2:60,3:30,4:10,5:10}
labelledIndex=chooseObsMan(li)
labelled[labelledIndex]=y_train[labelledIndex]


clfs ={}
clf1=GaussianNB()
clf2=KNeighborsClassifier(n_jobs=-1)
clf3=RandomForestClassifier ( n_estimators = 20 , random_state = 0 ,n_jobs=-1)
clf4=SVC(kernel='rbf', probability=True)
clf5=tree.DecisionTreeClassifier()
clf7=AdaBoostClassifier()
clf8=QuadraticDiscriminantAnalysis()

#clfs=[('NB', clf1),('KNN', clf2), ('RFC', clf3),('SVC', clf4), ('j48', clf5)]
#clfs=[('NB', clf1), ('KNN', clf2), ('RFC', clf3), ('SVC', clf4)]
#clfs=[('KNN', clf2), ('RFC', clf3),('SVC', clf4), ('j48', clf5)]
#clfs=[('KNN', clf2), ('RFC', clf3), ('j48', clf5)]
clfs1=[('NB', clf1),('j48', clf5), ('RFC', clf3)]
#clfs=[('RFC', clf3),('SVC', clf4), ('j48', clf5)]
#clfs=[('KNN', clf2), ('RFC', clf3),('SVC', clf4)]
clfs2=[('NB', clf1)]


#Model1
threshold=0.99
leftNumber=-1
temp_left_Group=np.arange(x_train.shape[0])
for i in range(0,x_train.shape[0]):
    temp_left=sum(labelled==-1)
    if temp_left<=0:
        break;
    ensemble_models = VotingClassifier(estimators=clfs2, voting='soft')
    ensemble_models.fit(x_train[labelled!=-1],y_train[labelled!=-1])
    predicted=ensemble_models.predict_proba(x_train[labelled==-1])
    final_decition=np.apply_along_axis(label_index, 1, predicted,threshold)
    labelled[labelled==-1]=final_decition
    if temp_left==leftNumber:
        threshold-0.01
    threshold_step=0.01
    leftNumber=temp_left
    temp_left=sum(labelled==-1)
    temp_left_Group[i]=temp_left
    print([temp_left,threshold])
    if i >0:
        change_rate=temp_left_Group[i-1] - temp_left_Group[i]
        if(change_rate<1000):
            threshold=threshold-threshold_step

print(i)

        
pre=labelled[labelled!=-1]
org=y_train[labelled!=-1]
pop=metrics.precision_recall_fscore_support(org, pre,labels=[1, 2, 3,4, 5])
results = pd.DataFrame();
for i in range(0,len(pop)):
    temp = pd.DataFrame([pop[i]],columns=['normal','DoS','Probe','R2L','U2R'])
    results=results.append(temp, ignore_index=True)
print(len(clfs1),'Classifier')
print('threshold step=',threshold_step)
results.insert(0,'  ',['Precision','Recall','Fscore','Support'])
print(results)
correctLabelled=metrics.accuracy_score(org, pre,normalize=False)
incorrectLabelled=sum(results.iloc[3,1:])-correctLabelled
print([correctLabelled,incorrectLabelled])
numbers=results.iloc[3,1:]
correctLabelledForEachclass=results.iloc[3,1:] *results.iloc[1,1:]
incorrectLabelledForEachclass=results.iloc[3,1:]-correctLabelledForEachclass
results1 = pd.DataFrame(columns = ['Total','Correct', 'Incorrect'])
results1['Total'] = numbers
results1['Correct'] = correctLabelledForEachclass
results1['Incorrect'] = incorrectLabelledForEachclass
print(results1)

CorrectNormal=int(results1.iloc[0,1])
CorrectDos=int(results1.iloc[1,1])
CorrectProbe=int(results1.iloc[2,1])
CorrectR2L=int(results1.iloc[3,1])
CorrectU2R=int(results1.iloc[4,1])

IncorrectNormal=int(results1.iloc[0,2])
IncorrectDos=int(results1.iloc[1,2])
IncorrectProbe=int(results1.iloc[2,2])
IncorrectR2L=int(results1.iloc[3,2])
IncorrectU2R=int(results1.iloc[4,2])

# importing the required module 

  
# x axis values 
x =range(0,50) 
# corresponding y axis values 
y = temp_left_Group[0:50]
#range(temp_left_Group[0],temp_left_Group[30])
  
# plotting the points  
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('Iterations') 
# naming the y axis 
plt.ylabel('No.of unlabeled observations') 
  
# giving a title to my graph 
plt.title(r'(a) Without updating $\alpha$ threshold') 
  
# function to show the plot 
plt.show()


       
# pre=labelled[labelled!=-1]
# org=y_train[labelled!=-1]
# pop=metrics.precision_recall_fscore_support(org, pre,labels=[1, 2, 3,4, 5])
# results2 = pd.DataFrame();
# for i in range(0,len(pop)):
#     temp = pd.DataFrame([pop[i]],columns=['normal','DoS','Probe','R2L','U2R'])
#     results2=results2.append(temp, ignore_index=True)
# results2.insert(0,'  ',['Precision','Recall','Fscore','Support'])
# print(results2)
# correctLabelled=metrics.accuracy_score(org, pre,normalize=False)
# incorrectLabelled=sum(results2.iloc[3,1:])-correctLabelled
# print([correctLabelled,incorrectLabelled])
# numbers=results2.iloc[3,1:]
# correctLabelledForEachclass=results2.iloc[3,1:] *results2.iloc[1,1:]
# incorrectLabelledForEachclass=results2.iloc[3,1:]-correctLabelledForEachclass
# results3 = pd.DataFrame(columns = ['Total','Correct', 'Incorrect'])
# results3['Total'] = numbers
# results3['Correct'] = correctLabelledForEachclass
# results3['Incorrect'] = incorrectLabelledForEachclass
# print(results3)

# y_data=all_data.iloc[:,-1]
# x_axis=np.arange(start=1,stop=y_data.shape[0]+1,step=1)
# plt.scatter(x_axis[y_data==0],all_scores[y_data==0],s=2,c='b',lw=2,marker='s',label='normal');
# plt.scatter(x_axis[y_data==1],all_scores[y_data==1],s=2,c='r',lw=2,marker='o',label='abnormal');
# plt.show()
# print('iteration',TH3+1,'is done')
# TH3+=1


# left = [1, 2, 3, 4, 5]
# classes=['normal','DoS','Probe','R2L','U2R']
# values=results1.iloc[:,0]
# plt.bar(left, values, tick_label = classes, 
#         width = 0.8, color = ['red','blue']) 
# plt.xlabel('classes') 
# plt.ylabel('y - axis')
# plt.show() 



##RFC1 = RandomForestClassifier ( n_estimators = 20 , random_state = 0 );
#RFC1= KNeighborsClassifier()
#RFC1.fit(x_train[labelled!=-1], labelled[labelled!=-1])
#y_predictedPro3 = RFC1.predict(x_test)
#precision=metrics.precision_score(y_test, y_predictedPro3)
#recall=metrics.recall_score(y_test , y_predictedPro3)
#F_measure=metrics.f1_score(y_test, y_predictedPro3)
#acu=metrics.accuracy_score(y_test, y_predictedPro3) 
#
##falsePositive = fp / ( fp + tp )
##print('FP : ' + str(falsePositive) )
#print('precision : ' + str(precision) )
#print('recall : ' + str(recall))
#print('F-measure : ' + str(F_measure))
#print('accuracy_score : ' + str(acu))
#    
#print('--------------- The results of small data------------------')
##RFC2 = RandomForestClassifier ( n_estimators = 20 , random_state = 0 );
#semi_labelled1=np.copy(semi_labelled)
#RFC2= KNeighborsClassifier()
#RFC2.fit(x_train[labelled!=-1], labelled[labelled!=-1])
#newLabels = RFC2.predict(x_train[semi_labelled1==-1])
#semi_labelled1[semi_labelled1==-1]=newLabels
#tn , fp , fn , tp = metrics.confusion_matrix(semi_labelled1, y_train[semi_labelled1!=-1]).ravel()
#print([tn , fp , fn , tp])
#
#
#RFC2.fit(x_train[semi_labelled1!=-1],semi_labelled1)
#y_predictedPro = RFC2.predict(x_test)
#precision=metrics.precision_score(y_test, y_predictedPro)
#recall=metrics.recall_score(y_test , y_predictedPro)
#F_measure=metrics.f1_score(y_test, y_predictedPro)
#acu=metrics.accuracy_score(y_test, y_predictedPro) 
#
##falsePositive = fp / ( fp + tp )
##print('FP : ' + str(falsePositive) )
#print('precision : ' + str(precision) )
#print('recall : ' + str(recall))
#print('F-measure : ' + str(F_measure))
#print('accuracy_score : ' + str(acu))
#print('--------------- The results ------------------')
#
##RFC2 = RandomForestClassifier ( n_estimators = 20 , random_state = 0 );
#RFC2= KNeighborsClassifier()
#RFC2.fit(x_semiData, y_semiData)
#newLabels = RFC2.predict(x_train[semi_labelled==-1])
#semi_labelled[semi_labelled==-1]=newLabels
#tn , fp , fn , tp = metrics.confusion_matrix(semi_labelled, y_train[semi_labelled!=-1]).ravel()
#print([tn , fp , fn , tp])
#
#
#RFC2.fit(x_train[semi_labelled!=-1],semi_labelled)
#y_predictedPro = RFC2.predict(x_test)
#precision=metrics.precision_score(y_test, y_predictedPro)
#recall=metrics.recall_score(y_test , y_predictedPro)
#F_measure=metrics.f1_score(y_test, y_predictedPro)
#acu=metrics.accuracy_score(y_test, y_predictedPro) 
#
##falsePositive = fp / ( fp + tp )
##print('FP : ' + str(falsePositive) )
#print('precision : ' + str(precision) )
#print('recall : ' + str(recall))
#print('F-measure : ' + str(F_measure))
#print('accuracy_score : ' + str(acu))
#print('--------------- The results ------------------')
#    
    
    
    