import matplotlib.pyplot as plt;
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score

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

#---------------------------------------------OurModel----------------------------------------
clfs ={}

#clf1=GaussianNB()
#clf2=KNeighborsClassifier(n_jobs=-1)
clf3=RandomForestClassifier ( n_estimators = 20 , random_state = 0 ,n_jobs=-1)
#clf4=SVC(kernel='rbf', probability=True)
clf5=tree.DecisionTreeClassifier()
clf7=AdaBoostClassifier()
#clf8=QuadraticDiscriminantAnalysis()

clfs1=[('AdaBoost', clf7),('j48', clf5), ('RFC', clf3)]
ensemble_models = VotingClassifier(estimators=clfs1, voting='soft')

threshold=0.3
leftNumber=-1
temp_left_Group=np.arange(x_train.shape[0])
for i in range(0,x_train.shape[0]):
    temp_left=sum(labelled==-1)
    #print(i, temp_left)
    if temp_left<=0:
        break
    #ensemble_models = VotingClassifier(estimators=clfs1, voting='soft')
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
    #print(temp_left, threshold, (final_decition != -1).mean() *400000 , (predicted != -1).mean() *400000)
    if i >0:
        change_rate=temp_left_Group[i-1] - temp_left_Group[i]
        #print(change_rate)
        if(change_rate<1000):
            threshold=threshold-threshold_step



print("------------------------------------Our Module---------------------------------------")
print('threshold step=',threshold_step)

print('\n*********Training*********\n')

pre=labelled[labelled!=-1]
org=y_train[labelled!=-1]
pop=metrics.precision_recall_fscore_support(org, pre)
results = pd.DataFrame()
for i in range(0,len(pop)):
    if len(pop[0]) > np.unique(data.iloc[:,5]).shape[0]:
        temp = pd.DataFrame([pop[i][:-1]],columns=label_names)
    else :
        temp = pd.DataFrame([pop[i]],columns=label_names)
    
    results=results.append(temp, ignore_index=True)
#print(len(clfs1),'Classifier')


results.insert(0,'  ',['Precision','Recall','Fscore','Support'])
print(results)
resultsDataFrame1 = results
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
