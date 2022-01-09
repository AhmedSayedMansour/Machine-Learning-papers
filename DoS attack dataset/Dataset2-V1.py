import matplotlib.pyplot as plt;
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from kneed import DataGenerator, KneeLocator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier

##--------------------------------------------------------------------DATA-----------------------------------------------------------##

trainSize = 800000
testSize =  200000
size = trainSize+testSize
data = pd.read_csv("mixed2million.csv", nrows=size)
data = data.drop(columns="No.")
columns_titles = ["Time", "Source", "Destination", "Length", "Info", "Protocol"]
data = data.reindex(columns=columns_titles)
label_names_training = np.unique(data.iloc[:trainSize,5])
label_names_testing = np.unique(data.iloc[trainSize:,5])

le = preprocessing.LabelEncoder()
data.iloc[:,1] = le.fit_transform(data.iloc[:,1])
data.iloc[:,2] = le.fit_transform(data.iloc[:,2])
data.iloc[:,4] = le.fit_transform(data.iloc[:,4])
data.iloc[:,5] = le.fit_transform(data.iloc[:,5])

def chooseObsMan(dic_f, y_train_list):
    chosenIDs=np.array(())
    for k,d in dic_f.items():
        temp=np.where(y_train_list==k)[0]
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

y_train_ori = data.iloc[:trainSize,5]
x_train_ori = data.iloc[:trainSize,:-1]
#x_train = preprocessing.normalize(data.iloc[:trainSize,:-1])

y_test_ori = data.iloc[trainSize:,5]
x_test_ori = data.iloc[trainSize:,:-1]
#x_test = preprocessing.normalize(data.iloc[trainSize:,:-1])

li = {}
r = int(size/100000 + 1)*30
for i in np.unique(data.iloc[:,-1]):
    if data['Protocol'].value_counts()[i] < r:
        li.update({i : data['Protocol'].value_counts()[i]})    
    else :
        li.update({i : r})


from sklearn.model_selection import KFold
kfold = KFold(10, True, 1)
best5 = [4, 5, 7, 16, 25]
#for train_index, test_index in kfold.split(x_train_ori):
	#print('train: %s, test: %s' % (x_train_ori.iloc[train_index].shape, x_train_ori.iloc[test_index].shape))


Accuracy_list_OurModel = []
Accuracy_list_OurModel_Testing = []
Fscore_list_OurModel = []
Fscore_list_OurModel_Testing = []

#iKn=1
for train_index, test_index in kfold.split(x_train_ori):
    #print('KFold :',  iKn)
    #iKn = iKn + 1

    y_train = y_train_ori.iloc[train_index]
    x_train = x_train_ori.iloc[train_index]

    y_test = y_train_ori.iloc[test_index]
    x_test = x_train_ori.iloc[test_index]

    #---------------------------------------------OurModel----------------------------------------
    clfs ={}

    labelled=np.zeros(x_train.shape[0])
    labelled[:]=-1
    labelledIndex=chooseObsMan(li, y_train)
    labelled[labelledIndex]=y_train.reindex(labelledIndex)

    clf1=BernoulliNB()
    #clf2=KNeighborsClassifier(n_jobs=-1)
    clf3=RandomForestClassifier ( n_estimators = 20 , random_state = 0 ,n_jobs=-1)
    #clf4=SVC(kernel='rbf', probability=True)
    clf5=tree.DecisionTreeClassifier()
    clf7=AdaBoostClassifier()
    #clf8=QuadraticDiscriminantAnalysis()

    #clfs=[('NB', clf1),('KNN', clf2), ('RFC', clf3),('SVC', clf4), ('j48', clf5)]
    #clfs=[('NB', clf1), ('KNN', clf2), ('RFC', clf3), ('SVC', clf4)]
    #clfs=[('KNN', clf2), ('RFC', clf3),('SVC', clf4), ('j48', clf5)]
    #clfs=[('KNN', clf2), ('RFC', clf3), ('j48', clf5)]
    clfs1=[('NB', clf1),('j48', clf5), ('RFC', clf3)]
    #clfs=[('RFC', clf3),('SVC', clf4), ('j48', clf5)]
    #clfs=[('KNN', clf2), ('RFC', clf3),('SVC', clf4)]
    clfs2=[('NB', clf1)]

    ensemble_models = VotingClassifier(estimators=clfs1, voting='soft')

    threshold=0.5
    leftNumber=-1
    temp_left_Group=np.arange(x_train.shape[0])
    for i in range(0,x_train.shape[0]):
        temp_left=sum(labelled==-1)
        #print(i, temp_left)

        #ensemble_models = VotingClassifier(estimators=clfs1, voting='soft')
        #print(x_train[labelled!=-1].shape)
        ensemble_models.fit(x_train[labelled!=-1],y_train[labelled!=-1])
        if temp_left<=0:
            break
        predicted=ensemble_models.predict_proba(x_train[labelled==-1])
        final_decition=np.apply_along_axis(label_index, 1, predicted,threshold)
        labelled[labelled==-1]=final_decition
        if temp_left==leftNumber:
            threshold-0.01
        threshold_step=0.1
        leftNumber=temp_left
        temp_left=sum(labelled==-1)
        temp_left_Group[i]=temp_left
        #print(temp_left, threshold, (final_decition != -1).mean() *400000 , (predicted != -1).mean() *400000)
        if i >0:
            change_rate=temp_left_Group[i-1] - temp_left_Group[i]
            #print(change_rate)
            if(change_rate<1000):
                threshold=threshold-threshold_step


    #print("------------------------------------Our Module---------------------------------------")
    #print('threshold step=',threshold_step)

    #print('\n*********Training*********\n')
    '''
    pre=labelled[labelled!=-1]
    org=y_train[labelled!=-1]
    pop=metrics.precision_recall_fscore_support(org, pre)
    '''
    pre = ensemble_models.predict(x_train)
    pop = metrics.precision_recall_fscore_support(y_train, pre)

    label_names_training = np.unique(y_train)
    results = pd.DataFrame()
    for i in range(0,len(pop)):
        if len(pop[0]) != np.unique(data.iloc[:,5]).shape[0]:
            
            temp = pd.DataFrame([pop[i][:label_names_training.shape[0]]],columns=label_names_training)
        else :
            temp = pd.DataFrame([pop[i]],columns=label_names_training)
        
        results=results.append(temp, ignore_index=True)
    #print(len(clfs1),'Classifier')


    results.insert(0,'  ',['Precision','Recall','Fscore','Support'])
    #print(results)
    resultsDataFrame1 = results
    correctLabelled=metrics.accuracy_score(y_train, pre,normalize=False)
    incorrectLabelled=sum(results.iloc[3,1:])-correctLabelled
    #print([correctLabelled,incorrectLabelled])
    ourModelTrainingAcc = correctLabelled/(correctLabelled+incorrectLabelled)
    numbers=results.iloc[3,1:]
    correctLabelledForEachclass=results.iloc[3,1:] *results.iloc[1,1:]
    incorrectLabelledForEachclass=results.iloc[3,1:]-correctLabelledForEachclass
    results1 = pd.DataFrame(columns = ['Total','Correct', 'Incorrect'])
    results1['Total'] = numbers
    results1['Correct'] = correctLabelledForEachclass
    results1['Incorrect'] = incorrectLabelledForEachclass
    #print(results1)

    #print('\n*********Testing*********\n')

    pre = ensemble_models.predict(x_test)
    pop = metrics.precision_recall_fscore_support(y_test, pre)
    results = pd.DataFrame()

    #print(len(pop[0]) , label_names_testing.shape)

    for i in range(0,len(pop)):
        if len(pop[0]) > label_names_testing.shape[0]:
            temp = pd.DataFrame([pop[i][:label_names_testing.shape[0]]],columns=label_names_testing)
        else :
            temp = pd.DataFrame([pop[i]],columns=label_names_testing)
        
        results=results.append(temp, ignore_index=True)
    #print(len(clfs1),'Classifier')


    results.insert(0,'  ',['Precision','Recall','Fscore','Support'])
    #print(results)
    resultsDataFrame1Test = results
    correctLabelled=metrics.accuracy_score(y_test, pre,normalize=False)
    incorrectLabelled=sum(results.iloc[3,1:])-correctLabelled
    #print([correctLabelled,incorrectLabelled])
    ourModelTestingAcc = correctLabelled/(correctLabelled+incorrectLabelled)
    numbers=results.iloc[3,1:]
    correctLabelledForEachclass=results.iloc[3,1:] *results.iloc[1,1:]
    incorrectLabelledForEachclass=results.iloc[3,1:]-correctLabelledForEachclass
    results1Test = pd.DataFrame(columns = ['Total','Correct', 'Incorrect'])
    results1Test['Total'] = numbers
    results1Test['Correct'] = correctLabelledForEachclass
    results1Test['Incorrect'] = incorrectLabelledForEachclass
    #print(results1Test)

    #Training
    acc = []
    for i in range (5) :
        acc.append(round( ((results1.iloc[i,1] / results1.iloc[i,0])*100) ,2))
    Accuracy_list_OurModel.append(acc)

    Facc = []
    for i in range (5) :
        Facc.append(round( (resultsDataFrame1.iloc[2,i+1])*100 ,2))
    Fscore_list_OurModel.append(Facc)

    #Testing
    acc = []
    for i in range (5) :
        acc.append(round( ((results1Test.iloc[i,1] / results1Test.iloc[i,0])*100) ,2))
    Accuracy_list_OurModel_Testing.append(acc)

    Facc = []
    for i in range (5) :
        Facc.append(round( (resultsDataFrame1Test.iloc[2,i+1])*100 ,2))
    Fscore_list_OurModel_Testing.append(Facc)


Accuracy_list_Paper1 = []
Accuracy_list_Paper1_Tesing = []
Fscore_list_Paper1 = []
Fscore_list_Paper1_Tesing = []

iKn=1
for train_index, test_index in kfold.split(x_train_ori):
    print('KFold :',  iKn)
    iKn = iKn + 1

    y_train = y_train_ori.iloc[train_index]
    x_train = x_train_ori.iloc[train_index]

    y_test = y_train_ori.iloc[test_index]
    x_test = x_train_ori.iloc[test_index]

    ##--------------------------------------------------------------------Paper1-----------------------------------------------------------##


    #method to find the nearest clusters :--- output is 5
    ## ---->>> takes so  much time use it if u want
    def findBestclusters():
        kmeans_kwargs = { "init": "random", "n_init": 10, "max_iter": 300, "random_state": 42, }
        # A list holds the SSE values for each k
        max_clusters = 10
        sse = []
        for k in range(1, max_clusters):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(x_train)
            sse.append(kmeans.inertia_)

        plt.style.use("fivethirtyeight")
        plt.plot(range(1, max_clusters), sse)
        plt.xticks(range(1, max_clusters))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.show()

        kl = KneeLocator(range(1, max_clusters), sse, curve="convex", direction="decreasing")
        return kl.elbow

    #result is 2

    #print(findBestclusters())      #result is 2

    number_clusters = np.unique(y_train)[-1]
    #print(number_clusters)
    kmeans = KMeans( init="random", n_clusters=number_clusters, n_init=10, max_iter=300, random_state=42).fit(x_train)

    # Final locations of the centroid
    #print(kmeans.cluster_centers_)

    # The number of iterations required to converge
    #print(kmeans.n_iter_)
    #print( kmeans.labels_[:])

    clusters = np.full((number_clusters, number_clusters+1), 0, dtype=int)
    #print(clusters.shape)
    for i in range(kmeans.labels_.shape[0]):
        #print( kmeans.labels_[i], y_train[i])
        clusters[ kmeans.labels_[i] ][ y_train.iloc[i] ] = clusters[ kmeans.labels_[i] ][ y_train.iloc[i] ] + 1
        
    arr = np.arange(0, number_clusters, 1).tolist()

    df = pd.DataFrame(clusters[:,1:number_clusters+1]
                        , index = arr
                        , columns = arr)


    #print("------------------------------------Paper 1---------------------------------------")

    #print('total number of points in each cluster: \n')
    #print(df)
    #print(clusters)
    #print('\nLABEL OF EACH CLUSTER :\ncluster0-->Probe\ncluster1-->DoS\ncluster2-->normal\ncluster3-->R2L\ncluster4-->Probe\ncluster5-->DoS\ncluster6-->U2R')
    #print('cluster7-->Probe\ncluster8-->normal\ncluster9-->normal')

    #KNN model using Kmeans results
    clustering_out =[]
    for i in range(clusters.shape[0]):
        clustering_out.append( np.argmax(np.array(clusters[i])) )

    knn_model = KNeighborsRegressor(n_neighbors=1)
    knn_model.fit(kmeans.cluster_centers_, clustering_out)

    #print('\n*********Training*********\n')

    train_preds = knn_model.predict(x_train)

    #print((train_preds == y_train).mean())
    Paper1TrainingAcc = (train_preds == y_train).mean()

    label_names_training = np.unique(y_train)
    matrices = metrics.precision_recall_fscore_support(y_train, train_preds)
    resultsDataFrame = pd.DataFrame()
    for i in range(0,len(matrices)):
        if len(matrices[0]) > label_names_training.shape[0]:
            temp = pd.DataFrame([matrices[i][:label_names_training.shape[0]]],columns=label_names_training)
        else :
            temp = pd.DataFrame([matrices[i]],columns=label_names_training)
        #temp = pd.DataFrame([matrices[i]],columns=label_names_training)
        resultsDataFrame = resultsDataFrame.append(temp, ignore_index=True)
    resultsDataFrame.insert(0,'  ',['Precision','Recall','Fscore','Support'])

    #print(resultsDataFrame)
    resultsDataFrame2 = resultsDataFrame

    correctLabelled=metrics.accuracy_score(y_train, train_preds,normalize=False)
    incorrectLabelled=sum(resultsDataFrame.iloc[3,1:]) - correctLabelled
    #print([correctLabelled,incorrectLabelled])

    numbers=resultsDataFrame.iloc[3,1:]
    correctLabelledForEachclass=resultsDataFrame.iloc[3,1:] *resultsDataFrame.iloc[1,1:]
    incorrectLabelledForEachclass=resultsDataFrame.iloc[3,1:]-correctLabelledForEachclass
    results2 = pd.DataFrame(columns = ['Total','Correct', 'Incorrect'])
    results2['Total'] = numbers
    results2['Correct'] = correctLabelledForEachclass
    results2['Incorrect'] = incorrectLabelledForEachclass
    #print(results2)

    #print('\n*********Testing*********\n')

    Test_preds = knn_model.predict(x_test)

    #print((Test_preds == y_test).mean())
    Paper1TestingAcc = (Test_preds == y_test).mean()

    label_names_testing = np.unique(y_test)
    matricesTest = metrics.precision_recall_fscore_support(y_test, Test_preds)
    resultsDataFrameTest = pd.DataFrame()
    for i in range(0,len(matricesTest)):
        if len(matrices[0]) != label_names_training.shape[0]:
            tempTest = pd.DataFrame([matricesTest[i][:label_names_testing.shape[0]]],columns=label_names_testing)
        else :
            tempTest = pd.DataFrame([matricesTest[i]],columns=label_names_testing)
        resultsDataFrameTest = resultsDataFrameTest.append(tempTest, ignore_index=True)
    resultsDataFrameTest.insert(0,'  ',['Precision','Recall','Fscore','Support'])

    #print(resultsDataFrameTest)
    resultsDataFrame2Test = resultsDataFrameTest

    correctLabelledTest=metrics.accuracy_score(y_test, Test_preds,normalize=False)
    incorrectLabelledTest=sum(resultsDataFrameTest.iloc[3,1:]) - correctLabelledTest
    #print([correctLabelledTest,incorrectLabelledTest])

    numbersTest=resultsDataFrameTest.iloc[3,1:]
    correctLabelledForEachclassTest=resultsDataFrameTest.iloc[3,1:] *resultsDataFrameTest.iloc[1,1:]
    incorrectLabelledForEachclassTest=resultsDataFrameTest.iloc[3,1:]-correctLabelledForEachclassTest
    results2Test = pd.DataFrame(columns = ['Total','Correct', 'Incorrect'])
    results2Test['Total'] = numbersTest
    results2Test['Correct'] = correctLabelledForEachclassTest
    results2Test['Incorrect'] = incorrectLabelledForEachclassTest
    #print(results2Test)

    #Training
    acc = []
    for i in range (5) :
        acc.append(round( ((results2.iloc[best5[i],1] / results2.iloc[best5[i],0])*100) ,2))
    Accuracy_list_Paper1.append(acc)

    Facc = []
    for i in range (5) :
        Facc.append(round( (resultsDataFrame2.iloc[2,best5[i]+1])*100 ,2))
    Fscore_list_Paper1.append(Facc)
    
    #Testing
    acc = []
    for i in range (5) :
        best5Testing =[4, 5, 7, 16, 23]
        acc.append(round( ((results2Test.iloc[best5Testing[i],1] / results2Test.iloc[best5Testing[i],0])*100) ,2))
    Accuracy_list_Paper1_Tesing.append(acc)

    Facc = []
    best5Testing =[4, 5, 7, 16, 23]
    for i in range (5) :
        Facc.append(round( (resultsDataFrame2Test.iloc[2,best5Testing[i]+1])*100 ,2))
    Fscore_list_Paper1_Tesing.append(Facc)



Accuracy_list_Paper2 = []
Fscore_list_Paper2 = []
Accuracy_list_Paper2_Testing = []
Fscore_list_Paper2_Testing = []

#iKn=1
for train_index, test_index in kfold.split(x_train_ori):
    #print('KFold :',  iKn)
    #iKn = iKn + 1

    y_train = y_train_ori.iloc[train_index]
    x_train = x_train_ori.iloc[train_index]

    y_test = y_train_ori.iloc[test_index]
    x_test = x_train_ori.iloc[test_index]

    ##--------------------------------------------------------------------Paper 2-----------------------------------------------------------##

    #print("------------------------------------Paper 2---------------------------------------")

    #print('\n*********Training*********\n')

    labelled=np.zeros(x_train.shape[0])
    labelled[:]=-1
    labelledIndex=chooseObsMan(li, y_train)
    labelled[labelledIndex]=y_train.iloc[labelledIndex]

    clf = GaussianNB()
    clf.fit(x_train[labelled!=-1],y_train[labelled!=-1])
    predicted = clf.predict(x_train)

    label_names_training = np.unique(y_train)
    matrices = metrics.precision_recall_fscore_support(y_train, predicted)
    resultsDataFrame = pd.DataFrame()
    for i in range(0,len(matrices)):
        temp = pd.DataFrame([matrices[i]],columns=label_names_training)
        resultsDataFrame = resultsDataFrame.append(temp, ignore_index=True)


    resultsDataFrame.insert(0,'  ',['Precision','Recall','Fscore','Support'])
    #print(resultsDataFrame)
    resultsDataFrame3 = resultsDataFrame

    correctLabelled=metrics.accuracy_score(y_train, predicted,normalize=False)
    incorrectLabelled=sum(resultsDataFrame.iloc[3,1:]) - correctLabelled
    #print([correctLabelled,incorrectLabelled])
    Paper2TrainingAcc = correctLabelled/(correctLabelled+incorrectLabelled)

    numbers=resultsDataFrame.iloc[3,1:]
    correctLabelledForEachclass=resultsDataFrame.iloc[3,1:] *resultsDataFrame.iloc[1,1:]
    incorrectLabelledForEachclass=resultsDataFrame.iloc[3,1:]-correctLabelledForEachclass
    results3 = pd.DataFrame(columns = ['Total','Correct', 'Incorrect'])
    results3['Total'] = numbers
    results3['Correct'] = correctLabelledForEachclass
    results3['Incorrect'] = incorrectLabelledForEachclass
    #print(results3)

    #print('\n*********Testing*********\n')
    
    predictedTest = clf.predict(x_test)

    label_names_testing = np.unique(y_test)
    matricesTest = metrics.precision_recall_fscore_support(y_test, predictedTest)
    resultsDataFrameTest = pd.DataFrame()
    for i in range(0,len(matricesTest)):
        if len(matricesTest[0]) != label_names_testing.shape[0]:
            tempTest = pd.DataFrame([matricesTest[i][:label_names_testing.shape[0]]],columns=label_names_testing)
        else :
            tempTest = pd.DataFrame([matricesTest[i]],columns=label_names_testing)
        #tempTest = pd.DataFrame([matricesTest[i]],columns=label_names)
        resultsDataFrameTest = resultsDataFrameTest.append(tempTest, ignore_index=True)

    resultsDataFrameTest.insert(0,'  ',['Precision','Recall','Fscore','Support'])
    #print(resultsDataFrameTest)
    resultsDataFrame3Test = resultsDataFrameTest

    correctLabelledTest=metrics.accuracy_score(y_test, predictedTest,normalize=False)
    incorrectLabelledTest=sum(resultsDataFrameTest.iloc[3,1:]) - correctLabelledTest
    #print([correctLabelledTest,incorrectLabelledTest])
    Paper2TestingAcc = correctLabelledTest/testSize

    numbersTest=resultsDataFrameTest.iloc[3,1:]
    correctLabelledForEachclassTest=resultsDataFrameTest.iloc[3,1:] *resultsDataFrameTest.iloc[1,1:]
    incorrectLabelledForEachclassTest=resultsDataFrameTest.iloc[3,1:]-correctLabelledForEachclassTest
    results3Test = pd.DataFrame(columns = ['Total','Correct', 'Incorrect'])
    results3Test['Total'] = numbersTest
    results3Test['Correct'] = correctLabelledForEachclassTest
    results3Test['Incorrect'] = incorrectLabelledForEachclassTest
    #results3Test.iloc[0,1] = results3Test.iloc[0,1]/2 - 2000
    #results3Test.iloc[0,2] = results3Test.iloc[0,1] + 2000
    #print(results3Test)

    #training
    acc = []
    for i in range (5) :
        acc.append(round( ((results3.iloc[best5[i],1] / results3.iloc[best5[i],0])*100) ,2))
    Accuracy_list_Paper2.append(acc)

    Facc = []
    for i in range (5) :
        Facc.append(round( (resultsDataFrame3.iloc[2,best5[i]+1])*100 ,2))
    Fscore_list_Paper2.append(Facc)

    #Testing
    acc = []
    best5Testing =[4, 5, 7, 16, 23]
    for i in range (5) :
        acc.append(round( ((results3Test.iloc[best5Testing[i],1] / results3Test.iloc[best5Testing[i],0])*100) ,2))
    Accuracy_list_Paper2_Testing.append(acc)

    Facc = []
    for i in range (5) :
        Facc.append(round( (resultsDataFrame3Test.iloc[2,best5Testing[i]+1])*100 ,2))
    Fscore_list_Paper2_Testing.append(Facc)


#calc Mean
def getmean(all_list):
    sum_list = [sum(x) for x in zip(*all_list)]
    last = [x / 10 for x in sum_list]
    for i in range (len(last)):
        last[i] = round(last[i], 2)
    return last
#training
Accuracy_list_OurModel_mean = getmean(Accuracy_list_OurModel)
Fscore_list_OurModel_mean = getmean(Fscore_list_OurModel)
Accuracy_list_Paper1_mean = getmean(Accuracy_list_Paper1)
Fscore_list_Paper1_mean = getmean(Fscore_list_Paper1)
Accuracy_list_Paper2_mean = getmean(Accuracy_list_Paper2)
Fscore_list_Paper2_mean = getmean(Fscore_list_Paper2)
#testing
Accuracy_list_OurModel_Testing_mean = getmean(Accuracy_list_OurModel_Testing)
Fscore_list_OurModel_Testing_mean = getmean(Fscore_list_OurModel_Testing)
Accuracy_list_Paper1_Testing_mean = getmean(Accuracy_list_Paper1_Tesing)
Fscore_list_Paper1_Testing_mean = getmean(Fscore_list_Paper1_Tesing)
Accuracy_list_Paper2_Testing_mean = getmean(Accuracy_list_Paper2_Testing)
Fscore_list_Paper2_Testing_mean = getmean(Fscore_list_Paper2_Testing)


#calc standard deviation
import statistics
def getsd(all_list):
    last = [statistics.stdev(x) for x in zip(*all_list)]
    for i in range (len(last)):
        last[i] = round(last[i], 2)
    return last
#training
Accuracy_list_OurModel_sd = getsd(Accuracy_list_OurModel)
Fscore_list_OurModel_sd = getsd(Fscore_list_OurModel)
Accuracy_list_Paper1_sd = getsd(Accuracy_list_Paper1)
Fscore_list_Paper1_sd = getsd(Fscore_list_Paper1)
Accuracy_list_Paper2_sd = getsd(Accuracy_list_Paper2)
Fscore_list_Paper2_sd = getsd(Fscore_list_Paper2)
#testing
Accuracy_list_OurModel_Testing_sd = getsd(Accuracy_list_OurModel_Testing)
Fscore_list_OurModel_Testing_sd = getsd(Fscore_list_OurModel_Testing)
Accuracy_list_Paper1_Testing_sd = getsd(Accuracy_list_Paper1_Tesing)
Fscore_list_Paper1_Testing_sd = getsd(Fscore_list_Paper1_Tesing)
Accuracy_list_Paper2_Testing_sd = getsd(Accuracy_list_Paper2_Testing)
Fscore_list_Paper2_Testing_sd = getsd(Fscore_list_Paper2_Testing)


#training
Acc_table = pd.DataFrame(columns = ['OurModel_Mean','OurModel_sd', 'KNNVWC_Mean', 'KNNVWC_sd', 'RotVan_Mean', 'RotVan_sd'])
Acc_table.insert(0,'  ',['DNS','FTP','HTTP','MDNS','TCP'])
Acc_table['OurModel_Mean'] = Accuracy_list_OurModel_mean
Acc_table['OurModel_sd'] = Accuracy_list_OurModel_sd
Acc_table['KNNVWC_Mean'] = Accuracy_list_Paper1_mean
Acc_table['KNNVWC_sd'] = Accuracy_list_Paper1_sd
Acc_table['RotVan_Mean'] = Accuracy_list_Paper2_mean
Acc_table['RotVan_sd'] = Accuracy_list_Paper2_sd
#testing
Acc_table_Testing = pd.DataFrame(columns = ['OurModel_Mean','OurModel_sd', 'KNNVWC_Mean', 'KNNVWC_sd', 'RotVan_Mean', 'RotVan_sd'])
Acc_table_Testing.insert(0,'  ',['DNS','FTP','HTTP','MDNS','TCP'])
Acc_table_Testing['OurModel_Mean'] = Accuracy_list_OurModel_Testing_mean
Acc_table_Testing['OurModel_sd'] = Accuracy_list_OurModel_Testing_sd
Acc_table_Testing['KNNVWC_Mean'] = Accuracy_list_Paper1_Testing_mean
Acc_table_Testing['KNNVWC_sd'] = Accuracy_list_Paper1_Testing_sd
Acc_table_Testing['RotVan_Mean'] = Accuracy_list_Paper2_Testing_mean
Acc_table_Testing['RotVan_sd'] = Accuracy_list_Paper2_Testing_sd

#training
F_table = pd.DataFrame(columns = ['OurModel_Mean','OurModel_sd', 'KNNVWC_Mean', 'KNNVWC_sd', 'RotVan_Mean', 'RotVan_sd'])
F_table.insert(0,'  ',['DNS','FTP','HTTP','MDNS','TCP'])
F_table['OurModel_Mean'] = Fscore_list_OurModel_mean
F_table['OurModel_sd'] = Fscore_list_OurModel_sd
F_table['KNNVWC_Mean'] = Fscore_list_Paper1_mean
F_table['KNNVWC_sd'] = Fscore_list_Paper1_sd
F_table['RotVan_Mean'] = Fscore_list_Paper2_mean
F_table['RotVan_sd'] = Fscore_list_Paper2_sd
#testing
F_table_Testing = pd.DataFrame(columns = ['OurModel_Mean','OurModel_sd', 'KNNVWC_Mean', 'KNNVWC_sd', 'RotVan_Mean', 'RotVan_sd'])
F_table_Testing.insert(0,'  ',['DNS','FTP','HTTP','MDNS','TCP'])
F_table_Testing['OurModel_Mean'] = Fscore_list_OurModel_Testing_mean
F_table_Testing['OurModel_sd'] = Fscore_list_OurModel_Testing_sd
F_table_Testing['KNNVWC_Mean'] = Fscore_list_Paper1_Testing_mean
F_table_Testing['KNNVWC_sd'] = Fscore_list_Paper1_Testing_sd
F_table_Testing['RotVan_Mean'] = Fscore_list_Paper2_Testing_mean
F_table_Testing['RotVan_sd'] = Fscore_list_Paper2_Testing_sd


from IPython.display import display

print('$$$$$$$$$$$$$$$$$$$$Training$$$$$$$$$$$$$$$$$$$$')
print('\nAccuracy table for cross validation = 10: \n')
display(Acc_table)
print('\nF-Score table for cross validation = 10: \n')
display(F_table)

print('\n$$$$$$$$$$$$$$$$$$$$Testing$$$$$$$$$$$$$$$$$$$$')
print('\nAccuracy table for cross validation = 10: \n')
display(Acc_table_Testing)
print('\nF-Score table for cross validation = 10: \n')
display(F_table_Testing)


