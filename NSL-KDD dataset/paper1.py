import matplotlib.pyplot as plt;
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from kneed import DataGenerator, KneeLocator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn import metrics

path=r''
traData = pd.read_csv(path+'trainingGroup.csv',index_col=None, header=None)
#traData = pd.read_csv('D:\\drive\\publications_2017\\datasets\\NSL-KDD\\NSL-KDD\\testingData.csv'  , index_col = None , header = None)
tesData = pd.read_csv(path+'testingGroup.csv',index_col=None, header=None)
target = 41

y_train = traData[[target]].values.reshape(len(traData)) 
x_train = preprocessing.normalize(traData.iloc[:,:-1])

y_test = tesData[[target]].values.reshape(len(tesData)) 
x_test = preprocessing.normalize(tesData.iloc[:,:-1])

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

labelled=np.zeros(x_train.shape[0])
labelled[:]=-1
li={1:60,2:60,3:30,4:10,5:10}
labelledIndex=chooseObsMan(li)
labelled[labelledIndex]=y_train[labelledIndex]

## ---->>> takes so  much time use it if u want
#method to find the nearest clusters :--- output is 5
def findBestclusters():
    kmeans_kwargs = { "init": "random", "n_init": 10, "max_iter": 300, "random_state": 42, }
    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 25):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(x_train)
        sse.append(kmeans.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 25), sse)
    plt.xticks(range(1, 25))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(1, 25), sse, curve="convex", direction="decreasing")
    return kl.elbow    #result is 5

#print(findBestclusters())

kmeans = KMeans( init="random", n_clusters=10, n_init=10, max_iter=300, random_state=42).fit(x_train)

# Final locations of the centroid
#print(kmeans.cluster_centers_)

# The number of iterations required to converge
#print(kmeans.n_iter_)

clusters = np.full((10, 6), 0, dtype=int)
for i in range(kmeans.labels_.shape[0]):
    clusters[ kmeans.labels_[i] ][ y_train[i] ] = clusters[ kmeans.labels_[i] ][ y_train[i] ] + 1

df = pd.DataFrame(clusters[:,1:11]
                    , index = ["Cluster0", "Cluster1", "Cluster2", "Cluster3", "Cluster4","Cluster5", "Cluster6", "Cluster7", "Cluster8", "Cluster9"]
                    , columns=['normal', 'DoS', 'Probe', 'R2L', 'U2R'])
print('total number of points in each cluster: \n')
print(df)
#print(clusters)
print('\nLABEL OF EACH CLUSTER :\ncluster0-->Probe\ncluster1-->DoS\ncluster2-->normal\ncluster3-->R2L\ncluster4-->Probe\ncluster5-->DoS\ncluster6-->U2R')
print('cluster7-->Probe\ncluster8-->normal\ncluster9-->normal')

#KNN model using Kmeans results
clustering_out = [3, 2, 1, 4, 3, 2, 5, 3, 1, 1]
clustering_out_label = ['Probe', 'DoS', 'normal', 'R2L', 'Probe', 'DoS', 'U2R', 'Probe', 'normal', 'normal']

knn_model = KNeighborsRegressor(n_neighbors=1)
knn_model.fit(kmeans.cluster_centers_, clustering_out)

train_preds = knn_model.predict(x_train)
print((train_preds == y_train).mean())

matrices = metrics.precision_recall_fscore_support(y_train, train_preds,labels=[1, 2, 3, 4, 5])
resultsDataFrame = pd.DataFrame()
for i in range(0,len(matrices)):
    temp = pd.DataFrame([matrices[i]],columns=['normal','DoS','Probe','R2L','U2R'])
    resultsDataFrame = resultsDataFrame.append(temp, ignore_index=True)
resultsDataFrame.insert(0,'  ',['Precision','Recall','Fscore','Support'])
print(resultsDataFrame)

correctLabelled=metrics.accuracy_score(y_train, train_preds,normalize=False)
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

