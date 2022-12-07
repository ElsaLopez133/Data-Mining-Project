import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from kneed import KneeLocator

###################################################################
## Read data
###################################################################

data = pd.read_csv("./data_house/database.csv", sep = ',') 
column_names = data.columns
n = len(data.columns)
print("Dataset shape:", data.shape)

# Normalize data
data_normal = pd.DataFrame(StandardScaler().fit_transform(data), columns = column_names)

###################################################################
## Dimensionality reduction
###################################################################

pca = PCA(n_components = 0.70)
data = pca.fit_transform(data_normal)
data = pd.DataFrame(data)
print(data.shape)

###################################################################
## Find parameters of DBSCAN
###################################################################

neighb = NearestNeighbors(n_neighbors=n) 
nbrs=neighb.fit(data) # fitting the data to the object
distances,indices=nbrs.kneighbors(data) # finding the nearest neighbours

# Sort and plot the distances results
plt.figure()
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.savefig('./data_house/figure_pre_dbsacn') # showing the plot


###################################################################
## DBSCAN
###################################################################
# clusters
print('\n-----------------DBSCAN--------------\n')
dbscan = DBSCAN(eps = 2, min_samples = 6).fit(data) 
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_ # getting the labels

print("Labels for DBSCAN: ",np.unique(labels))
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('n_clusters for DBSCAN: ', n_clusters_)


###################################################################
## K-MEANS
###################################################################
print('\n-----------------k-means--------------\n')
kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42}
# A list holds the SSE values for each k
sse = []
for k in range(1, 30):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10,10))
plt.style.use("fivethirtyeight")
plt.plot(range(1, 30), sse)
plt.xticks(range(1, 30))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.savefig('./data_house/kmeans.png')

kl = KneeLocator(range(1, 30), sse, curve="convex", direction="decreasing")

print('knee elbow: ',kl.elbow)
   
kmeans = KMeans(init="random", n_clusters=kl.elbow, n_init=10, max_iter=300,random_state=42)
kmeans.fit(data)
print('labels k-means: ', np.unique(kmeans.labels_))
n_clusters_kmeans = len(set(kmeans.labels_)) - (1 if -1 in kmeans.labels_ else 0)
print('n_clusters for DBSCAN: ', n_clusters_kmeans)


###################################################################
## Plot
###################################################################


if data.shape[1] != 2:  
    ###################################################################
    ## PCA
    ###################################################################

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data_normal)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])
    principalDf['color'] = kmeans.labels_
    
    

    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plt.scatter(principalComponents[:,0], principalComponents[:,1], c=kmeans.labels_, cmap= "plasma") 
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.title('pca (kmeans)')
    
    plt.subplot(1,2,2)
    plt.scatter(principalComponents[:,0], principalComponents[:,1], c=labels, cmap= "plasma") 
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.title('pca (DBSCAN)')
    plt.savefig('./data_house/figure_dbsacn_pca') # showing the plot


    ###################################################################
    ## TSNE
    ###################################################################

    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(data_normal)
    tsne_result.shape
    
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plt.scatter(tsne_result[:,0], tsne_result[:,1], c=kmeans.labels_, cmap= "plasma") 
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.title('tsne (kmeans)')
    
    plt.subplot(1,2,2)
    plt.scatter(tsne_result[:,0], tsne_result[:,1], c=labels, cmap= "plasma") 
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.title('tsne (DBSCAN)')
    
    plt.savefig('./data_house/figure_dbsacn_tsne')
else:
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plt.scatter(data[:,0], data[:,1], c=kmeans.labels_, cmap= "plasma") 
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.title('kmeans')
    
    plt.subplot(1,2,2)
    plt.scatter(data[:,0], data[:,1], c=labels, cmap= "plasma") 
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.title('DBSCAN')
    
    plt.savefig('./data_house/figure_dbsacn') # showing the plot


###################################################################
## Assign queries to clusters
###################################################################

queries =  pd.read_csv("./data_house/queries_to_use.csv", sep = ',', index_col = 0)
data = pd.read_csv("./data_house/database.csv", sep = ',') 

column_names_queries = queries.columns
print(column_names_queries)
# We create an empty dataframe where we are going to store the matching outputs
matching_outputs = pd.DataFrame()

for i in range(10):
    print(queries.iloc[i])
    # We generate the condition
    condition = ""
    for j in range(1,n):
        if np.isnan(queries.iloc[i][j]):
            continue
        else:
            condition = condition + column_names_queries[j] + " == " + str(int(queries.iloc[i][j])) + ' and '
    condition = condition[:-4]
    print(condition)
            
    matching = data.query(str(condition))
    print(matching.index)
    
    
    


