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
import collections
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
dbscan = DBSCAN(eps = 1.75, min_samples = 8).fit(data) 
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
print('n_clusters for kmeans: ', n_clusters_kmeans)


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
data['cluster_id_kmeans'] = kmeans.labels_
data['cluster_id_dbscan'] = labels
print(data.head())

column_names_queries = queries.columns
#print(column_names_queries)

# We create an empty dataframe where we are going to store the matching outputs
columns_matching = [str(i) for i in np.unique(kmeans.labels_)]
matching_outputs = pd.DataFrame(0, index = range(len(queries)), columns=columns_matching)

for i in range(len(queries)):
    #print(queries.iloc[i])
    # We generate the condition
    condition = ""
    for j in range(1,n):
        if np.isnan(queries.iloc[i][j]):
            continue
        else:
            condition = condition + column_names_queries[j] + " == " + str(int(queries.iloc[i][j])) + ' and '
    condition = condition[:-4]
    #print(condition)
            
    matching = data.query(str(condition))
    #print(matching.index)
    
    for k in range(len(matching)):
        label_id = data['cluster_id_kmeans'].iloc[k]
        #print(label_id)
        matching_outputs[str(label_id)].iloc[i] += 1
    

print(matching_outputs[:10])    
matching_outputs.to_csv('data_house/matching_outputs.csv', header = False, sep = ',', index=False)
maxValueIndex = matching_outputs.idxmax(axis = 1)
queries['kmeans_label_id'] = maxValueIndex


print('-------------queries-----------\n')
print("Used clusters: ", np.unique(queries['kmeans_label_id']))
event_counts = collections.Counter(queries['kmeans_label_id'])
import pprint
pprint.pprint(event_counts)

print('-------------database-----------\n')
print("Database: ", np.unique(data['cluster_id_kmeans']))
event_counts = collections.Counter(data['cluster_id_kmeans'])
import pprint
pprint.pprint(event_counts)

###################################################################
## Jaccard similarity between queries
###################################################################
print('--------------jaccard similarity-----------\n')
# Create the set for each query
#print(column_names_queries)
dict_queries = {}
for i in range(len(queries)):
    set_query = []
    for j in range(12):
        if np.isnan(queries.iloc[i][j+1]):
            continue
        else:
            for k in range(int(queries.iloc[i][j+1])):
                set_query.append(column_names_queries[j+1])
    dict_queries.update( {str(queries['query_id'].iloc[i]) : set_query} )


def jaccard_similarity(A, B):
    
    #Find intersection
    nominator = intersection(A,B)
    #print(nominator)
    #Find union 
    denominator = union(A,B)
    #print(denominator)
    #Take the ratio of sizes
    similarity = len(nominator)/len(denominator)
    
    
    #print(similarity)
    return similarity

def intersection(lst1, lst2):
    lst3 = []
    for value in lst1:
        if ((value in lst2) & (len(lst2) > 0)):
            lst3.append(str(value))
            lst2.remove(value)
    return lst3

def union(lst1, lst2):
    lst3 = lst1 + lst2 
    return lst3
    

user_queries =  pd.read_csv("./data_house/user_queries.csv", sep = ',', index_col = 0)

for i in range(1): #TODO: Update!
    dict_cluster = {}
    user_queries_non_nan = []
    user_queries_nan = []
    
    for t,j in user_queries.iloc[i].items():
        if (np.isnan(j)):
            user_queries_nan.append(t)
        else:
            user_queries_non_nan.append(t)
                      
    #user_queries_non_nan = user_queries.iloc[i].dropna() 
    #user_queries_nan = user_queries.iloc[i].isnan()

    # Create a dictionary
    for j in range(len(np.unique(queries['kmeans_label_id']))):
        dict_cluster.update({str(np.unique(queries['kmeans_label_id'][j][0])) : []})
    #index_queries = user_queries_non_nan.index
    for k in range(len(user_queries_non_nan)):
        dict_cluster[str([queries['kmeans_label_id'].iloc[k]])].append(user_queries_non_nan[k])
    #print(dict_cluster)
    
    #print(dict_queries)
    for item in user_queries_nan:
        key = str([queries['kmeans_label_id'].iloc[int(item)]])
        similarity = []
        index_top_3 = [0,0,0]
        value_top_3 =[0,0,0]
        for query_id in dict_cluster[key]:
            similarity_value = jaccard_similarity(dict_queries[str(item)], dict_queries[str(query_id)])
            
            if similarity_value > min(value_top_3):
                min_index = value_top_3.index(min(value_top_3))
                index_top_3[min_index] = int(query_id)
                value_top_3[min_index] = similarity_value 
                 
        if any(index_top_3) != 0:
        # similarity.append(jaccard_similarity(dict_queries[str(item)], dict_queries[str(query)]))
            print(value_top_3)
            print(index_top_3)
    
        
        # Fill the ranking of the current nan query for the current user by averaging the top 3 values
        if any(index_top_3) != 0:
            
            rankings = [user_queries[str(index_top_3[0])].iloc[i], user_queries[str(index_top_3[1])].iloc[i], user_queries[str(index_top_3[2])].loc[i]]
            print(rankings)

        # Weighted ranking based on similarity score of top 3!
        
        # Edge case if some of the top 3 are 0, don't use them!

        # Edge case if all top 3 are 0 average of all queries in a given cluster instead for ranking

        # Return top k queries which were previously nan and now have a high rating

      
    

###################################################################
## Fill out utility matrix
###################################################################


'''
fill_value = pd.DataFrame({col: user_queries.mean(axis=1) for col in user_queries.columns})
user_queries.fillna(fill_value.astype(int), inplace=True)
print(user_queries[:10])

df_fake_user = pd.read_csv("./data_house/user.csv", sep = ',')
user_queries.insert(0, "user_id", [k for k in df_fake_user['user_id']], True)
user_queries.to_csv('data_house/user_queries_fill.csv', header = True, sep = ',', index=False)

    
'''

    


