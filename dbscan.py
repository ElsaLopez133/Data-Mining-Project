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
import json
import auxiliary_functions
###################################################################
## Read data
###################################################################
print('------------Read data--------------\n')
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
print('\n----------------Plot-----------------\n')
auxiliary_functions.plot_data(data, data_normal, kmeans.labels_, labels)


###################################################################
## Assign queries to clusters
###################################################################
print('-------------------query assignement-------------\n')

queries =  pd.read_csv("./data_house/queries_to_use.csv", sep = ',', index_col = 0)
data = pd.read_csv("./data_house/database.csv", sep = ',') 
data['cluster_id_kmeans'] = kmeans.labels_
data['cluster_id_dbscan'] = labels
column_names_queries = queries.columns

matching_outputs = auxiliary_functions.queries_to_tuples(queries,data, kmeans.labels_, n)
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

dict_queries = auxiliary_functions.queries_as_sets(queries)
    
user_queries =  pd.read_csv("./data_house/utility_matrix.csv", sep = ',')
recomendations_index = pd.DataFrame(0, index = range(len(user_queries)), columns =['user_id','top1', 'top2', 'top3', 'top4', 'top5'])
recomendations_value = pd.DataFrame(0, index = range(len(user_queries)), columns =['user_id','top1', 'top2', 'top3', 'top4', 'top5'])

for i in range(len(user_queries)):
    gvn_jsonfile = open("./data_house/query_set.json")
    json_data = json.load(gvn_jsonfile)
    
    print("---------------user {}------------\n ".format(i+1))
    dict_cluster = {}
    average_cluster = {}
    user_queries_non_nan = []
    user_queries_nan = []
    
    # We create lists containing the indexes of no ranked queries and ranked queries
    count = 0
    for t,j in user_queries.iloc[i][1:].items():           
      if (np.isnan(j)):
          user_queries_nan.append(t)
      else:
          user_queries_non_nan.append(t)

    # Create a dictionary
    for j in range(len(np.unique(queries['kmeans_label_id']))):
        dict_cluster.update({str(np.unique(queries['kmeans_label_id'])[j]) : []})
        average_cluster.update({str(np.unique(queries['kmeans_label_id'])[j]) : []})
    
    for k in range(len(user_queries_non_nan)):
        dict_cluster[str(queries['kmeans_label_id'].iloc[k])].append(user_queries_non_nan[k])
            
    # We calculate the average ranking of ranked queries in each cluster
    for j in range(len(np.unique(queries['kmeans_label_id']))):
        key = str(np.unique(queries['kmeans_label_id'])[j])
        ranking_temp = []
        for query_id in dict_cluster[key]:
            ranking_temp.append(user_queries[str(query_id)].iloc[i])
        average_cluster[key].append(sum(ranking_temp)/len(ranking_temp))

    index_top_ranking = [0,0,0,0,0]
    value_top_ranking = [0,0,0,0,0]

    for item in user_queries_nan:
        set_query_nan = json_data[str(item)]
        
        key = str(queries['kmeans_label_id'].iloc[int(item)])
        similarity = []
        index_top_3 = [0,0,0]
        value_top_3 =[0,0,0]
                
        for query_id in dict_cluster[key]:
            set_query_non_nan = json_data[str(query_id)]
            similarity_value = auxiliary_functions.jaccard_similarity(set_query_non_nan, set_query_nan)
            # similarity.append(similarity_value)
            
            if similarity_value > min(value_top_3):
                min_index = value_top_3.index(min(value_top_3))
                index_top_3[min_index] = int(query_id)
                value_top_3[min_index] = similarity_value 
        
        # Fill the ranking of the current nan query for the current user by averaging the top 3 values
        ranking = auxiliary_functions.ranking_calculation(i,index_top_3, value_top_3, user_queries, average_cluster, key)
        user_queries.at[i, str(item)] = ranking
        
        min_value = min(value_top_ranking) 
        if ranking > min_value:
            min_index_ranking = value_top_ranking.index(min(value_top_ranking))
            index_top_ranking[min_index_ranking] = int(item)
            value_top_ranking[min_index_ranking] =  float(ranking) 
        
        
    # Return top k queries which were previously nan and now have a high rating
    #print('Any left nan values: ', user_queries.iloc[i].hasnans)
    #print("Reccommended queries for user {}: {} .".format(user_queries['user_id'].iloc[i], sort_by_indexes(index_top_ranking, value_top_ranking, True)))
    
    # Write in the dataframe 
    recomendations_index.iloc[i] = [user_queries['user_id'].iloc[i]] + auxiliary_functions.sort_by_indexes(index_top_ranking, value_top_ranking, True)
    value_top_ranking.sort(reverse = True)
    recomendations_value.iloc[i] = [user_queries['user_id'].iloc[i]] + value_top_ranking

# We save the dataframe   
recomendations_index.to_csv("./data_house/recomendations_index.csv", sep = ',', header = True, index = False)
recomendations_value.to_csv("./data_house/recomendations_value.csv", sep = ',', header = True, index = False)

###################################################################
## Fill out utility matrix
###################################################################
print(user_queries)
user_queries.to_csv('./data_house/utility_matrix_fill.csv', header = True, sep = ',')
user_queries =  pd.read_csv("./data_house/utility_matrix_fill.csv", sep = ',')

