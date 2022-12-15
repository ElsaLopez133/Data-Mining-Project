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

###################################################################
## Function to calculate the ranking of new queries
###################################################################

def ranking_calculation(i,index_top_3, value_top_3, user_queries, average_cluster, key, count):
    #We define a count vector that keeps track of the number of cases
    # Edge case if all top 3 are 0 average of all queries in a given cluster instead for ranking
    if all([val == 0 for val in index_top_3]):
        #print('we are in case 1\n')
        ranking = round(average_cluster[key][0],2)
        count[0] += 1
        
    # Weighted ranking based on similarity score of top 3!
    elif all([val != 0 for val in index_top_3]):
        #print('we are in case 2\n')
        rankings = [int(user_queries[str(index_top_3[j])].iloc[i]) for j in range(len(index_top_3))]
        ranking = round(np.average(rankings, weights = value_top_3),2)   
        count[1] += 1
    # Edge case if some of the top 3 are 0, don't use them!
    else:
        #print('we are in case 3\n')
        while 0 in index_top_3:
            index = index_top_3.index(0)
            value_top_3.pop(index)
            index_top_3.pop(index)
        
        rankings = [int(user_queries[str(index_top_3[j])].iloc[i]) for j in range(len(index_top_3))]
        if len(index_top_3) == 2:
            ranking = round(np.average(rankings, weights = value_top_3),2)
        else:
            ranking = rankings[0]
        count[2] += 1    
    return ranking, count


###################################################################
## Function to represent queries as "sets"
###################################################################
      
def queries_as_sets(queries):
    column_names_queries = queries.columns
    dict_queries = {}
    for i in range(len(queries)):
        set_query = []
        for j in range(1, len(queries.columns)-1):
            if np.isnan(queries.iloc[i][j]):
                continue
            else:
                for k in range(int(queries.iloc[i][j])):
                    set_query.append(column_names_queries[j])
        dict_queries.update( {str(queries['query_id'].iloc[i]) : set_query} )
        
    with open("query_set.json", "w") as outfile:
        json.dump(dict_queries, outfile)
    return dict_queries


###################################################################
## Function to assign queries to clusters based on tuples
###################################################################
    
def queries_to_tuples(queries,data,labels,n):
    column_names_queries = queries.columns
    # We create an empty dataframe where we are going to store the matching outputs
    columns_matching = [str(i) for i in np.unique(labels)]
    matching_outputs = pd.DataFrame(0, index = range(len(queries)), columns=columns_matching)

    for i in range(len(queries)):
        # We generate the condition
        condition = ""
        for j in range(1,n):
            if np.isnan(queries.iloc[i][j]):
                continue
            else:
                condition = condition + column_names_queries[j] + " == " + str(int(queries.iloc[i][j])) + ' and '
        condition = condition[:-4]                
        matching = data.query(str(condition))
        
        for k in range(len(matching)):
            label_id = data['cluster_id_kmeans'].iloc[k]
            matching_outputs[str(label_id)].iloc[i] += 1
    return matching_outputs


###################################################################
## Function to sort list based on a different list
###################################################################

def sort_by_indexes(lst, indexes, reverse=False):
  return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: \
          x[0], reverse=reverse)]

###################################################################
## Jaccard Similarity
###################################################################

def jaccard_similarity(A, B):
    #Find intersection
    nominator = intersection(A,B)
    #Find union 
    denominator = union(A,B)
    #Take the ratio of sizes
    similarity = len(nominator)/len(denominator)
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

###################################################################
## Function to plot 
###################################################################

def plot_data(data, data_normal, kmeans_labels, dbscan_labels):
    if data.shape[1] != 2: 

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(data_normal)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2'])
        principalDf['color'] = kmeans_labels

        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.scatter(principalComponents[:,0], principalComponents[:,1], c=kmeans_labels, cmap= "plasma") 
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('pca (kmeans)')
        
        plt.subplot(1,2,2)
        plt.scatter(principalComponents[:,0], principalComponents[:,1], c=dbscan_labels, cmap= "plasma") 
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('pca (DBSCAN)')
        plt.savefig('./data_house/figure_dbsacn_pca') # showing the plot

        n_components = 2
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(data_normal)
        tsne_result.shape
        
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.scatter(tsne_result[:,0], tsne_result[:,1], c=kmeans_labels, cmap= "plasma") 
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('tsne (kmeans)')
        
        plt.subplot(1,2,2)
        plt.scatter(tsne_result[:,0], tsne_result[:,1], c=dbscan_labels, cmap= "plasma") 
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('tsne (DBSCAN)')
        plt.savefig('./data_house/figure_dbsacn_tsne')
    else:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.scatter(data[:,0], data[:,1], c=kmeans_labels, cmap= "plasma") 
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('kmeans')
        
        plt.subplot(1,2,2)
        plt.scatter(data[:,0], data[:,1], c=dbscan_labels, cmap= "plasma") 
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('DBSCAN')
        plt.savefig('./data_house/figure_dbsacn') # showing the plot