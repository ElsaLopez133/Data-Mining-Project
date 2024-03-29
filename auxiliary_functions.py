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
from itertools import combinations
import random
from sklearn.preprocessing import MinMaxScaler

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
      
def queries_as_sets(queries, filename):
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
        
    with open(str(filename), "w") as outfile:
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
        

###################################################################
## Combinations
###################################################################          

def combination(row):
    list_combinations = list()
    for n in range(len(row) + 1):
        list_combinations += list(combinations(row, n))
    for i in range(len(list_combinations)):
        if len(list_combinations[i]) == 1:
            list_combinations[i] = list_combinations[i][0]
    return list_combinations[1:], len(list_combinations) -1


###################################################################
## Matches
################################################################### 

def matching_queries(length, query_columns, query, dict_query, queries):
    if (length == 1):
        print('Case 1: one common value')
        idx = list(queries[queries[str(query_columns[0])] == query.iloc[0,0]].index)
        dict_query.update( {str(query_columns[0]) : idx} )
        
        print('Dictionary: ', dict_query)
            
    elif (length == 2):
        print('Case 2: up to 2 common value')
        for i in range(2):
            idx = list(queries[queries[str(query_columns[i])] == query.iloc[0,i]].index)
            dict_query.update( {str(query_columns[i]) : idx} )
                
        idx = list(queries[queries[str(query_columns[0])] == query.iloc[0,0]][queries[str(query_columns[1])] == query.iloc[0,1]].index)
        cond = str(query_columns[0]), str(query_columns[1])
        dict_query.update( {str(cond) : idx} )
            
    elif (length == 3):
        print('Case 3: up to 3 common value')
        for i in range(3):
            idx = list(queries[queries[str(query_columns[i])] == query.iloc[0,i]].index)
            dict_query.update( {str(query_columns[i]) : idx} )
                
        for j in range(3):
            for i in range(j+1,3):
                idx = list(queries[queries[str(query_columns[j])] == query.iloc[0,j]][queries[str(query_columns[i])] == query.iloc[0,i]].index)
                cond = str(query_columns[j]), str(query_columns[i])
                dict_query.update( {str(cond) : idx} )
        
        idx = list(queries[queries[str(query_columns[0])] == query.iloc[0,0]][queries[str(query_columns[1])] == query.iloc[0,1]][queries[str(query_columns[2])] == query.iloc[0,2]].index)  
        cond = str(query_columns[0]), str(query_columns[1]), str(query_columns[2])
        dict_query.update( {str(cond) : idx} ) 
        
    elif (length == 4):
        print('Case 4: up to 4 common value')
        for i in range(4):
            idx = list(queries[queries[str(query_columns[i])] == query.iloc[0,i]].index)
            dict_query.update( {str(query_columns[i]) : idx} )
                
        for j in range(4):
            for i in range(j+1,4):
                idx = list(queries[queries[str(query_columns[j])] == query.iloc[0,j]][queries[str(query_columns[i])] == query.iloc[0,i]].index)
                cond = str(query_columns[j]), str(query_columns[i])
                dict_query.update( {str(cond) : idx} )
        
        for j in range(4):
            for i in range(j+1,4):
                for k in range(i+1,4):
                    idx = list(queries[queries[str(query_columns[j])] == query.iloc[0,j]][queries[str(query_columns[i])] == query.iloc[0,i]][queries[queries[str(query_columns[k])] == query.iloc[0,k]]].index)
                    cond = str(query_columns[j]), str(query_columns[i]), str(query_columns[k])
                    dict_query.update( {str(cond) : idx} )
        
        
        idx = list(queries[queries[str(query_columns[0])] == query.iloc[0,0]][queries[str(query_columns[1])] == query.iloc[0,1]][queries[str(query_columns[2])] == query.iloc[0,2]][queries[str(query_columns[3])] == query.iloc[0,3]].index)  
        cond = str(query_columns[0]), str(query_columns[1]), str(query_columns[2]), str(query_columns[3] )
        dict_query.update( {str(cond) : idx} ) 
        
        with open("query_matches_partb.json", "w") as outfile:
            json.dump(dict_query, outfile)
    return dict_query


###################################################################
## Find k highest values
################################################################### 

def find_highest_values(list_to_search, ordered_nums_to_return=None):
    if ordered_nums_to_return:
        return sorted(set(list_to_search), reverse=True)[0:ordered_nums_to_return]
    return [sorted(list_to_search, reverse=True)[0]]


###################################################################
## Remove ranodm values of utility matrix
################################################################### 

def remove_numbers(user_queries, len_list, row, columns, list_remove, user_queries_test):
    while len_list < 600:
        i = random.randint(0,row-1)
        j = random.randint(0,columns-1)
        if pd.isnull(user_queries.iloc[i,j]):
            continue
        else:
            if [i,j] in list_remove:
                continue
            else:
                list_remove.append([i,j])
                user_queries_test.iloc[i,j] = np.nan
        
        len_list = len(list_remove)
    return list_remove, user_queries_test

###################################################################
## Remove ranodm values of user in utility matrix
################################################################### 

def remove_numbers2(user_queries, len_list, row, columns, user_queries_test):
    #We select a random user
    user = random.randint(1,len(user_queries))
    li = list(user_queries.iloc[user,1:])
    li_not_nan = [i for i, element in enumerate(li) if np.isnan(element)]
    li_nan = [i for i in range(len(li)) if i not in li_not_nan]
        
    len_list = len(li_nan)
    
    list_remove = []
    while len_list < 1400:
        j = random.randint(0,len(li_not_nan)-1)
        
        if [user,li_not_nan[j]] in list_remove:
            continue
        else:
            list_remove.append([user,li_not_nan[j]])
            user_queries_test.iloc[user,li_not_nan[j]] = np.nan
        
        len_list += len(list_remove)
    return user,list_remove, user_queries_test


def remove_numbers3(user_queries, len_list, row, columns, user_queries_test):
    #We select a random query
    query = random.randint(0,columns-1)

    li = list(user_queries.iloc[1:,query])
    li_not_nan = [i for i, element in enumerate(li) if np.isnan(element)]
    li_nan = [i for i in range(len(li)) if i not in li_not_nan]
        
    len_list = len(li_nan)
    
    list_remove = []
    while len_list < 70:
        i = random.randint(0,len(li_not_nan)-1)
        
        if [li_not_nan[i],query] in list_remove:
            continue
        else:
            list_remove.append([li_not_nan[i],query])
            user_queries_test.iloc[li_not_nan[i],query] = np.nan
        
        len_list += len(list_remove)
    return query,list_remove, user_queries_test

###################################################################
## Recommender function
################################################################### 

def utility_matrix_rec(path_user_queries_test, labels, n, kmeans_labels, matching_outputs_name, user_queries_fill_name ):
    #print('-------------------query assignement-------------\n')
    queries =  pd.read_csv("./data_house/queries_to_use.csv", sep = ',', index_col = 0)
    data = pd.read_csv("./data_house/database.csv", sep = ',') 
    data['cluster_id_kmeans'] = kmeans_labels
    data['cluster_id_dbscan'] = labels
    column_names_queries = queries.columns

    matching_outputs = queries_to_tuples(queries,data, kmeans_labels, n)
    matching_outputs.to_csv(matching_outputs_name, header = False, sep = ',', index=False)
    maxValueIndex = matching_outputs.idxmax(axis = 1)
    queries['kmeans_label_id'] = maxValueIndex

    #print('-------------queries-----------\n')
    event_counts = collections.Counter(queries['kmeans_label_id'])

    #print('-------------database-----------\n')
    event_counts = collections.Counter(data['cluster_id_kmeans'])
    
    #print('--------------jaccard similarity-----------\n')
        
    user_queries =  pd.read_csv(path_user_queries_test, sep = ',', index_col = 0)
    recomendations_index = pd.DataFrame(0, index = range(len(user_queries)), columns =['user_id','top1', 'top2', 'top3', 'top4', 'top5'])
    recomendations_value = pd.DataFrame(0, index = range(len(user_queries)), columns =['user_id','top1', 'top2', 'top3', 'top4', 'top5'])

    for i in range(len(user_queries)):
        gvn_jsonfile = open("./jsonfiles/query_set.json")
        json_data = json.load(gvn_jsonfile)
        
        #print("---------------user {}------------\n ".format(i+1))
        dict_cluster = {}
        average_cluster = {}
        user_queries_non_nan = []
        user_queries_nan = []
        
        # We create lists containing the indexes of no ranked queries and ranked queries
        for t,j in user_queries.iloc[i][1:].items():           
            if (np.isnan(j)):
                user_queries_nan.append(t)
            else:
                user_queries_non_nan.append(t)
            n_nan_queries = len(user_queries_nan)
            count = [0,0,0]

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
                similarity_value = jaccard_similarity(set_query_non_nan, set_query_nan)
                # similarity.append(similarity_value)
                
                if similarity_value > min(value_top_3):
                    min_index = value_top_3.index(min(value_top_3))
                    index_top_3[min_index] = int(query_id)
                    value_top_3[min_index] = similarity_value 
            
            # Fill the ranking of the current nan query for the current user by averaging the top 3 values
            ranking, count = ranking_calculation(i,index_top_3, value_top_3, user_queries, average_cluster, key, count)
            user_queries.at[i, str(item)] = ranking
            
            
            min_value = min(value_top_ranking) 
            if ranking > min_value:
                min_index_ranking = value_top_ranking.index(min(value_top_ranking))
                index_top_ranking[min_index_ranking] = int(item)
                value_top_ranking[min_index_ranking] =  float(ranking) 
        
        user_queries.to_csv(user_queries_fill_name, header = True, sep = ',')


###################################################################
## Normalizing utility matrix
################################################################### 
def scaler_utility_matrix(user_queries):
    seq = user_queries['user_id']
    x = user_queries.iloc[:,2:].values #returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.transpose())
    user_queries_minmax = pd.DataFrame(x_scaled.transpose()*100)
    user_queries_minmax.insert(0,'user_id',seq)
    return user_queries_minmax