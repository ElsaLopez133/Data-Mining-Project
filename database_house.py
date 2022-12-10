import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from faker import Faker
from collections import defaultdict
from sqlalchemy import create_engine
import faker.providers
from faker.providers import DynamicProvider
import random

#Generate dummy data. We initialize the Faker instance.
fake = Faker()

fake_data = defaultdict(list)
fake_user = defaultdict(list)
fake_queries = defaultdict(list)

for _ in range(1000):
    fake_data['nrooms'].append(random.randint(1,5))
    fake_data['nbedrooms'].append(random.randint(1,3))
    fake_data['nbath'].append(random.randint(1,3))
    fake_data['sm'].append(random.randint(1,40))
    fake_data['garden_sm'].append(random.randint(1,10))
    fake_data['floors'].append(random.randint(1,3))
    fake_data['gargae_sm'].append(random.randint(1, 10))
    fake_data['price'].append(random.randint(1, 50))
    fake_data['year'].append(random.randint(1, 30))
    fake_data['windows'].append(random.randint(1,20))
    fake_data['dist_city'].append(random.randint(1, 20))
    fake_data['doors'].append(random.randint(1,10))
    
    
df_fake_data = pd.DataFrame(fake_data)
df_fake_data.to_csv('data_house/database.csv', header = True, sep = ',')

print(df_fake_data[:10])
n = len(df_fake_data.columns)

# We create the users with their id
for _ in range(100):
    fake_user["user_id"].append(fake.unique.ssn())
    df_fake_user = pd.DataFrame(fake_user, columns=['user_id'])
    
df_fake_user.to_csv('data_house/user.csv', header = True, sep = ',')

# We create the queries. We can create queries with 1 until n conditions, where n is the number of columns of fake_data
with open('data_house/queries.csv', 'w', encoding='UTF8', newline = '') as f:
    writer = csv.writer(f)
    columns_names = ['query_id'] 
    columns_names.extend(df_fake_data.columns)
    df_fake_queries = pd.DataFrame(index = range(2000), columns = columns_names)

    for i in range(2000):
        row = [i]
        df_fake_queries['query_id'].iloc[i] = i
        
        m = random.randint(1,4)
        df = df_fake_data.sample(n = m, axis = 'columns').sample()
        
        for j in range(len(df.columns)):
            row.append(''.join((str(df.columns[j]),'=',str(df.iloc[0][j]))))
            df_fake_queries[str(df.columns[j])].iloc[i] = df.iloc[0][j]
            
        writer.writerow(row)

print(df_fake_queries[:5])
df_fake_queries.to_csv('data_house/queries_to_use.csv', header = True, sep = ',')


column_names = [i for i in df_fake_queries['query_id']]
df_user_queries = pd.DataFrame(columns = column_names)

values = np.array([[60,100],[1,100],[1,50],[30,80]])
for i in range(len(df_fake_user)):
    index = np.random.randint(0, len(values),1)[0]
    # We choose from 100 to 1000 quereis for each user
    m = random.randint(500,1000)
    queries = np.random.randint(0,len(df_fake_queries),m)
    
    row = []
    for j in range(len(df_fake_queries)):
        if df_fake_queries['query_id'].iloc[j] in queries:
            row.append(np.random.randint(values[index][0],values[index][1],1)[0])
        else:
            row.append(np.nan)
            
    df_user_queries.loc[len(df_user_queries.index)] = row


df_user_queries.insert(0, "user_id", [k for k in df_fake_user['user_id']], True)
print(df_user_queries)
df_user_queries.to_csv('data_house/user_queries.csv', header = True, sep = ',', index=False)



'''
df_user_queries = pd.DataFrame(columns = ['query_id', 'user_id', 'rank'])
for i in range(len(df_fake_user)):
    # We choose from 1 to 200 quereis for each user
    m = random.randint(1,200)
    queries = np.random.randint(0,len(df_fake_queries),m)
    user = [df_fake_user.iloc[i][0]]*m
    rank = np.random.randint(0,100,m)
    
    df_temp = pd.DataFrame(list(zip(queries, user, rank)), columns = ['query_id', 'user_id', 'rank'])
    df_user_queries = pd.concat([df_user_queries, df_temp], ignore_index=True)
    df_user_queries.reset_index()

print(df_user_queries)
df_user_queries.to_csv('data_house/user_queries.csv', header = True, sep = ',', index=False)
    
'''
