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
import re

medical_professions_provider = DynamicProvider(
     provider_name="medical_profession",
     elements=["dr.", "doctor", "nurse", "surgeon", "clerk"],
)

colors_provider = DynamicProvider(
     provider_name="colors",
     elements=["orange", "yellow", "blue", "red", "green", "black", "pink", "white"],
)

data2 = pd.read_csv("data/flavors.csv", sep = "," )
aliment_provider = DynamicProvider(
     provider_name="aliment",
     elements=data2['aliment'].tolist(),
)

data2 = data2.dropna()
flavor_provider = DynamicProvider(
     provider_name="flavor",
     elements=data2['flavor'].tolist(),
)

data = pd.read_csv("data/it.csv", sep = "," )
city_provider = DynamicProvider(
     provider_name="city",
     elements= data['city'][:100].tolist(),
)

region_provider = DynamicProvider(
     provider_name="region",
     elements= data['admin_name'].tolist(),
)

data1 = pd.read_csv("data/animal_names.csv", sep = "," )
animals_provider = DynamicProvider(
     provider_name="animal",
     elements= data1['name'][:500].tolist(),
)

#Generate dummy data. We initialize the Faker instance.
fake = Faker()

# then add new provider to faker instance
fake.add_provider(medical_professions_provider)
fake.add_provider(colors_provider)
fake.add_provider(city_provider)
fake.add_provider(region_provider)
fake.add_provider(animals_provider)
fake.add_provider(flavor_provider)
fake.add_provider(aliment_provider)

# We’ll use fake_data to create our dictionary. defaultdict(list) will create a dictionary that will create key-value pairs 
# that are not currently stored within the dictionary when accessed. Essentially, you do not need to define any keys within 
# your dictionary.

fake_data = defaultdict(list)
fake_user = defaultdict(list)
fake_queries = defaultdict(list)

for _ in range(1000):
    fake_data["first_name"].append( fake.first_name() )
    fake_data["last_name"].append( fake.last_name() )
    fake_data["occupation"].append( fake.job() )
    fake_data["dob"].append( fake.date_of_birth() )
    fake_data["country"].append( fake.country() )
    fake_data["color"].append(fake.colors())
    fake_data["aliment"].append(fake.aliment())
    fake_data["company"].append(fake.company())
    fake_data["medical"].append(fake.medical_profession())
    fake_data["flavor"].append(fake.flavor())
    fake_data["city"].append(fake.city())
    fake_data["region"].append(fake.region())
    fake_data["animal"].append(fake.animal())
    fake_data["user_agent"].append(fake.user_agent())
    fake_data["currency"].append(fake.currency())

df_fake_data = pd.DataFrame(fake_data)

df_fake_data.to_csv('data/database.csv', header = True, sep = ',')

print(df_fake_data[:10])
n = len(df_fake_data.columns)

# We create the users with their id
for _ in range(100):
    fake_user["user_id"].append(fake.unique.ssn())
    
df_fake_user = pd.DataFrame(fake_user, columns=['user_id'])
df_fake_user.to_csv('data/user.csv', header = True, sep = ',')

# We create the queries. We can create queries with 1 until n conditions, where n is the number of columns of fake_data
with open('data/queries.csv', 'w', encoding='UTF8', newline = '') as f:
    writer = csv.writer(f)
    columns_names = ['query_id'] 
    columns_names.extend(df_fake_data.columns)
    df_fake_queries = pd.DataFrame(index = range(2000), columns = columns_names)

    for i in range(2000):
        row = [i]
        df_fake_queries['query_id'].iloc[i] = i
        
        m = random.randint(1,n)
        df = df_fake_data.sample(n = m, axis = 'columns').sample()
        
        for j in range(len(df.columns)):
            row.append(''.join((str(df.columns[j]),'=',str(df.iloc[0][j]))))
            df_fake_queries[str(df.columns[j])].iloc[i] = df.iloc[0][j]
            
        writer.writerow(row)

print(df_fake_queries[:5])

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
df_user_queries.to_csv('data/user_queries.csv', header = True, sep = ',')
    
 # We remove teh double quotes from the csv files
    
