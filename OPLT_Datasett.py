#!/usr/bin/env python
# coding: utf-8

# In[154]:


import pandas as pd
import pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# In[155]:


pd.options.mode.chained_assignment = None
#df = pd.read_excel('C:\Users\k.samarth.ashvinbhai\Desktop\Work folder\Excel files')


# In[156]:


Holidays=set(['01-01','04-19','05-01','06-20','12-24','12-25','12-31','09-07','10-12','11-15'])
def get_inbetween_date(start):
    delta=start[1]-start[0]
    date_range = [(start[0] + timedelta(days=x)).strftime("%m-%d") for x in list(range(delta.days + 1))]
    if len(date_range) == 0:
        return 0
    result = len(set(date_range).intersection(Holidays)) 
    return result
#df['Holiday_Delta'] = df[["Created on", "Pstng Date"]].apply(get_inbetween_date, axis=1)
#df.to_excel("OPLT_Dataset.xlsx")


# In[157]:


df = pd.read_excel(r'C:\Users\k.samarth.ashvinbhai\Desktop\Work folder\Excel files\OPLT- datalatest_excel_op.xlsx')


# In[158]:


print(df.shape)
print(df.isnull().sum(axis = 0))
df=df.dropna()#1 nan row is dropped
print(df.shape)
df = df.drop_duplicates(keep=False)
print(df.shape)


# In[159]:


df = df.drop(columns=['Sales Doc.','BUn','Delivery','Mat. Doc.'])
df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])
df['Created on'] = pd.to_datetime(df['Created on'])


# In[160]:


dff=df
N_df=df


# In[161]:


dff['SaTy']=dff['SaTy'].map({'ZPRT':0, 'ZROP':1, 'ZDSP':2, 'ZORM':3, 'ZREP':4, 'ZORU':5, 'ZRDM':6, 'ZOSF':7,'ZMOT':8})
dff['SOrg.']=dff['SOrg.'].map({3000:0, 3020:1})
dff['DChl']=dff['DChl'].map({'RT':0, 'WH':1})
dff['Dv']=dff['Dv'].map({'MI':0, 'CO':1, 'EN':2, 'DI':3, 'PM':4})
dff['Sold-to pt']=dff['Sold-to pt'].map({'CUSTOMER_1':0, 'CUSTOMER_2':1, 'CUSTOMER_3':2})
dff['Material']=dff['Material'].map({'PRODUCT_A':0, 'PRODUCT_B':1, 'PRODUCT_C':2, 'PRODUCT_D':3, "PRODUCT_E":4})
dff['StorageLoc']=dff['StorageLoc'].map({'0001':0, '0002':1, '0100':2, 'MQSO':3, 'MQSU':4, 'MQSI':5, 'MTPI':6, 'MQAN':7})


# In[162]:


dff['Creation day']=dff['Creation day'].map({'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Monday':1, 'Tuesday':2, 'Sunday':0})
dff['Posting day']=dff['Posting day'].map({'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Monday':1, 'Tuesday':2, 'Sunday':0})
dff['Route']=dff['Route'].map({'TRUCK':0, 'AIR':1})
dff['Plant/Sourceloc']=dff['Plant/Sourceloc'].map({'00F4':0, '0026':1, '0037':2})
dff['stock availability']=dff['stock availability'].map({'YES':0, 'NO':1})
dff['credit block/Other block']=dff['credit block/Other block'].map({'YES':0, 'NO':1})


# In[163]:


dff.dtypes


# In[164]:


dff['Created on']=dff['Created on'].dt.date
dff['Pstng Date']=dff['Pstng Date'].dt.date
dff['Delta'] = dff['Pstng Date'] - df['Created on']
dff['Delta']=dff['Delta'].dt.days


# In[ ]:





# In[165]:


dff.describe()


# In[166]:


plt.scatter(dff['Delta'], df['Creation day'])
plt.ylabel("Creation Days")
plt.show()
plt.scatter(dff['Delta'], df['Sold-to pt'])
plt.ylabel("Customers")
plt.show()
plt.scatter(dff['Delta'], df['Material'])
plt.ylabel("Products")
plt.show()
plt.scatter(dff['Sold-to pt'], df['Material'])
plt.ylabel("Products")
plt.show()


# In[167]:


source_list = list(dff['Plant/Sourceloc'].unique())
source_dict={}
for source in source_list:
    source_dict[source] = len(df[df['Plant/Sourceloc']==source])
    
plt.bar(list(source_dict.keys()),  list(source_dict.values()))
print(source_dict, "{'00F4':0, '0026':1, '0037':2}")
plt.xlabel("Plant/Sourceloc")
plt.show()

product_list = list(dff['Posting day'].unique())
prod_dict={}
for prod in product_list:
    prod_dict[prod] = len(df[df['Posting day']==prod])
    
plt.bar(list(prod_dict.keys()),  list(prod_dict.values()))
plt.xlabel("Posting day")
plt.show()


created_list = list(dff['Creation day'].unique())
cred_dict={}
for cred in created_list:
    cred_dict[cred] = len(df[df['Creation day']==cred])
    
plt.bar(list(cred_dict.keys()),  list(cred_dict.values()))
plt.xlabel("Creation Day")
plt.show()


route_list= list(dff['Route'].unique())
route_dict={}
for rout in route_list:
    route_dict[rout] = len(df[df['Route']==rout])
    
plt.bar(list(route_dict.keys()),  list(route_dict.values()))
plt.xlabel("Route-Truck/Air")
print(route_dict, "{'TRUCK':0, 'AIR':1}")
plt.show()


# In[168]:


product_list = list(dff['Sold-to pt'].unique())
prod_dict={}
for prod in product_list:
    prod_dict[prod] = len(dff[dff['Sold-to pt']==prod])
    
plt.bar(list(prod_dict.keys()),  list(prod_dict.values()))
plt.xlabel("Customers")
plt.show()

product_list = list(dff['Material'].unique())
prod_dict={}
for prod in product_list:
    prod_dict[prod] = len(dff[dff['Material']==prod])
    
plt.bar(list(prod_dict.keys()),  list(prod_dict.values()))
plt.xlabel("Products")
plt.show()


# In[169]:


plt.hist(dff['Delta'])
plt.show() 


# In[170]:


print(dff.shape)
q_low = dff["Delta"].quantile(0.00)
q_hi  = dff["Delta"].quantile(0.90)

df_filtered = dff[(df["Delta"] < q_hi) & (dff["Delta"] > q_low)]
print(df_filtered.shape)


# In[171]:


plt.hist(df_filtered['Delta'])
plt.show() 


# In[172]:


product_list = list(df_filtered['Sold-to pt'].unique())
prod_dict={}
for prod in product_list:
    prod_dict[prod] = len(df_filtered[df_filtered['Sold-to pt']==prod])
    
plt.bar(list(prod_dict.keys()),  list(prod_dict.values()))
plt.xlabel("Customers")
plt.show()

product_list = list(df_filtered['Material'].unique())
prod_dict={}
for prod in product_list:
    prod_dict[prod] = len(df_filtered[df_filtered['Material']==prod])
    
plt.bar(list(prod_dict.keys()),  list(prod_dict.values()))
plt.xlabel("Products")
plt.show()


# In[173]:


df_filtered.columns


# In[174]:


df_filtered.describe()


# In[175]:


df_filtered.corr()


# In[176]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[177]:


columns=['SaTy', 'SOrg.', 'DChl', 'Dv', 'Sold-to pt', 'Item', 'Material',
       'Creation day', 'Quantity', 'Plant/Sourceloc','StorageLoc',
        'Route', 'Posting day',
       'stock availability', 'credit block/Other block', 'Delta']
features = df_filtered[columns]

df_scaled = pd.DataFrame(scaler.fit_transform(features), columns = columns)


# In[178]:


out=df_scaled['Delta']
df_scaled=df_scaled.drop(columns=['Delta'])


# In[179]:


X_train, X_test, y_train, y_test = train_test_split(df_scaled, out, test_size=0.30)


# In[180]:


regr = RandomForestRegressor(max_depth=20, n_estimators= 100, random_state=0)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)


# In[181]:


r2_score(y_test, y_pred)


# In[182]:


mean_squared_error(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[183]:


N_df=df


# In[184]:


N_df.columns


# In[185]:


#Norm_df['Created on']=Norm_df['Created on'].dt.date
#Norm_df['Pstng Date']=Norm_df['Pstng Date'].dt.date
N_df['Delta'] = N_df['Pstng Date'] - N_df['Created on']
N_df['Delta']=N_df['Delta'].dt.days


# In[186]:


print(N_df.shape)
q_low = N_df["Delta"].quantile(0.00)
q_hi  = N_df["Delta"].quantile(0.90)

N_df = N_df[(N_df["Delta"] < q_hi) & (N_df["Delta"] > q_low)]
print(N_df.shape)


# In[187]:


Norm_df=N_df


# In[188]:



#Normalisation
cols_to_norm = ['Item', 'Quantity','Delta']
N_df[cols_to_norm] = N_df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#categorical data
categorical_cols = ['SaTy', 'SOrg.', 'DChl', 'Dv', 'Sold-to pt', 'Material',
     'Creation day', 'Plant/Sourceloc',
       'StorageLoc', 'Route', 'Posting day',
       'stock availability', 'credit block/Other block'] 
print(N_df.columns)
#import pandas as pd
One_hot_df = pd.get_dummies(N_df, columns = categorical_cols)


# In[189]:


#Using label Encoder
Norm_df=N_df
cols_to = ['Item', 'Quantity','Delta']
categorical_cols = ['SaTy', 'SOrg.', 'DChl', 'Dv', 'Sold-to pt', 'Material',
     'Creation day', 'Plant/Sourceloc',
       'StorageLoc', 'Route', 'Posting day',
       'stock availability', 'credit block/Other block'] 
labelencoder = LabelEncoder()

New_Norm_df = Norm_df[categorical_cols]
New_Norm_df1=New_Norm_df.apply(LabelEncoder().fit_transform)

onehotencoder = OneHotEncoder()
transformed_data = onehotencoder.fit_transform(New_Norm_df1).toarray()
encoded_data = pd.DataFrame(transformed_data, index=New_Norm_df1.index)
concatenated_data = pd.concat([Norm_df[['Item', 'Quantity','Delta']], encoded_data], axis=1)


# In[190]:


concatenated_data


# In[191]:


output=One_hot_df['Delta']
One_hot_df=One_hot_df.drop(columns=['Delta','Created on', 'Pstng Date'])


# In[192]:


regr = RandomForestRegressor(max_depth=100, n_estimators= 100, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(One_hot_df, output, test_size=0.30)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)


# In[193]:


r2_score(y_test, y_pred)


# In[194]:


mean_squared_error(y_test, y_pred)


# In[ ]:





# In[195]:


#Using label Encoder and One hot Encoded Data
output=concatenated_data['Delta']
concatenated_data_drop=concatenated_data.drop(columns=['Delta'])
regr = RandomForestRegressor(max_depth=100, n_estimators= 100, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(concatenated_data_drop, output, test_size=0.30)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
print(r2_score(y_test, y_pred))
mean_squared_error(y_test, y_pred)


# In[205]:


#with open('Random_For_Model', 'wb') as f:
#    cPickle.dump(regr, f)

import joblib

joblib.dump(regr, 'model.pkl')


# In[209]:


#with open('Random_For_Model', 'rb') as f:
#   regr = cPickle.load(f)

lr = joblib.load('model.pkl')

preds = lr.predict(X_test[:1])
print(preds)
xtest_json=X_test[:1].to_json()
joblib.dump(xtest_json,'data.pkl')
#for i in X_test.index:
#    df.loc[i].to_json("row{}.json".format(i))

test_x=joblib.load('data.pkl')



# In[198]:


df_pred = pd.DataFrame(data =preds,columns=['Prediction'])


# In[199]:


maxi=df_pred['Prediction'].max()
mini=df_pred['Prediction'].min()


# In[200]:


maxi,mini


# In[201]:


df_pred['Prediction1'] = df_pred['Prediction'].apply(lambda x: (x-mini) / (maxi -mini))


# In[202]:


df_pred['Prediction'][1:20], y_test[1:20]

