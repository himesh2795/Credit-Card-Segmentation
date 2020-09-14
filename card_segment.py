# import libraries

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch

# Load data
card_df = pd.read_csv("credit-card-data.csv")
total_rows = len(card_df)

# No need of CUST_ID so remove it
card_df.drop('CUST_ID',axis=1, inplace=True)


####  Missing value analysis ####
missing_val = pd.DataFrame(card_df.isnull().sum(), columns = ['Count'])
missing_val['Percentage'] = (missing_val.loc[:,'Count']*100)/len(card_df)

# Remove missing values
card_df.drop((card_df[card_df['CREDIT_LIMIT'].isnull() | card_df['MINIMUM_PAYMENTS'].isnull()]).index, inplace=True)
print(f"DataLoss After removing missing values : {total_rows-len(card_df)}")

card_df.info()

##################  Data Analysis ####################
############ 1. Deriving key performance indicators(KPI)  ##############
# 1.1 - monthly average purchase
# 1.2 - monthly cash advance amount
# 1.3 - purchases by type (one-off, instalments)
# 1.4 - average amount per purchase
# 1.5 - cash advance transaction
# 1.6 - limit usage (balance to credit limit ratio)
# 1.7 - payments to minimum payments ratio 

## 1.1 - monthly average purchase
# PURCHASES Total purchase amount spent during last 12 months
# TENURE Number of months as a customer
card_df.TENURE.unique()
card_df_KPI = pd.DataFrame()
card_df_KPI['monthly_avg_purchase'] = card_df.PURCHASES / card_df.TENURE

## 1.2 - monthly cash advance amount
card_df_KPI['monthly_cash_advance'] = card_df.CASH_ADVANCE / card_df.TENURE

## 1.3 - monthly purchases by type (one-off, instalments)
card_df_KPI['monthly_oneoff_purchase'] = card_df.ONEOFF_PURCHASES / card_df.TENURE
card_df_KPI['monthly_installment_purchase'] = card_df.INSTALLMENTS_PURCHASES / card_df.TENURE

## 1.4 - average amount per purchase
card_df.PURCHASES.mean()

## 1.5 - average cash advance transaction
card_df.CASH_ADVANCE.mean()

## 1.6 - limit usage (balance to credit limit ratio)

## 1.7 - payments to minimum payments ratio 


######## K-Means Clustering ###########
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(card_df)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker="o")
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

####### Hierarchical Clustering ###########
dendrogram = sch.dendrogram(sch.linkage(card_df, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

