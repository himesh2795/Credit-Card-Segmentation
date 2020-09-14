# import libraries

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import FactorAnalysis


# Load data
card_df = pd.read_csv("credit-card-data.csv")
total_rows = len(card_df)

# No need of CUST_ID so remove it
card_df.drop('CUST_ID',axis=1, inplace=True)


####  Missing value analysis ####
missing_val = pd.DataFrame(card_df.isnull().sum(), columns = ['Count'])
missing_val['Percentage'] = (missing_val.loc[:,'Count']*100)/len(card_df)

# Remove missing values
# card_df.drop((card_df[card_df['CREDIT_LIMIT'].isnull() | card_df['MINIMUM_PAYMENTS'].isnull()]).index, inplace=True)
card_df.dropna(inplace=True)
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


########## Dimensionality Reduction ##########
## Factor Analysis

# Bartlett’s test of sphericity checks whether or not the observed variables intercorrelate at all using the observed correlation matrix against the identity matrix. If the test found statistically insignificant, you should not employ a factor analysis
chi_square_value,p_value=calculate_bartlett_sphericity(card_df)
print(chi_square_value, p_value)

#In this Bartlett ’s test, the p-value is 0. The test was statistically significant, indicating that the observed correlation matrix is not an identity matrix.

# Kaiser-Meyer-Olkin (KMO) Test measures the suitability of data for factor analysis. It determines the adequacy for each observed variable and for the complete model. KMO estimates the proportion of variance among all the observed variable. Lower proportion id more suitable for factor analysis. KMO values range between 0 and 1. Value of KMO less than 0.6 is considered inadequate.
kmo_all,kmo_model=calculate_kmo(card_df)
# Here kmo_model value is 0.64, so it is adequate.

## Choosing the Number of Factors
# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.set_params(n_factors=25, rotation=None)
fa.fit(card_df)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

# Create scree plot using matplotlib
plt.scatter(range(1,card_df.shape[1]+1),ev)
plt.plot(range(1,card_df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# The scree plot method draws a straight line for each factor and its eigenvalues. Number eigenvalues greater than one considered as the number of factors.
# Here, you can see only for 5-factors eigenvalues are greater than one. It means we need to choose only 5 factors (or unobserved variables).
fa = FactorAnalyzer()
fa.set_params(n_factors=5, rotation="varimax")
fa.fit(card_df)

loadings = fa.loadings_





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

