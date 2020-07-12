#!/usr/bin/env python
# coding: utf-8

# In[363]:


import numpy as np
import sklearn as sk
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.random import sample_without_replacement
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from IPython.display import Image
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


# In[364]:


data = datasets.load_iris()
iris = pd.DataFrame(np.column_stack((data.data, data.target)), columns = data.feature_names + ["target"])
iris["target"] = iris["target"].astype(int)


# In[365]:


desc = iris.describe()
desc.to_csv("describtion.csv")


# From the matrix we can see the basic statistical propertey of each column.

# In[366]:


def data_cleansing(data, fill = None):
    # handle with empty value
    print("Check If the column contain Nan value")
    print(data.isna().any())
    if not fill is None:
        print("Replace all empty values with {}".format(fill))
        data.fillna()
    else:
        print("Drop all rows that contains empty values")
        data.dropna()
        
    # handle with outliers 
    # detect outliers in each column and remove them 
    print("Detecting Outliers")
    indexs = []
    for col in data.columns[:-1]:
        IR = desc[col]["75%"] - desc[col]["25%"]
        ifOutlier = lambda x: x < desc[col]["25%"] - 1.5 * IR or x > desc[col]["75%"] + 1.5 * IR
        index = data[col][data[col].apply(ifOutlier)].index
        indexs.extend(index)
    
    # now index is the list the rows that contain outlier. Remove them
    
    print("Rows with outlier: ", indexs)
    data.drop(indexs)
    data.reset_index(drop = True)

data = data_cleansing(iris)


# This function will remove the columns with empty value, or fill empty with value provided.
# For iris data set, there is no missing value.
# 
# 
# This function will detect outliers in each column using quantile method, and remove the rows with outlier

# In[367]:


def statistical_relationship(data):
    
    # Correlation
    corr = data.corr()
    corr.to_csv("correlation.csv")
    
    # LR
    lr = LogisticRegression()
    rfe = RFE(lr, 1)
    fit = rfe.fit(data[data.columns[:-1]], iris["target"])
    print(fit.ranking_)
    
    return sns.heatmap(corr)


# The funcion finds the relationship between features by finding the correlation matrix, output as csv, and visualize the result using heat map
# 
# The function use logistic regression to check the significance order of features.

# In[368]:


statistical_relationship(iris)


# We can see that sepal length is not correlated with other features and targets. However, the rest three features have strong corelation between each other.
# 
# From the rank list of four features, we can see that petal width (cm) is the most significant to target, followed by sepal width, petal length, sepal length. 

# In[369]:


random.seed(123)

irisc = deepcopy(iris)

index = sample_without_replacement(irisc.shape[0], 50)
irisc.loc[:,iris.columns[:-1]] = normalize(irisc[irisc.columns[:-1]])
test = irisc.iloc[index]
test.reset_index(drop = True,inplace = True)
train = irisc[~irisc.index.isin(index)]
train.reset_index(drop = True,inplace = True)


result_test = iris.iloc[index]
result_train = iris.iloc[~iris.index.isin(index)]


## Random Forest

rdf = RandomForestClassifier()
rdf.fit(train[train.columns[:-1]],train["target"])
print(rdf.score(test[test.columns[:-1]],test["target"]))
result_test["Random Forest"] = rdf.predict(test[test.columns[:-1]])
result_train["Random Forest"] = rdf.predict(train[train.columns[:-1]])

export_graphviz(rdf.estimators_[9], out_file = 'tree.dot')
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

## Nueral Network

nn = MLPClassifier(activation = "relu", 
                   solver = "adam",
                  learning_rate_init = 0.01)
nn.fit(train[train.columns[:-1]],train["target"])
print(nn.score(test[test.columns[:-1]],test["target"]))

result_test["Neural Network"] = nn.predict(test[test.columns[:-1]])
result_train["Neural Network"] = nn.predict(train[train.columns[:-1]])
results = pd.concat([result_train, result_test])
results.to_csv("results.csv")
#normalize(iris[])


# We use Random Forest and Neural Network to classify the iris data. we can see the accuracy is above 95% for both classifier.

# In[370]:


# visualize the 10th tree of the random forest

Image(filename = 'tree.png')


# The 10th tree of the random forest is visualized here. we can see that if a data entry is [0.4, 0.6, 0.7, 0.8], is will be cassified as class 3.
