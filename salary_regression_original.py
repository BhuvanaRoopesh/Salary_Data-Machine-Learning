# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:32:43 2024

@author: ACER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('Salary_Data.csv')
print(dataset.head())
print(dataset.tail())
print(dataset.info())
print(dataset.describe())
print(dataset.isnull())
print(dataset.isnull().sum())
print(dataset.corr())

#plotting the correclation Heat Map
corr_plot=sns.heatmap(dataset.corr(),annot=True)

#Find Independent and Dependent Variables
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,:-1].values

#import Linear Regression and sklearn library
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,Y)

#predicting the Y values
Y_pred=regressor.predict(X)

#Genarating the results from the slope Equation
print('Intercept is',regressor.intercept_)
print('\n')
print('Coefficient is',regressor.coef_)
print('\n')

#Generating performance metrics
from sklearn import metrics
print('Evaluation of results for Original set')
print('Mean absolute error',metrics.mean_absolute_error(Y,Y_pred))
print('Mean squared error',metrics.mean_squared_error(Y,Y_pred))
print('Root mean squared error',np.sqrt(metrics.mean_squared_error(Y,Y_pred)))
print('R2 Score',metrics.r2_score(Y,Y_pred))

#Plotting the results in Graph
#Visualizing the original set results
plt.scatter(X,Y, color='blue')
plt.plot(X,Y_pred, color='cyan')
plt.title('Salary Vs Years of Experince (original set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()