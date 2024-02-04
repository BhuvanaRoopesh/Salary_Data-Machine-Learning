import pandas as pd
import numpy as np
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

corr_plot=sns.heatmap(dataset.corr(),annot=True)
plt.show()

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
regressor.fit(X_test, Y_test)

Y_pred_train=regressor.predict(X_train)
Y_pred_test=regressor.predict(X_test)

print('Intercept is',regressor.intercept_)
print('\n')
print('Coefficient is',regressor.coef_)
print('\n')

from sklearn import metrics
print('Evaluation metrics for test set')
print('Mean absolute error',metrics.mean_absolute_error(Y_test,Y_pred_test))
print('Mean square error',metrics.mean_squared_error(Y_test,Y_pred_test))
print('Root MeanSquare error',np.sqrt(metrics.mean_squared_error(Y_test,Y_pred_test)))
print('R2 Score/ Train data Accuracy', metrics.r2_score(Y_test, Y_pred_test))
print('\n')

from sklearn import metrics
print('Evaluation metrics for training set')
print('Mean absolute error',metrics.mean_absolute_error(Y_train,Y_pred_train))
print('Mean square error',metrics.mean_squared_error(Y_train,Y_pred_train))
print('Root MeanSquare error',np.sqrt(metrics.mean_squared_error(Y_train,Y_pred_train)))
print('R2 Score/ Train data Accuracy', metrics.r2_score(Y_train, Y_pred_train))
print('\n')

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test,Y_pred_test, color='cyan')
plt.title('Salary Vs Years of Experince (test set,size=0.2)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train,Y_pred_train, color='cyan')
plt.title('Salary Vs Years of Experince (training set,size=0.8)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

