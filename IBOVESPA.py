import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.dates as mdates

#Read CSV
df = pd.read_csv('IBOVESPA.csv')
df.set_index('Date', inplace=True)
df.dropna(inplace=True)

#Exploring the data
df.head()
df.tail()
df.describe()
df.info()

#Plotting the data
df.plot(subplots=True, figsize=(12, 8))
plt.show()

#Separating the data 80% training and 20% test
train = df.iloc[:int(df.shape[0]*0.8)]
test = df.iloc[int(df.shape[0]*0.8):]

#Separating the data in X and Y
train_X = train.drop('Close', axis=1)
train_Y = train['Close']
test_X = test.drop('Close', axis=1)
test_Y = test['Close']

#Training Linear Regression
linearModel = LinearRegression()
linearModel.fit(train_X, train_Y)

#Training Grid Search Polynomial Regression
model = make_pipeline(PolynomialFeatures(), LinearRegression())
param_grid = {'polynomialfeatures__degree': np.arange(2, 5)}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(train_X, train_Y)
polyModel = grid.best_estimator_

#Training Grid Search KNN Regression
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
param_grid = {'n_neighbors': np.arange(2, 5)}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(train_X, train_Y)
knnModel = grid.best_estimator_

#Predicting the data
linearPredict = linearModel.predict(test_X)
polyPredict = polyModel.predict(test_X)
knnPredict = knnModel.predict(test_X)

#Plotting the data
plt.figure(figsize=(12, 8))
plt.plot(test_Y, label='Real')
plt.plot(linearPredict, label='Linear')
plt.plot(polyPredict, label='Polynomial')
plt.plot(knnPredict, label='KNN')
plt.title('IBOVESPA')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()
plt.show()

#Calculating the error
from sklearn.metrics import mean_squared_error
linearError = mean_squared_error(test_Y, linearPredict)
polyError = mean_squared_error(test_Y, polyPredict)
knnError = mean_squared_error(test_Y, knnPredict)

#Printing the error
print('Linear Error: ', linearError)
print('Polynomial Error: ', polyError)
print('KNN Error: ', knnError)