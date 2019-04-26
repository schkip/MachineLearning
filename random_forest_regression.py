#random forest regression

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting the random forest regression to the dataset
from sklearn.ensemble import RandomForestRegressor #class
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42) #object
regressor.fit(x, y) #method


# predicting a new value
y_pred = regressor.predict(np.array([[6.5]]))

#visualise the results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('a walk through the rnd forest')
plt.xlabel('Positions')
plt.ylabel('salary $')