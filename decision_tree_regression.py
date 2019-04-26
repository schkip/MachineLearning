#decision tree regression

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#fit the decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor #class
regressor = DecisionTreeRegressor(random_state = 0) #object
regressor.fit(x, y) #method


#predict
y_pred = regressor.predict(np.array([[6.5]]))

#visualise (non continuous model, need high resoltion)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or bluff - decision tree regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

