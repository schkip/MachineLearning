#polynomial regression

#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#split into training and test set
#make the most accurate model therefore the training set =100% of the dataset

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#fitting linear regression model to the dataset as a reference
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)   #fit linreg object to x and y

#fitting a polynomial regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures  #class
poly_reg = PolynomialFeatures(degree = 4) #object of this class
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#visualising the linear regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff - Linear Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visualising the polynomial regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff - Poly Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




