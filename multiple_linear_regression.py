#MULTIPLE LINEAR REGRESSION

#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#avoiding the dummy variable trap
x = x[:, 1:] #removes one of the dummy variables to ensure information is retained


#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#feature scaling - not required for multiple linear regression - handled in library
"""from sklearn.preprocessing import StandardScaler
sc_x = Standard Scaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression #import class
regressor = LinearRegression() #create object of the class LinearRegression
regressor.fit(x_train, y_train) #fits to the training model


#predicting the test set results
y_pred = regressor.predict(x_test) #use the predictor method to predict the observations of the test set

import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis =1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog =x_opt).fit() # calling from the OLS class, specify the arguments need some parameters exog and endog
regressor_OLS.summary()

#x_opt_next = x[:, [0, 1, 4]]
#regressor_OLS = sm.OLS(endog = y, exog =x_opt_next).fit() # calling from the OLS class, specify the arguments need some parameters exog and endog
#regressor_OLS.summary()

#y_pred_new = regressor.predict(x_opt_next)

import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)

x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt_next = x_opt[:, [0, 1, 4]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt_next).fit()
regressor_OLS.summary()

x_opt_next = x_opt[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt_next).fit()
regressor_OLS.summary()
