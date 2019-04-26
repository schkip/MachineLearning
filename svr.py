#svr

#import the libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# feature scaling is required
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = np.array(x).reshape(-1,1) #notice that this is not in the tutorial and is required to reshape to a 2D array
y = np.array(y).reshape(-1,1)
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


#fitting the SVR to the dataset
#create your regressor here
from sklearn.svm import SVR #class
regressor = SVR(kernel = 'rbf') #object
regressor.fit(x, y)

#predict new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))


#visualise
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('truth or bluff SVR')
plt.xlabel('position level')
plt.ylabel('Salary $')
plt.show()
