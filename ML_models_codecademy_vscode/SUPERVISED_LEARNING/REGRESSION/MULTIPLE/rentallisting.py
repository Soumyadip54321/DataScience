import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D



#MANHATTAN dataset considered here.
man_df=pd.read_csv('ManhattanRentalListing.csv')

#here we select 'rent' as dependent variable and other columns as independent
x=man_df[['bedrooms', 'bathrooms', 'size_sqft',
       'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck',
       'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher',
       'has_patio', 'has_gym']]
y=man_df[['rent']]

#Now we split data
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=6)


#fitting data & model creating
mlr=LinearRegression()
mlr.fit(x_train,y_train)

print(mlr.coef_,mlr.intercept_)


#predict 'rent' from test data ,test model
y_predict=mlr.predict(x_test)
""" sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]  #This is what we need to predict the price for, an apartment with these specifications.


rent_predict=mlr.predict(sonny_apartment)
print(f"predicted rent is ${rent_predict[0][0]}") """

#display relationship between actual and predicted rents.
""" plt.scatter(y_test,y_predict,alpha=0.4)
plt.xlabel("actual rent")
plt.ylabel("predicted rent")
plt.title("Relationship between actual and predicted rent in $")
plt.plot()
plt.show()
plt.close() """

#display relationship b/w size and rent,min distance to subway & rent
""" plt.scatter(man_df['size_sqft'],man_df[['rent']],alpha=0.4)
plt.xlabel("size in ft2")
plt.ylabel("rent in $")
plt.title("Relationship between size and rent in $")
plt.show()
plt.close()

plt.scatter(man_df['min_to_subway'],man_df[['rent']],alpha=0.4)
plt.xlabel("minimum distance to subway in minutes")
plt.ylabel("rent in $")
plt.title("Relationship between minimum distance to subway and rent in $")
plt.show()
plt.close() """


#Now we compute the accuracy of our model

#here we compute the R2 coefficient that describes %variation as explained by all x-variables.

print("Train set accuracy score:")
print(mlr.score(x_train,y_train))

print("Test set accuracy score:")
print(mlr.score(x_test,y_test))

#displaying scatterplot of residuals/error v/s predicted y values
error=y_predict-y_test
""" plt.scatter(error,y_predict,alpha=0.4)
plt.title('Residual Analysis')
plt.xlabel("error")
plt.ylabel("rent predicted")
plt.show()
plt.close() """




