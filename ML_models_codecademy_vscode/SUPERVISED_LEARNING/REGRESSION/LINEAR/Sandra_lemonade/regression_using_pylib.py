from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np

#---DATASET CREATED OF TEMPERATURES VS SOUP SALES-------#
temperature = np.array(range(60, 100, 2))       
temperature = temperature.reshape(-1, 1)
sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]

#--PLOTTING THE TREND ABOVE----#
plt.plot(temperature,sales,'x')
plt.title("relationship between temperature and sales of soup")
plt.xlabel("temperature")
plt.ylabel("sales")


#---setting an instance of Linear Regrresion model--#
line_fitter=LinearRegression()
line_fitter.fit(temperature,sales)          #here we don't furnish LEARNING RATE and ITERATIONS like we did for our without library usage case and gives the model
                                            #2 variables b and m that we require to fit in best.


#--Now we create a list of---#
sales_predict=line_fitter.predict(temperature)
plt.plot(temperature,sales_predict)
plt.show()
plt.close()

