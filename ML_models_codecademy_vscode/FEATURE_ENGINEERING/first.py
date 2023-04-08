import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# AT FIRST WE READ IN THE DATASET
coffee = pd.read_csv('starbucks_customers.csv')
#print(coffee.head())

#OBSERVE THE STATISTICAL DESCRIPTION OF THE DATASET
#print(coffee.info())

#   NOW WE FIGURE OUT AGE DISTRIBUTION OF CUSTOMERS WHO TOOK PART IN SURVEY
ages=coffee['age']
min_age=ages.min()
max_age=ages.max()
print("MAX AND MIN AGE OF CUSTOMERS IS: ",max_age,min_age)
print("RANGE OF AGE IS :",max_age-min_age)

centred_ages=ages-ages.mean()

#plot the distribution
#sns.histplot(ages,x=centred_ages)
#plt.title(f"DISTRIBUTION OF CUSTOMERS CENTRED AROUND MEAN AGE OF {int(ages.mean())} yrs.")
#plt.show()
#plt.close()

# NOW WE STANDARDIZE OUR AGE FEATURES SUCH THAT ALL OF THEM ARE ON THE SAME SCALE.
std_dev_age=ages.std()
ages_standardized=centred_ages*(1/std_dev_age)

#Let's check to see the std_dev & mean
#print(ages_standardized.mean(),ages_standardized.std())

#BELOW WE MAKE USE OF standard scaler from Sklearn library to perform the same task of standardizing
scaler = StandardScaler()
ages_reshaped=np.array(ages).reshape(-1,1)
ages_scaled=scaler.fit_transform(ages_reshaped)
#print(np.mean(ages_scaled),np.std(ages_scaled))

# WE NOW DEMONSTRATE USAGE OF MIN-MAX NORMALIZATION 
spent=coffee['spent']
max_spent=spent.max()
min_spent=spent.min()
spent_range=max_spent-min_spent
#print("DIFFERENCE OF AMOUNT EXPENDED ON PURCHASE BY CUSTOMERS :",spent_range)
spent_normalized=(spent-min_spent)/spent_range
#print(spent_normalized)

#NOW WE DEMONSTRATE THE ABOVE USING A MIN-MAX SCALER FROM Sklearn lib
mmscaler=MinMaxScaler()
spent_reshaped=np.array(spent).reshape(-1,1)
reshaped_scaled=mmscaler.fit_transform(spent_reshaped)
print(np.min(reshaped_scaled),np.max(reshaped_scaled))

#BINNING

#Let's first display the histogram of ages to know of the distribution.
plt.hist(ages)
plt.show()
plt.close()

#FROM THE PLOT WE SEE THE bins limit could be set as={12,20,30,40,71}
age_bins=[12,20,30,40,71]
coffee['binned_ages']=pd.cut(ages,age_bins,right=False)
print(coffee.binned_ages.head(10))

#SINCE THESE FALL INTO CATEGORICAL VARIABLE WE USE value_count() to count the respective number of bins present.
coffee.binned_ages.value_counts().plot(kind="bar")
plt.title("age distribution of customers visiting nearby STARBUCKS")
plt.show()
plt.close()




