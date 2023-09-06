import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier


breast_cancer_data=load_breast_cancer()                         #this loads data in the form of a dictionary.
print(breast_cancer_data.keys())                            # Keys:
#print(breast_cancer_data.DESCR)                             # 1. data:- refers to all the attributes data present that helps to train the classifier.
                                                            # 2. target:- variable you wanna predict/label
                                                            # 3. feature_names:- these are names of the columns in data
                                                            # 4. target_names:- name of the target column(s)
                                                            # 5. DESCR:- provides a short description of the dataset
                                                            # 6. filename:- path to the actual file of the data in CSV format


#print(breast_cancer_data.feature_names)
#print(breast_cancer_data.target,breast_cancer_data.target_names)

# now we create a data frame using the above data
breast_cancer_df=pd.DataFrame(breast_cancer_data.data,columns=breast_cancer_data.feature_names)
breast_cancer_df['target']=breast_cancer_data.target

#split data into train and test sets
training_data,validation_data,training_labels,validation_labels=train_test_split(breast_cancer_data.data,breast_cancer_data.target,test_size=0.2,random_state=100)
print(len(training_data),len(training_labels))


#create  a KNN model with k=3
classifier=KNeighborsClassifier(n_neighbors=3)
#train model
classifier.fit(training_data,training_labels)

#check accuracy of validation data
print(classifier.score(validation_data,validation_labels))
print(len(breast_cancer_df))

#now we check for optimal k value such that the accuracy of the classifier is maximum
all_k={}
for k in range(1,101):
    classifier=KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data,training_labels)
    all_k[k]=classifier.score(validation_data,validation_labels)

max_acc=-1
for k in all_k:
    if all_k[k]>max_acc:
        max_acc=all_k[k]
        ideal_k=k

print(max_acc,ideal_k)

#display k v/s classifier-accuracy plot
plt.plot(all_k.keys(),all_k.values())
plt.title("Breast Cancer Classifier Accuracy")
plt.xlabel("k(number of nearest neighbors to consider)")
plt.ylabel("classifier accuracy")
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()
plt.close()




