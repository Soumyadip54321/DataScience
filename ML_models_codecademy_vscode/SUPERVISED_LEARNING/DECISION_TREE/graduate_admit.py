import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeRegressor

graduate_df=pd.read_csv('ADMIT_PREDICT.csv')
""" print(graduate_df.columns) """
#we change the columns such that they do not contains spaces to help in TAB completion
graduate_df=graduate_df.rename(columns={
    'Serial No.':'Serial_no',
    'GRE Score':'GRE_score',
    'TOEFL Score':'TOEFL_score',
    'University Rating':'University_rating',
    'SOP':'SOP',
    'LOR ':'LOR',
    'CGPA':'CGPA',
    'Research':'Research',
    'Chance of Admit ':'Admit_chance'
})
""" print(graduate_df.head()) """

#we use a DECISION TREE with all cols except Admit_chance as predictors
X=graduate_df.iloc[:,1:8]
y=graduate_df.Admit_chance>=0.8

#split data & train model
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
dtree=DecisionTreeClassifier(max_depth=2,ccp_alpha=0.01,criterion='gini')
dtree.fit(x_train,y_train)

#predict unknown data
y_pred=dtree.predict(x_test)

#compute accuracy of prediction w.r.t validation set
print(dtree.score(x_test,y_test))
print(accuracy_score(y_test,y_pred))

#visualize tree
""" tree.plot_tree(dtree,feature_names=x_train.columns,class_names=['unlikely admit', 'likley admit'],filled=True)
plt.show()
plt.close() """

#we check how did the model chose 8.845 as the threshold value of "cgpa" to perform the first split

def gini(y_train):                                          #Computes the gini impurity score
    data=pd.Series(y_train)
    return 1-sum(data.value_counts(normalize=True)**2)

gi=gini(y_train)
print(f'Gini impurity at root: {round(gi,3)}')

def info_gain(left, right, current_impurity):
    """Information Gain associated with creating a node/split data.
    Input: left, right are data in left branch, right banch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_impurity - w * gini(left) - (1 - w) * gini(right)


#here we go through every continuous value of "CGPA" to determine how did the model select 8.845 as the threshold
info_gain_list = []
for i in x_train.CGPA.unique():
    left = y_train[x_train.CGPA<=i]
    right = y_train[x_train.CGPA>i]
    info_gain_list.append([i, info_gain(left, right, gi)])
 
ig_table = pd.DataFrame(info_gain_list, columns=['split_value', 'info_gain']).sort_values('info_gain',ascending=False)
print(ig_table.head(10))

#visualize split vs info gain for CGPA
plt.plot(ig_table['split_value'], ig_table['info_gain'],'o')
plt.plot(ig_table['split_value'].iloc[0], ig_table['info_gain'].iloc[0],'r*')
plt.xlabel('cgpa split value')
plt.ylabel('info gain')
plt.show()
plt.close()

#now for regression instead of GINI IMPURITY we compute the MSE(Mean Squared Error) such that there exists the max info gain in regard to MSE to decode the 
#feature to split at at first which happens to be "CGPA" again and at 8.845 which we verify below.

#here at each level the value is the average "Admit_chance" of all samples at that level which satisfies the logical criteria

X=graduate_df.iloc[:,1:8]
y = graduate_df['Admit_chance']

#predict unknown data
y_pred=dtree.predict(x_test)

#compute accuracy of prediction w.r.t validation set
print(dtree.score(x_test,y_test))
print(accuracy_score(y_test,y_pred))

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
dtree_reg=DecisionTreeRegressor(max_depth=2,ccp_alpha=0.001)
dtree_reg.fit(x_train,y_train)


def mse(data):
    """Calculate the MSE of a data set
    """
    return np.mean((data - data.mean())**2)
 
def mse_gain(left, right, current_mse):
    """Information Gain (MSE) associated with creating a node/split data based on MSE.
    Input: left, right are data in left branch, right banch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_mse - w * mse(left) - (1 - w) * mse(right)
 
m = mse(y_train)
print(f'MSE at root: {round(m,3)}')
 
mse_gain_list = []
for i in x_train.CGPA.unique():
    left = y_train[x_train.CGPA<=i]
    right = y_train[x_train.CGPA>i]
    mse_gain_list.append([i, mse_gain(left, right, m)])
 
mse_table = pd.DataFrame(mse_gain_list,columns=['split_value', 'info_gain_wrt_mse']).sort_values('info_gain_wrt_mse',ascending=False)
print(mse_table.head(10))

#visualize split vs info gain for CGPA
plt.plot(mse_table['split_value'], mse_table['info_gain_wrt_mse'],'o')
plt.plot(mse_table['split_value'].iloc[0], mse_table['info_gain_wrt_mse'].iloc[0],'r*')
plt.xlabel('cgpa split value')
plt.ylabel('info gain w.r.t mse')
plt.show()
plt.close()





