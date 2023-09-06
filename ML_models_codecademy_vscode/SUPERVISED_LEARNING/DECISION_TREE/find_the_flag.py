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

cols=['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']

#create a dataframe
flag_df=pd.read_csv('flag_data.csv',names=cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']
""" print(flag_df.head()) """

#we print number of continents by landmass
""" print(flag_df.landmass.value_counts()) """

#we look at flags from Europe and Oceania only, hence we create a dataframe holding data pertaining to them only
europe_oceania_df=flag_df[flag_df['landmass'].isin([3,6])]
print(europe_oceania_df.head())
#we compute the average of each of the predictor variables for the above continents

""" print(europe_oceania_df.groupby(['landmass'])[var].mean()) """

#inspect variable types for each predictors
""" print(europe_oceania_df[var].dtypes) """

#we OHE the variables
data=pd.get_dummies(europe_oceania_df[var])

#we split data into train and test set
labels=europe_oceania_df['landmass']
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.4,random_state=1)

#below we tune the Decision Tree Classifier with a range of depths, appending into a list the accuracy of each
depth=range(1,21)
acc_depth=[]

for i in depth:
    dtree=DecisionTreeClassifier(max_depth=i,random_state=10)
    dtree.fit(x_train,y_train)
    acc_depth.append(dtree.score(x_test,y_test))


#we plot accuracy vs depth of Decision Tree models
plt.plot(depth,acc_depth,color='blue')
plt.plot(depth[acc_depth.index(max(acc_depth))],max(acc_depth),"r*")
plt.title(f"Graph highlighting depth vs accuracy of tree")
plt.xlabel("depth of tree")
plt.ylabel("accuracy achieved")
plt.show()
plt.close()

#now we display the tree at which highest accuracy is obtained
best_depth=depth[acc_depth.index(max(acc_depth))]
dtree=DecisionTreeClassifier(max_depth=best_depth,random_state=1)
dtree.fit(x_train,y_train)

plt.figure(figsize=(14,8))
tree.plot_tree(dtree,feature_names=x_train.columns,class_names=['Europe','Oceania'],filled=True)
plt.show()

#now we prune the Decision Tree
acc_pruned=[]
ccp=np.logspace(-3,0,num=20)
for i in ccp:
    dt_prune = DecisionTreeClassifier(random_state = 1, max_depth = best_depth, ccp_alpha=i)
    dt_prune.fit(x_train, y_train)
    acc_pruned.append(dt_prune.score(x_test, y_test))

#now we plot accuracy vs ccp
plt.plot(ccp,acc_pruned,color='red')
plt.plot(ccp[acc_pruned.index(max(acc_pruned))],max(acc_pruned),"b*")
plt.title(f"graph of ccp vs accuracy obtained")
plt.xlabel("ccp")
plt.ylabel("accuracy achieved")
plt.show()
plt.close()

#having obtained the ideal ccp and depth at which optimal accuracy is achieved we plot tree now
best_ccp=ccp[acc_pruned.index(max(acc_pruned))]
dtree_final=DecisionTreeClassifier(max_depth=best_depth,ccp_alpha=best_ccp,random_state=10)
dtree_final.fit(x_train,y_train)

accuracy_obtained=dtree_final.score(x_test,y_test)

#we display the tree here
plt.figure(figsize=(14,8))
tree.plot_tree(dtree_final,feature_names=x_train.columns,class_names=['Europe','Oceania'],filled=True)
plt.show()







