import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#THIS PROJECT DEALS WITH DETECTION OF FRAUDULENT CREDIT CARD TRANSACTIONS OVER A SIMULATED SET OF TRANSACTIONS

transactions=pd.read_csv('trans.csv')
t_df=transactions[['type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
       'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']]

#we create a column "IS PAYMENT?" that denotes transaction out
t_df['isPayment']=t_df.type.apply(lambda x:1 if x=='PAYMENT' or x=='DEBIT' else 0)
t_df['isMovement']=t_df.type.apply(lambda x:1 if x=='CASH_OUT' or x=='TRANSFER' else 0)


#here theory is that existence of large difference in origin and destination account if considered as fraudulent
t_df['account_diff']=abs(transactions['oldbalanceOrg']-transactions['oldbalanceDest'])

#now we perform logistic regression
#feature creation
features=t_df[['amount','isPayment','isMovement','account_diff']]
label=t_df['isFraud']

#now we scale the features above since they are of different nature

#The StandardScaler function of sklearn is based on the theory that the dataset's variables ...
# whose values lie in different ranges do not have an equal contribution to the model's fit parameters and ....
# training function and may even lead to bias in the predictions made with that model.
scale=StandardScaler()
scale.fit(features)                             #here we obtain mean & std_dev
features=scale.transform(features)

#here we break into test & train samples
features_train,features_test,label_train,label_test=train_test_split(features,label,test_size=0.3)

log_r=LogisticRegression()
log_r.fit(features_train,label_train)       ##train model
label_pred=log_r.predict(features_test)

# display the R_sq coefficient for train set
print("Train accuracy is: ",log_r.score(features_train,label_train))
print("Test accuracy is: ",log_r.score(features_test,label_test))
print(log_r.coef_)

#predict unknown data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

sample_transactions=[transaction1,transaction2,transaction3]
sample_transactions=scale.fit_transform(sample_transactions)

print(log_r.predict(sample_transactions))
print(log_r.predict_proba(sample_transactions))

#display confusion matrix
print(confusion_matrix(label_test,label_pred))

#display accuracy,F1_score,precision and recall
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print("ACCUARCY IS: ",accuracy_score(label_test,label_pred))
print("PRECISION IS: ",precision_score(label_test,label_pred))
print("RECALL IS: ",recall_score(label_test,label_pred))
print("F1_SCORE IS: ",f1_score(label_test,label_pred))


