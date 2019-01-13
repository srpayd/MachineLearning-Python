#Second analysis after obtaining essential features
import pandas as pd
# Read customer data
customer_data = pd.read_excel('Mst_List_v2.xls')

# Read customer product used data
customer_product_data = pd.read_excel('Urn_List_v2.xls')

#merge two data
customer_data = pd.merge(customer_data,customer_product_data)

#Convert categorical variables into binary
customer_data = pd.get_dummies(customer_data)

#second stage - splitting train and test data
import numpy as np
prediction_values = np.array(customer_data['Kredi Kartı'])
customer_data= customer_data.drop('Kredi Kartı', axis = 1)
# Saving feature names for later use
customer_data_column_list = list(customer_data.columns)
customer_data_array = np.array(customer_data)

#Splitting training and testing data set
from sklearn.model_selection import train_test_split
train_customer_data, test_customer_data, train_prediction_values, test_prediction_values = train_test_split(customer_data_array, prediction_values, test_size = 0.2, random_state = 42)  

# splitting training and testing data set with Cross Validation
from sklearn.cross_validation import train_test_split
train_customer_data, test_customer_data, train_prediction_values, test_prediction_values = train_test_split(customer_data_array, prediction_values, test_size = 0.2, random_state = 42)  

#Building the model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42,criterion="entropy")
rf.fit(train_customer_data, train_prediction_values);
predictions = rf.predict(test_customer_data)
errors = sum(abs(predictions - test_prediction_values))
print(errors)
len(predictions)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(test_prediction_values, predictions)
print(cnf_matrix)

#Evaluating model performance 
from sklearn.metrics import accuracy_score
accuracy_score(test_prediction_values, predictions)  


importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(customer_data_column_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


#Lets calculate score using different metrics
from sklearn.model_selection import cross_val_score
cross_val_score(rf,train_customer_data,train_prediction_values,cv=5)

from sklearn.model_selection import KFold
kf=KFold(n_splits=5,shuffle=True)
scores=cross_val_score(rf,train_customer_data,train_prediction_values,cv=kf);scores

#Finally, evaluating meaningfulness of the model.
from sklearn.metrics import classification_report
 print(classification_report(test_prediction_values,predictions))
