# First analysis
import pandas as pd
# Read customer data
customer_data = pd.read_csv('Cust_List.csv')
#drop columns in customer data with few input
customer_data.drop('Mst Toplam Gelir', axis=1, inplace=True)
customer_data.drop('Cocuksayisi', axis=1, inplace=True)
customer_data.drop('Dogum Tarihi', axis = 1, inplace=True)

# Read customer product used data
customer_product_data = pd.read_csv('Flag_List.csv')
#drop columns with customer product information that will not be used in the analysis
customer_product_data.drop(customer_product_data.columns[[1,3,5,7,9,10,11,12,13,14,15,16,17,18]], axis=1, inplace=True)

#merge two data
customer_data = pd.merge(customer_data,customer_product_data)

#describe attributes
customer_data.describe()    

#correlation among attributes
customer_data.corr()

#Convert categorical variables into binary
customer_data = pd.get_dummies(customer_data)

#second stage - splitting train and test data
import numpy as np
prediction_values = np.array(customer_data['Kredi Kartı'])
customer_data= customer_data.drop('Kredi Kartı', axis = 1)
# Saving feature names for later use
customer_data_column_list = list(customer_data.columns)
customer_data_array = np.array(customer_data)

# splitting training and testing data set.
from sklearn.model_selection import train_test_split
train_customer_data, test_customer_data, train_prediction_values, test_prediction_values = train_test_split(customer_data_array, prediction_values, test_size = 0.2, random_state = 42)  
 
# splitting training and testing data set with Cross Validation
from sklearn.cross_validation import train_test_split
train_customer_data, test_customer_data, train_prediction_values, test_prediction_values = train_test_split(customer_data_array, prediction_values, test_size = 0.2, random_state = 42)  


#third stage - prediction
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 100 decision trees
rf = RandomForestClassifier(n_estimators = 100, random_state = 42,criterion="entropy")
# Train the model on training data
rf.fit(train_customer_data, train_prediction_values);

predictions = rf.predict(test_customer_data)
# Calculate the  errors
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


#Variable Importances
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(customer_data_column_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


