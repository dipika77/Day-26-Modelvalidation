#Model validation in python

#instructions
'''Using X_train and X_test as input data, create arrays of predictions using model.predict().
Calculate model accuracy on both data the model has seen and data the model has not seen before.
Use the print statements to print the seen and unseen data.'''

# The model is fit using X_train and y_train
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true=y_train, y_pred= train_predictions)
test_error = mae(y_true=y_test, y_pred= test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))


#Regression models
#instructions
'''Add a parameter to rfr so that the number of trees built is 100 and the maximum depth of these trees is 6.
Make sure the model is reproducible by adding a random state of 1111.
Use the .fit() method to train the random forest regression model with X_train as the input data and y_train as the response.'''

# Set the number of trees
rfr.n_estimators = 100

# Add a maximum depth
rfr.maximum_depth = 6

# Set the random state
rfr.random_state = 1111

# Fit the model
rfr.fit(X_train, y_train)


#instructions
'''Loop through the feature importance output of rfr.
Print the column names of X_train and the importance score for that column.'''

# Fit the model using X and y
rfr.fit(X_train, y_train)

# Print how important each column is to the model
for i, item in enumerate(rfr.feature_importances_):
      # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))
    
    
    
#Classification models
#instructions
'''Create two arrays of predictions. One for the classification values and one for the predicted probabilities.
Use the .value_counts() method for a pandas Series to print the number of observations that were assigned to each class.
Print the first observation of probability_predictions to see how the probabilities are structured.'''

# Fit the rfc model. 
rfc.fit(X_train, y_train)

# Create arrays of predictions
classification_predictions = rfc.predict(X_test)
probability_predictions = rfc.predict_proba(X_test)

# Print out count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# Print the first value from probability_predictions
print('The first predicted probabilities are: {}'.format(probability_predictions[0]))



#instructions
'''Print out the characteristics of the model rfc by simply printing the model.
Print just the random state of the model.
Print the dictionary of model parameters.'''

rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Print the classification model
print(rfc)

# Print the classification model's random state parameter
print('The random state is: {}'.format(rfc.random_state))

# Print all parameters
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))


#instructions
'''Create rfc using the scikit-learn implementation of random forest classifiers and set a random state of 1111.
Fit rfc using X_train for the training data and y_train for the responses.
Predict the class values for X_test.
Use the method .score() to print an accuracy metric for X_test given the actual values y_test.'''

from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Fit rfc using X_train and y_train
rfc.fit(X_train, y_train)

# Create predictions on X_test
predictions = rfc.predict(X_test)
print(predictions[0:5])

# Print model accuracy using score() and the testing data
print(rfc.score(X_test,y_test))



#Creating train, test, and validation sets
#instructions
'''Create the X dataset by creating dummy variables for all of the categorical columns.
Split X and y into train (X_train, y_train) and test (X_test, y_test) datasets.
Split the datasets using 10% for testing'''

# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:,0:9])
y = tic_tac_toe.iloc[:, 9]

# Create training and testing datasets. Use 10% for the test set
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.10, random_state=1111)


#instructions
'''Create temporary datasets and testing datasets (X_test, y_test). Use 20% of the overall data for the testing datasets.
Using the temporary datasets (X_temp, y_temp), create training (X_train, y_train) and validation (X_val, y_val) datasets.
Use 25% of the temporary data for the validation datasets.'''

# Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test  =\
    train_test_split(X, y, test_size = 0.20, random_state=1111)

# Create the final training and validation datasets
X_train, X_val,y_train,y_val =\
    train_test_split(X_temp, y_temp, test_size= 0.25, random_state=1111)
    
    
#Accuracy metrices: regression models
#instructions
'''Manually calculate the MAE using n as the number of observations predicted.
Calculate the MAE using sklearn.
Print off both accuracy values using the print statements.'''

from sklearn.metrics import mean_absolute_error

# Manually calculate the MAE
n = len(predictions)
mae_one = sum(abs(y_test - predictions)) / n
print('With a manual calculation, the error is {}'.format(mae_one))

# Use scikit-learn to calculate the MAE
mae_two = mean_absolute_error(y_test,predictions)
print('Using scikit-learn, the error is {}'.format(mae_two))



#instructions
'''Manually calculate the MSE.
 Calculate the MSE using sklearn.'''

from sklearn.metrics import mean_squared_error

n = len(predictions)
# Finish the manual calculation of the MSE
mse_one = sum((y_test - predictions)**2) / n
print('With a manual calculation, the error is {}'.format(mse_one))

# Use the scikit-learn function to calculate MSE
mse_two = mean_squared_error(y_test,predictions)
print('Using scikit-learn, the error is {}'.format(mse_two))


#instructions
'''Create an array east_teams that can be used to filter labels to East conference teams.
Create the arrays true_east and preds_east by filtering the arrays y_test and predictions.
Use the print statements to print the MAE (using scikit-learn) for the East conference. The mean_absolute_error function has been loaded as mae.
The variable west_error contains the MAE for the West teams. Use the print statement to print out the Western conference MAE.'''

# Find the East conference teams
east_teams = labels == "E"

# Create arrays for the true and predicted values
true_east = y_test[east_teams]
preds_east = predictions[east_teams]

# Print the accuracy metrics
print('The MAE for East teams is {}'.format(
    mae(true_east, preds_east)))

# Print the West accuracy
print('The MAE for West conference is {}'.format(west_error))


#Classification metrics
#instructions
'''Use the confusion matrix to calculate overall accuracy.
Use the confusion matrix to calculate precision and recall.
Use the three print statements to print each accuracy value.'''

# Calculate and print the accuracy
accuracy = (491 + 324 ) / (953)
print("The overall accuracy is {0: 0.2f}".format(accuracy))

# Calculate and print the precision
precision = (491) / (491 + 15)
print("The precision is {0: 0.2f}".format(precision))

# Calculate and print the recall
recall = (491) / (491 + 123)
print("The recall is {0: 0.2f}".format(recall))



#instructions
'''Import sklearn's function for creating confusion matrices.
Using the model rfc, create category predictions on the test set X_test.
Create a confusion matrix using sklearn.
Print the value from cm that represents the actual 1s that were predicted as 1s (true positives).'''

from sklearn.metrics import confusion_matrix

# Create predictions
test_predictions = rfc.predict(X_test)

# Create and print the confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# Print the true positives (actual 1s that were predicted 1s)
print("The number of true positives is: {}".format(cm[1,1]))


#instructions
'''Import the precision or the recall metric for sklearn. Only one method is correct for the given context.
Calculate the precision or recall using y_test for the true values and test_predictions for the predictions.
Print the final score based on your selected metric.'''

from sklearn.metrics import precision_score

test_predictions = rfc.predict(X_test)

# Create precision or recall score based on the metric you imported
score = precision_score(y_test, test_predictions)

# Print the final result
print("The precision value is {0:.2f}".format(score))



#The bias variance tradeoff
#instructions
'''Create a random forest model with 25 trees, a random state of 1111, and max_features of 2. Read the print statements.'''

# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=2)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))


#instructions
'''For each loop, predict values for both the X_train and X_test datasets.
For each loop, append the accuracy_score() of the y_train dataset and the corresponding predictions to train_scores.
For each loop, append the accuracy_score() of the y_test dataset and the corresponding predictions to test_scores.
Print the training and testing scores using the print statements.'''

from sklearn.metrics import accuracy_score

test_scores, train_scores = [], []
for i in [1, 2, 3, 4, 5, 10, 20, 50]:
    rfc = RandomForestClassifier(n_estimators=i, random_state=1111)
    rfc.fit(X_train, y_train)
    # Create predictions for the X_train and X_test datasets.
    train_predictions = rfc.predict(X_train)
    test_predictions = rfc.predict(X_test)
    # Append the accuracy score for the test and train predictions.
    train_scores.append(round(accuracy_score(y_train,train_predictions), 2))
    test_scores.append(round(accuracy_score(y_test, test_predictions), 2))
# Print the train and test scores.
print("The training scores were: {}".format(train_scores))
print("The testing scores were: {}".format(test_scores))



