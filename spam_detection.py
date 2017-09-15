import pandas as pd

import numpy as numpy


# Understanding the data

df = pd.read_table('SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])

# Outputting first 5 column

print(df.head())
print('--------------------------------------')

# Converting the values in the label column to numerical values

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
print('------------------------------------------------------')
print(df.head())
print('---------------------------------------------------')

# Splitting data into Training and Testing sets

from sklearn.model_selection import train_test_split

'''X_train is our training data for the 'sms_message' column
    X_test is our testing data for the 'sms_message' column
    y_train is our training data for the 'label' column'
    y_test is our testing data for the 'label' column '''

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the testing set: {}'.format(X_test.shape[0]))
print('----------------------------------------------------------------------')

# Applying Bag of Words processing to our data set

from sklearn.feature_extraction.text import CountVectorizer

# Instantiate tthe CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transfrom testing data and return the matrix

testing_data = count_vector.transform(X_test)

# Naive-Bayes implementation (multinomial because of discrete features on texts)

from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()

print(naive_bayes.fit(training_data, y_train))
print('----------------------------------------------------------------------')

# Making some predictions on the test data stored in 'testing_data' using .predict().

predictions = naive_bayes.predict(testing_data)

# Evaluating our model by computing(accuracy, precision, recall, and f1 score)

''' Accuracy: Measures how often the classifier makes the correct prediction. It's the ratio of the number of correct
predictions to the total number of predictions. (Number of test data)

    Precision: What proportion of messages we classified as spam, actually were spam.
        [True Positive / (True Prositive + False Positive)]

    Recall(Sensitivity): What proportion of the messages that actually were spam were classified by us spam.
        [True Positive / (True Positive + False Negative)]

    I will be using all 4 metrics to make sure our model does well. FOr all 4 metrics whose values can range from 0 to 1,
    having a score as close to 1 as possible is a good indicator of how well our model is doing.
'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))



