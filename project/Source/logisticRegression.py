# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


#loading the dataset
df = pd.read_csv('E:\\creditcard.csv')
head=df.head()
print(head)
describe=df.describe()
print(describe)

# checking Null Values!
Null=df.isnull().sum().max()
print(Null)

#no of columns
columns=df.columns
print(columns)

# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

#Notice how imbalanced is our original dataset! Most of the transactions are non-fraud. If we use this dataframe as the base for our predictive models and analysis we might get a lot of errors and our algorithms will probably overfit since it will "assume" that most transactions are not fraud. But we don't want our model to assume, we want our model to detect patterns that give signs of fraud!


colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()


#scaling of the dataset and creating sub sample of dataframe

# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.


rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

df.head()
print(df.head())



#Random Undersampling

# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

# Lets shuffle the data before creating the subsamples

df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
print(new_df.head())

# Equally Distributing.
# Now that we have our dataframe correctly balanced, we can go further with our analysis and data preprocessing.

print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

sns.countplot('Class', data=new_df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

#Classifiers
#training Four type of classifiers

# Undersampling before cross validating (prone to overfit)
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# Our data is already scaled we should split our training and test sets
from sklearn.model_selection import train_test_split, cross_val_score

# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

print("The split of the under_sampled data is as follows")
print("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))



#Using the logistic regression to build the initail model. Let us see if this is the best parameter later
classifier= LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train.ravel())

#Predict the class using X_test
y_pred = classifier.predict(X_test)



#confusion matrix
#cm1 is the confusion matrix 1 which uses the undersampled dataset
cm1 = confusion_matrix(y_test,y_pred)


import seaborn as sns
import matplotlib.pyplot as plt

ax= plt.subplot()
sns.heatmap(cm1, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted ');ax.set_ylabel('Actual');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['1', '0']);
plt.show()

print("The accuracy is " + str((cm1[1, 1] + cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1] + cm1[1, 0] + cm1[1, 1]) * 100) + " %")
print("The recall from the confusion matrix is " + str(cm1[1, 1] / (cm1[1, 0] + cm1[1, 1]) * 100) + " %")

#Applying 10 fold cross validation
accuracies = cross_val_score(estimator = classifier, X=X_train, y = y_train.ravel(), cv = 10)
mean_accuracy= accuracies.mean()*100
std_accuracy= accuracies.std()*100
print("The mean accuracy in %: ", accuracies.mean()*100)
print("The standard deviation in % ", accuracies.std()*100)
print("The accuracy of our model in % is betweeen {} and {}".format(mean_accuracy-std_accuracy, mean_accuracy+std_accuracy))



# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV


# Logistic Regression
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_search=grid_log_reg.fit(X_train, y_train)
best_accuracy=grid_search.best_score_
print( "The best accuracy using gridSearch is", best_accuracy)
# We automatically get the logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_
# print("The best parameters for using this model is", log_reg)

#fitting the model with the best parameters
classifier_with_best_parameters =  LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
classifier_with_best_parameters.fit(X_train, y_train.ravel())

#predicting the Class
y_pred_best_parameters = classifier_with_best_parameters.predict(X_test)

#creating a confusion matrix
#cm2 is the confusion matrix  which uses the best parameters
cm2 = confusion_matrix(y_test, y_pred_best_parameters)

#plotting
ax= plt.subplot()
sns.heatmap(cm2, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted ');ax.set_ylabel('Actual');
ax.set_title('Confusion Matrix2 using best parameters');
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['1', '0']);
plt.show()

print("The accuracy is " + str((cm2[1, 1] + cm2[0, 0]) / (cm2[0, 0] + cm2[0, 1] + cm2[1, 0] + cm2[1, 1]) * 100) + " %")
print("The recall from the confusion matrix 2 is " + str(cm2[1, 1] / (cm2[1, 0] + cm2[1, 1]) * 100) + " %")



# Testing the model against the full dataset(skewed)
#creating a new dataset to test our model
datanew= df.copy()

#Now to test the model with the whole dataset
# datanew['scaled_amount'] = rob_scaler.fit_transform(datanew['Amount'].values.reshape(-1,1))
#dropping time and old amount column
# datanew= datanew.drop(["Time","Amount"], axis= 1)

#separating the x and y variables to fit our model
X_full= datanew.iloc[:, new_df.columns != "Class"].values

y_full= datanew.iloc[:, new_df.columns == "Class"].values

#
# Splitting the full dataset into training and test set
#splitting the full dataset into training and test set
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size= 0.25, random_state= 0)

print("The split of the full dataset is as follows")
print("X_train_full: ", len(X_train_full))
print("X_test_full: ", len(X_test_full))
print("y_train_full: ", len(y_train_full))
print("y_test_full: ", len(y_test_full))


#predicting y_pred_full_dataset
y_pred_full_dataset= classifier_with_best_parameters.predict(X_test_full)

#confusion matrix usign y_test_full and ypred_full
cm3 = confusion_matrix(y_test_full, y_pred_full_dataset)


#plotting
ax= plt.subplot()
sns.heatmap(cm2, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted ');ax.set_ylabel('Actual');
ax.set_title('Confusion Matrix3 using full dataset');
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['1', '0']);
plt.show()

print("The accuracy is " + str((cm3[1, 1] + cm3[0, 0]) / (cm3[0, 0] + cm3[0, 1] + cm3[1, 0] + cm3[1, 1]) * 100) + " %")
print("The recall from the confusion matrix 3 is " + str(cm3[1, 1] / (cm3[1, 0] + cm3[1, 1]) * 100) + " %")