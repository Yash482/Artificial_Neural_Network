#Part 1: Data Processing

#Importing libs
import numpy as np
import matplotlib.pyplot as mlt
import pandas as pd

#get dataset
dataset = pd.read_csv('churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X1 = LabelEncoder()
X[:, 2] = labelencoder_X1.fit_transform(X[:, 2])
ct = ColumnTransformer(transformers = [('encoder' , OneHotEncoder(), [1])], remainder = 'passthrough')
X= np.array(ct.fit_transform(X))
"""
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
"""
#splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#evaluating ANN
#no tunning
from keras.models import Sequential
from keras.layers import Dense
#these are for structure
from keras.wrappers.scikit_learn import KerasClassifier
#this is to make the classifier like this to wrap it with sklearn ob for tunning
from sklearn.model_selection import cross_val_score
#this class is for evaluating by k folds

#now we define the build function
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu'))
    classifier.add(Dense(units = 6, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

"""
it takes 7-8 hrs to run

#we will implement best ANN by tunning
from keras.models import Sequential
from keras.layers import Dense
#these are for structure
from keras.wrappers.scikit_learn import KerasClassifier
#this is to make the classifier like this to wrap it with sklearn ob for tunning
from sklearn.model_selection import GridSearchCV
#this class is for tunning

#now we define the build function
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu'))
    classifier.add(Dense(units = 6, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [10, 25, 32],
              'epochs' : [100, 250, 500],
              'optimizer' : ['adams', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_para = grid_search.best_params_
best_accuracy = grid_search.best_score_

"""
