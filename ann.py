#it is done in 2 parts

#Part 1: Data Processing

#Importing libs
import numpy as np
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

#Part 2
#create ANN

#import keras
#import keras
#here we import 2 modules 
from keras.models import Sequential # this is to initialise our ANN
from keras.layers import Dense #this is to add layers to our ANN

# initiasing ANN : this means defining it as a sequence of layers
classifier = Sequential()

#adding input layer and 1st hidden layer
classifier.add(Dense(units = 6, activation = 'relu'))

#adding other hidden layer
classifier.add(Dense(units = 6, activation = 'relu'))

#adding output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#training our model
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, validation_split= 0.2)

#predicting result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#to predict for particular data
"""
Credit : 600
geography : France(1 0 0)
Gender : Male(1)
Age : 40
Tenure: 3
Bal: 60000
no of products: 2
credit card : yes
is active: yes
expected sal: 50000
"""
to_predict = [[1, 0 ,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
print(classifier.predict(sc.transform(to_predict)))
