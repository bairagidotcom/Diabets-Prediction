import pandas as pd
import numpy as np
import pickle


df= pd.read_csv('diabetes.csv')
inputs = df.drop('Outcome',axis='columns')
target = df.Outcome

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)


from sklearn import svm
model = svm.SVC(kernel='linear',C=20,gamma='auto')
model.fit(X_train,y_train)



pickle.dump(model , open('diabetes_predict.pkl' ,'wb'))
diabetes_predict = pickle.load(open('diabetes_predict.pkl', 'rb'))

