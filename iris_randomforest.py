import pickle 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.data')

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

score = accuracy_score(y_test,y_pred)

print ('accuracy',metrics.accuracy_score(y_test, y_pred))

# print(x_test)


import streamlit as st
import streamlit.components.v1 as components

col1, col2, col3 = st.columns(3)
col1.metric(" ", 'accuracy',metrics.accuracy_score(y_test, y_pred))
# col2.metric("Wind", "9 mph", "-8%")
# col3.metric("Humidity", "86%", "4%")



pickle_out = open("model_iris.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()