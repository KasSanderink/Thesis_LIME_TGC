# Basic libraries
import random
import pandas as pd
import numpy as np
from time import time
import os

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
from sklearn.preprocessing import FunctionTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

random.seed(42)

def visualize_tree(tree_string):
    source = Source(tree_string)
    source.render('tree', view=True)
    return 0

def train(data_path, model_name):
    data = pd.read_pickle(data_path)
    target = np.array(data['target']).astype('int')
    data = data.drop(['target', 'year', 'ID', 'FACILITY', '--'], axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(data, target) 
    pipeline = Pipeline([('tree', RandomForestClassifier())])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_name)
    predictions = pipeline.predict(X_test)
    score = accuracy_score(y_test, predictions)
    print("Accuracy: {0:.2f}".format(score))
    return 0

def print_predictions():
    data = pd.read_pickle('data_explain_tabular.pkl')
    target = np.array(data['target']).astype('int')
    ID = np.array(data['ID']).astype('int')
    data = data.drop(['target', 'year', 'ID', '--', 'FACILITY'], axis=1)
    clf = joblib.load('model_tabular_forest.pkl')

    predictions_proba = clf.predict_proba(data)
    predictions = [np.argmax(i) for i in predictions_proba]
    score = accuracy_score(target, predictions)
    print("Accuracy: {0:.2f}".format(score))
    N = len(ID)
    for i in range(N):
        if target[i] == 0:
            if predictions_proba[i][0] > 0.8:
                print(ID[i], predictions_proba[i], target[i])
    print('pffrt')
    return 0

def retrieve_instance_by_ID(ID):
    df = pd.read_pickle('data_explain.pkl')
    for index, row in df.iterrows():
        if row['ID'] == ID:
            print(row['text'])

#print_predictions()
retrieve_instance_by_ID(92289)
#train('data_model_tabular.pkl', 'model_tabular_forest.pkl')