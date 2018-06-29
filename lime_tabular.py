import pandas as pd
import os
import numpy as np
import warnings
import random

from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import matplotlib.pyplot as plt

from generate_tabular import generate_features

warnings.filterwarnings(module='re*', action='ignore', 
                        category=FutureWarning)
random.seed(42)

def get_feature_names():
    names = ['#', '$', "''", '(', ')', ',', '--', '.', ':', 'CC', 'CD', 'DT', 
             'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 
             'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 
             'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
             'WDT', 'WP', 'WP$', 'WRB', '``', 'CARDINAL', 'DATE', 'EVENT', 
             'FAC', 'FACILITY', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 
             'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 
             'QUANTITY', 'TIME', 'WORK_OF_ART', 'n_sent', 'n_word', 'n_syl', 
             'flesh', 'flesh_kincaid','neg', 'neu', 'pos', 'compound']
    return names

def remove_features(features, removed_features):
    names = get_feature_names()
    feature_series = pd.Series(features)
    feature_series.index = names
    feature_series = feature_series.drop(removed_features)
    feature_series = feature_series.as_matrix()
    return feature_series

# Load a sample text file
def load_file(path):
    with open(path, 'r') as file:
        text = file.read()
    return text

def lime_tabular_local():
    targets = ['academic', 'fiction', 'magazine', 'newspaper']
    df = pd.read_pickle('data_model_tabular.pkl')
    df = df.drop(['year', 'target', 'ID', '--', 'FACILITY'], axis=1)
    feature_names = list(df)
    feature_names[66] = 'flesch'
    feature_names[67] = 'flesch-k'
    train = df.as_matrix()
    clf = joblib.load('model_tabular_forest.pkl')
    text = load_file('text_magazine_hard.txt')
    features = generate_features(text)
    features = remove_features(features, ['--', 'FACILITY'])
    probas = clf.predict_proba([features])
    result = 1
    print(result)
    print(probas)
    explainer = LimeTabularExplainer(train, feature_names=feature_names, 
                                     class_names=targets)
    explanation = explainer.explain_instance(features, clf.predict_proba,
                                             num_features=10, top_labels=1)
    explanation.as_pyplot_figure(label=result)
    plt.title("")
    plt.show()

# It saves every found set of explanations in a different list, and saves
# these lists seperatly in a pretty little pickle. For wrong predicions, the
# wrong prediction and the true label are saved.
def lime_tabular_global():
    targets = ['academic', 'fiction', 'magazine', 'newspaper']
    data = pd.read_pickle('data_explain_tabular.pkl')
    clf = joblib.load('model_forest_tabular.pkl')
    feature_names = list(data)
    target = np.array(data['target'])
    data = data.drop(['target', 'year', 'ID'], axis=1).as_matrix()
    explainer = LimeTabularExplainer(data, feature_names=feature_names, 
                                     class_names=targets)
    N = data.shape[0]
    academic, fiction, magazine, newspaper = ([],[],[],[])
    academic_w, fiction_w, magazine_w, newspaper_w = ([],[],[],[])
    for i in range(N):
        pred = clf.predict(data[i].reshape(1,-1))[0]
        if pred == target[i]:
            explanation = explainer.explain_instance(data[i], 
                                                     clf.predict_proba,
                                                     num_features=10,
                                                     top_labels=4)
            result = explanation.as_list(label=pred)
            if 0 == target[i]:
                academic.append((result, pred))
            elif 1 == target[i]:
                fiction.append((result, pred))
            elif 2 == target[i]:
                magazine.append((result, pred))
            elif 3 == target[i]:
                newspaper.append((result, pred))
            else:
                return 1
        else:
            explanation = explainer.explain_instance(data[i], 
                                                     clf.predict_proba,
                                                     num_features=10,
                                                     top_labels=4)
            result = explanation.as_list(label=pred)
            if 0 == target[i]:
                academic_w.append((result, pred))
            elif 1 == target[i]:
                fiction_w.append((result, pred))
            elif 2 == target[i]:
                magazine_w.append((result, pred))
            elif 3 == target[i]:
                newspaper_w.append((result, pred))
            else:
                return 1

    joblib.dump(academic, 'lime_academic.pkl')
    joblib.dump(fiction, 'lime_fiction.pkl')
    joblib.dump(magazine, 'lime_magazine.pkl')
    joblib.dump(newspaper, 'lime_newspaper.pkl')
    all_explanations = academic + fiction + magazine + newspaper
    joblib.dump(all_explanations, 'lime_all.pkl')

    joblib.dump(academic_w, 'lime_academic_wrong.pkl')
    joblib.dump(fiction_w, 'lime_fiction_wrong.pkl')
    joblib.dump(magazine_w, 'lime_magazine_wrong.pkl')
    joblib.dump(newspaper_w, 'lime_newspaper_wrong.pkl')
    all_explanations_w = academic_w + fiction_w + magazine_w + newspaper_w
    joblib.dump(all_explanations_w, 'lime_all_wrong.pkl')

lime_tabular_local()