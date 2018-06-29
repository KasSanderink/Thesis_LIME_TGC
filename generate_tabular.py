import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import random

import pyphen
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale
from sklearn.externals import joblib
import spacy

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
             'flesh', 'flesh-k','neg', 'neu', 'pos', 'compound']
    return names

def generate_features(text):

    # Initialize objects
    nlp = spacy.load('en', disable=['tagger', 'parser', 'textcat'])
    dic = pyphen.Pyphen(lang='en')
    tok = RegexpTokenizer('\w+')
    sid = SentimentIntensityAnalyzer()
    genre_result = []

    # Tagsets
    POS_tags = {"''", '(', ')', ',', '--', '.', ':', 'CC', 'CD', 'DT', 'EX', 
                 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 
                 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
                 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 
                 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', '$', '#'}
    entity_tags = {'PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 
                  'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
                  'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'FAC'}

    # POS-tags
    empty_counter = {key: 0 for key in POS_tags}
    tags = nltk.pos_tag(nltk.word_tokenize(text))
    tags_counter = Counter(tag for w,tag in tags)
    final_dict = {**empty_counter, **dict(tags_counter)}
    sorted_items = sorted(final_dict.items())
    tag_count = np.array([item[1] for item in sorted_items])
    
    # Entities
    empty_counter = {key: 0 for key in entity_tags}
    doc = nlp(text)
    entity_counter = Counter(ent.label_ for ent in doc.ents)
    final_dict = {**empty_counter, **dict(entity_counter)}
    sorted_items = sorted(final_dict.items())
    ent_count = np.array([item[1] for item in sorted_items])

    # Sentence, word and syllable count
    n_sent = len(nltk.sent_tokenize(text)) 
    words = tok.tokenize(text)
    n_word = len(words)
    syllables = [dic.inserted(word) for word in words]
    syllable_list = [len(re.findall('-', word)) + 1 for word in syllables] 
    n_syl = sum(syllable_list)
    syntax_count = np.array([n_sent, n_word, n_syl])

    # Readability score
    try:
        flesh = 206.835-1.015*(n_word/n_sent)-84.6*(n_syl/n_word)
        flesh_kincaid = 0.39*(n_word/n_sent)+11.8*(n_syl/n_word)-15.59
    except ZeroDivisionError:
        flesh = 100
        flesh_kincaid = 10

    readability_score = np.array([flesh, flesh_kincaid])

    # Sentiment
    score_dic = sid.polarity_scores(text)
    sentiment = np.array([score_dic['neg'], score_dic['neu'], 
                          score_dic['pos'], score_dic['compound']])

    # Concat all features
    instance_result = np.concatenate([tag_count/n_word, ent_count/n_word, 
                                      syntax_count, readability_score, 
                                      sentiment])
                                      
    return instance_result

def generate_tabular(data_path):
    names_features = get_feature_names()
    names_info = ['target', 'year', 'ID']
    data = pd.read_pickle(data_path)
    texts = data['text']
    features_list = []
    for text in texts:
        features = list(generate_features(text))
        features_list.append(features)
    features_df = pd.DataFrame.from_records(features_list)
    information = data[['target', 'year', 'ID']].reset_index()
    data_tabular = pd.concat([features_df, information], axis=1)
    data_tabular = data_tabular.drop(['index'], axis=1)
    data_tabular.columns = names_features + names_info
    return data_tabular

def generate_tabular_model_explain():
    data_model_tabular = generate_tabular('data_model.pkl')
    data_model_tabular.to_pickle('/datastore/10814418/data_model_tabular.pkl')
    data_explain_tabular = generate_tabular('data_explain.pkl')
    data_explain_tabular.to_pickle('/datastore/10814418/data_explain_tabular.pkl')

if __name__ == '__main__':
    generate_tabular_model_explain()