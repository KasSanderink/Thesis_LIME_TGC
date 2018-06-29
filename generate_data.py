# Create a generator. Generates one instance at a time.
###############################################################################

# Import libraries
import glob
import os
import pandas as pd
import numpy as np
import re
import random

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


random.seed(42)

def document_generator():
    path = '/datastore/10814418'
    folders = glob.glob(path+'/COCA/*')
    n_folders = len(folders)
    n_instance = -1
    for i in range(n_folders):
        folder = folders[i]
        target = i
        files = glob.glob(folder+'/*')
        n_files = len(files)
        for j in range(n_files):
            file = files[j]
            year = files[j].rsplit("_",1)[1][:4]
            print(year)
            with open (file, "r") as current_file:
                raw_text = current_file.read()
                if re.search('##\d\d\d\d\d\d\d', raw_text):
                    text = re.split('##\d\d\d\d\d\d\d', raw_text)
                else:
                    text = re.split('@@\d\d\d\d\d\d\d', raw_text)
                text = [item.replace(" @ @ @ @ @ @ @ @ @ @ ", " ") 
                        for item in text]
                n_strings = len(text) # Shrink dataset
                for k in range(n_strings):
                    if len(text[k]) < 100:
                        continue
                    n_instance += 1
                    yield text[k], target, year, n_instance

def generate_data_all():
    gen = document_generator()
    data = []
    for file in gen:
        data.append(file)
    df = pd.DataFrame.from_records(data, columns=['text', 'target', 'year', 'ID'])
    df.to_pickle('data_all.pkl')
    return 0

def generate_data_model_explain():
    data_all = pd.read_pickle('data_all.pkl')
    data_model, data_explain, _, _ = train_test_split(data_all, data_all,
                                                      test_size=0.25,
                                                      train_size=0.75)
    data_model.to_pickle('data_model.pkl')
    data_explain.to_pickle('data_explain.pkl')
    return 0