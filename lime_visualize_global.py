import os
from collections import Counter

from sklearn.externals import joblib
import matplotlib.pyplot as plt

def determine_most_common_misclassifications(path):
    full = os.getcwd() + '/LIME/' + path
    data = joblib.load(full)
    aca, fic, mag, new, correct = [0] * 5
    for explanation in data:
        if len(explanation) == 2:
            wrong_pred = explanation[1]
            explanation = explanation[0]
            if wrong_pred == 0:
                aca += 1
            elif wrong_pred == 1:
                fic += 1
            elif wrong_pred == 2:
                mag += 1
            else:
                new += 1
        else:
            correct += 1
    print(aca, fic, mag, new)
    print(correct)

def important_feauture_gen(path, weight='absolute', operator=False, misclass=[0,1,2,3]):
    full = os.getcwd() + '/LIME/' + path
    data = joblib.load(full)
    for explanation in data:

        # Wrong predictions
        if len(explanation) == 2:
            wrong_pred = explanation[1]
            explanation = explanation[0]
            if wrong_pred in misclass:
                for importance in explanation:
                    importance_split = importance[0].split(' ')

                    # Show > or <
                    if operator:
                        if 3 == len(importance_split):
                            feature = "" + importance_split[0] + " " + importance_split[1][0]
                        elif 5 == len(importance_split):
                            continue
                        if weight == 'relative':
                            feature = (feature, importance[1])
                        yield feature
                    else:
                        if 3 == len(importance_split):
                            feature = importance_split[0]
                        elif 5 == len(importance_split):
                            feature = importance_split[2]
                        if weight == 'relative':
                            feature = (feature, importance[1])
                        yield feature

        # Correct predictions    
        else:
            for importance in explanation:
                importance_split = importance[0].split(' ')
                
                # Show > or <
                if operator:
                    if 3 == len(importance_split):
                        feature = "" + importance_split[0] + " " + importance_split[1][0]
                    elif 5 == len(importance_split):
                        continue
                    if weight == 'relative':
                        feature = (feature, importance[1])
                    yield feature
                else:
                    if 3 == len(importance_split):
                        feature = importance_split[0]
                    elif 5 == len(importance_split):
                        feature = importance_split[2]
                    if weight == 'relative':
                        feature = (feature, importance[1])
                    yield feature

# Used when weight='relative' in importance_hist module.
def relative_gen_to_dict(gen):
    my_dict = {}
    for instance in gen:
        try:
            my_dict[instance[0]] += instance[1]
        except KeyError:
            my_dict[instance[0]] = instance[1]
    return Counter(my_dict)

# Very cool graphs. Very insightfull yes. show means if you want bad, good or
# all explanation-features shown. Weight tells the program what to do with the
# importance generated in the importnat_feature_gen module. It can be set to
# 'absolute', purely counting explantion-occurences. Other option is 'relative'
# taking the weight provided by LIME into account. Very documentation yes.
# show: 'bad', 'good' or 'all'. What type of explantions is shown
# features: #features to show in diagram
# weight: if 'relative' show importance sum rather than occurence
# operator: if True, show > and <
# misclass: if show=='bad', what misclassifications are shown. List. [0,1,2,3]
def importance_hist(label, show='all', n_features=10, weight='absolute', operator=False, misclass=[0,1,2,3]):
    my_cute_counter = Counter()
    if type(label) == list:
        for elm in label:
            good = 'lime_tabular_' + elm + '_correct.pkl'
            bad = 'lime_tabular_' + elm + '_wrong.pkl'
            dict_good = important_feauture_gen(good, weight, operator, misclass)
            dict_bad = important_feauture_gen(bad, weight, operator, misclass)
            if weight == 'absolute':
                occurence_good = Counter(dict_good)
                occurence_bad = Counter(dict_bad)
            elif weight == 'relative':
                occurence_good = relative_gen_to_dict(dict_good)
                occurence_bad = relative_gen_to_dict(dict_bad)
            if show == 'all':
                occurence = occurence_bad + occurence_good
            elif show == 'bad':
                occurence = occurence_bad
            elif show == 'good':
                occurence = occurence_good
            else:
                print("Wow what a dumb idea")
                return 1
            my_cute_counter += occurence
    values, labels = ([],[])
    for pair in my_cute_counter.most_common()[:n_features][::-1]:
        values.append(pair[1])
        labels.append(pair[0])
    plt.barh(labels, values)
    plt.show()



labels = ['newspaper']
importance_hist(labels, 'bad', 10, 'relative', True, [2])