#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
import pandas as pd
from time import time
from collections import Counter

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# NB: This list is not the list that is used for the final classifier, but is the list of
# features considered when performing feature selection. This list along with the
# engineered features are filtered to only include the features produced by the
# feature selection process.
features_list = [
    'poi',
    'exercised_stock_options',
    'total_stock_value',
    'deferred_income',
    'salary',
    'bonus',
    'from_poi_to_this_person',
    'expenses',
    'from_this_person_to_poi',
    'long_term_incentive',
    'shared_receipt_with_poi',
    'director_fees',
    'deferral_payments']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame()
people = []
for person in data_dict.keys():
    if person == 'TOTAL': continue
    people.append(person)
    data = data_dict[person]
    columns = data.keys()
    values = [0 if v == 'NaN' else v for v in data.values()]
    df = df.append(pd.DataFrame([values], columns=columns), ignore_index=True)

df_without_email = df.copy().drop('email_address', 1)

### Task 2: Remove outliers
# NB: The only outlier to be removed was the 'TOTAL' row. This was excluded in the previous step.
good_data = df_without_email[features_list].copy()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# This new feature should highlight users who had email conversations with
# a POI rather than people who only sent or received emails to / from a POI.

good_data['to_poi_to_the_from_poi_power'] = good_data['from_this_person_to_poi'] ** good_data['from_poi_to_this_person']
features_list.append('to_poi_to_the_from_poi_power')

good_data['from_poi_to_the_to_poi_power'] = good_data['from_poi_to_this_person'] ** good_data['from_this_person_to_poi']
features_list.append('from_poi_to_the_to_poi_power')

good_data['expenses_to_the_bonus_power'] = good_data['expenses'] ** good_data['bonus']
features_list.append('expenses_to_the_bonus_power')

good_data['lti_times_expenses_to_the_bonus_power'] = good_data['long_term_incentive'] * good_data['expenses_to_the_bonus_power']
features_list.append('lti_times_expenses_to_the_bonus_power')

good_data['salary_times_expenses_to_the_bonus_power'] = good_data['salary'] * good_data['expenses_to_the_bonus_power']
features_list.append('salary_times_expenses_to_the_bonus_power')


# Normalise the remaining features (except for 'poi').
perform_feature_scaling_and_normalisation = True
if perform_feature_scaling_and_normalisation:
    pois = good_data['poi']
    df_to_normalise = good_data.copy().drop('poi', 1)

    df_norm = (df_to_normalise - df_to_normalise.min()) / (df_to_normalise.max() - df_to_normalise.min())
    df_exp = df_norm.apply(np.exp)
    df_norm_sm = df_exp / df_exp.sum()

    normalised_df = pd.concat([pois, df_norm_sm], axis=1)
    good_data = normalised_df.copy()

print "Performed feature scaling and normalisation: {}".format(perform_feature_scaling_and_normalisation)

# Convert the pandas data_frame back into a dictionary so it can be
# fed into featureFormat and exported at the end of the script.
def reconstruct_dict_data(df, features_list, people):
    data_dict = {}
    for index, row in df.iterrows():
        data_dict[people[index]] = row[features_list].to_dict()
    return data_dict

my_dataset = reconstruct_dict_data(good_data, features_list, people)

# Feature Selection:

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# This code was used to score and evaluate features during the feature selection
# portion of this project.
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score)
from sklearn.cross_validation import train_test_split

results = {}
clf = DecisionTreeClassifier()
num_features = len(features[0])
for k in xrange(1, num_features + 1):
    # Test a classifier with the k best features.
    data_with_k_best_features = SelectKBest(k=k).fit_transform(features, labels)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(data_with_k_best_features, labels, test_size=0.3, random_state=42)

    clf = clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    precision = precision_score(labels_test, predictions)
    recall = recall_score(labels_test, predictions)

    # Get the names and scores of the k best features.
    feature_selector = SelectKBest().fit(features, labels)
    sorted_features = sorted(zip(feature_selector.scores_, features_list[1:]), reverse=True)
    k_best_features = sorted_features[:k]

    results[k] = {
        'precision': precision,
        'recall': recall,
        'features_with_scores': k_best_features
    }

# Precision and Recall are maximised for five features.
print "The five best features (with SelectKBest scores): {}".format(results[5]['features_with_scores'])
# [(26.685985390699152, 'exercised_stock_options'), (24.38043027658092, 'total_stock_value'), (18.206830357713191, 'salary'), (17.349092543772667, 'bonus'), (15.420749210493559, 'to_poi_to_the_from_poi_power')]

plot_feature_selection_results = False
if plot_feature_selection_results:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    k_values = results.keys()
    precision_scores = [r['precision'] for r in results.values()]
    recall_scores = [r['recall'] for r in results.values()]

    sns.set_style("darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(k_values, precision_scores, label='Precision Score')
    plt.plot(k_values, recall_scores, label='Recall Score')
    ax.set_title('Decision Tree Classifier Precision and Recall Scores\nfor K Best Features')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Score')
    plt.legend()
    plt.show()

# 'poi' plus the 4 features that maximised precision and recall during the feature selection process.
features_list = [
    'poi',
    'exercised_stock_options',
    'total_stock_value',
    'salary',
    'bonus',
    'to_poi_to_the_from_poi_power'
]
my_dataset = reconstruct_dict_data(good_data, features_list, people)
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Initial classifiers to consider.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
classifiers_to_compare = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB()
]

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print "Number of Features:", len(features_train[0])

compare_classifiers = False
optimise_classifier_with_grid_search = True
train_and_evaluate_classifier = True

if train_and_evaluate_classifier:

    if compare_classifiers:
        for clf in classifiers_to_compare:
            print '****************************************'
            print clf
            t0 = time()
            clf.fit(features_train, labels_train)
            print "training time:", round(time()-t0, 3), "s"

            t0 = time()
            y_pred = clf.predict(features_test)
            print "prediction time:", round(time()-t0, 3), "s"

            from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef)
            print '-------------- Scores ------------------'
            print "accuracy_score: {}".format(accuracy_score(labels_test, y_pred))
            print "precision_score: {}".format(precision_score(labels_test, y_pred))
            print "recall_score: {}".format(recall_score(labels_test, y_pred))
            print "f1_score: {}".format(f1_score(labels_test, y_pred))
            # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
            print "matthews_corrcoef: {}".format(matthews_corrcoef(labels_test, y_pred))
            print '****************************************'



    if optimise_classifier_with_grid_search:
        from sklearn.cross_validation import StratifiedShuffleSplit
        from sklearn.model_selection import GridSearchCV

        use_decision_tree = True
        use_knn = False

        if use_decision_tree:
            search_clf = DecisionTreeClassifier()

            # Grid Search Paramters for the Random Forest Classifier
            search_parameters = {
                # Search with feature scaling and normalisation.
                'criterion': ['gini', 'entropy'],
                'max_features': range(1, len(features_train[0])+1)
                # Fitting 100 folds for each of 10 candidates, totalling 1000 fits
                # [Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed:    3.2s finished
                # training time: 3.26 s
                # {'max_features': 3, 'criterion': 'gini'}
                # prediction time: 0.0 s
                # -------------- Scores ------------------
                # accuracy_score: 1.0
                # precision_score: 1.0
                # recall_score: 1.0
                # f1_score: 1.0
                # matthews_corrcoef: 1.0
                # ****************************************
                # DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                #             max_features=3, max_leaf_nodes=None, min_impurity_split=1e-07,
                #             min_samples_leaf=1, min_samples_split=2,
                #             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                #             splitter='best')
                # Accuracy: 0.81967	Precision: 0.33566	Recall: 0.36000	F1: 0.34741	F2: 0.35485
                # Total predictions: 15000	True positives:  720	False positives: 1425	False negatives: 1280	True negatives: 11575
            }
        elif use_knn:
            search_clf = KNeighborsClassifier()

            # Grid Search Paramters for the Random Forest Classifier
            search_parameters = {
                # Search with feature scaling and normalisation.
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'n_neighbors': range(1, 6)
                # Fitting 100 folds for each of 15 candidates, totalling 1500 fits
                # [Parallel(n_jobs=1)]: Done 1500 out of 1500 | elapsed:    8.1s finished
                # training time: 8.118 s
                # {'n_neighbors': 1, 'algorithm': 'ball_tree'}
                # prediction time: 0.001 s
                # -------------- Scores ------------------
                # accuracy_score: 1.0
                # precision_score: 1.0
                # recall_score: 1.0
                # f1_score: 1.0
                # matthews_corrcoef: 1.0
                # ****************************************
                # KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
                #            metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                #            weights='uniform')
                # Accuracy: 0.80920	Precision: 0.26082	Recall: 0.23500	F1: 0.24724	F2: 0.23975
                # Total predictions: 15000	True positives:  470	False positives: 1332	False negatives: 1530	True negatives: 11668
            }


        ssscv = StratifiedShuffleSplit(labels, 100)
        gs = GridSearchCV(search_clf, search_parameters, cv=ssscv, scoring="recall", verbose=1)

        print '****************************************'
        t0 = time()
        gs.fit(features, labels)
        print "training time:", round(time()-t0, 3), "s"

        clf = gs.best_estimator_
        print gs.best_params_

        t0 = time()
        y_pred = clf.predict(features_test)
        print "prediction time:", round(time()-t0, 3), "s"

        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef)
        print '-------------- Scores ------------------'
        print "accuracy_score: {}".format(accuracy_score(labels_test, y_pred))
        print "precision_score: {}".format(precision_score(labels_test, y_pred))
        print "recall_score: {}".format(recall_score(labels_test, y_pred))
        print "f1_score: {}".format(f1_score(labels_test, y_pred))
        # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        print "matthews_corrcoef: {}".format(matthews_corrcoef(labels_test, y_pred))
        print '****************************************'

        ### Task 6: Dump your classifier, dataset, and features_list so anyone can
        ### check your results. You do not need to change anything below, but make sure
        ### that the version of poi_id.py that you submit can be run on its own and
        ### generates the necessary .pkl files for validating your results.
        dump_classifier_and_data(clf, my_dataset, features_list)
