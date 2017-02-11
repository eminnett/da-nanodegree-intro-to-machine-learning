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
    'deferral_payments'] # All features from SelectKBest(f_classif, k='all').fit(features, labels) with a score > 1

# [(14.569952785199289, 'exercised_stock_options'), (13.463788685733716, 'total_stock_value'), (11.048887139176928, 'deferred_income'), (10.069349192344491, 'salary'), (7.9918239677239162, 'bonus'), (4.5168302920608543, 'from_poi_to_this_person'), (4.0562445750011982, 'expenses'), (3.9070832918362397, 'from_this_person_to_poi'), (2.6690894423299607, 'long_term_incentive'), (2.4159431210204221, 'shared_receipt_with_poi'), (1.7112920228258224, 'director_fees'), (1.1406908069267196, 'deferral_payments'), (0.7449846666821448, 'restricted_stock'), (0.26832025035757689, 'total_payments'), (0.25759402193820735, 'product_of_conversations_with_poi'), (0.2059939057186104, 'other'), (0.12019652570588263, 'loan_advances'), (0.11554997571943461, 'from_messages'), (0.078064044802292093, 'restricted_stock_deferred'), (0.0001337062921765587, 'to_messages')]

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
# NB: This outlier classification process has been borrowed from
# my work on the Machine Learning Udacity Nano Degree. A more verbose version that
# prints intermediate steps can be found in the accompanying IPython notebook.

df_with_features = df_without_email[features_list]

# For each feature find the data points with extreme high or low values
ouliers_by_feature = {}
df_with_outliers = df_with_features.copy()#.drop('restricted_stock_deferred', 1) Removing this from consideration feels a bit smelly.
for feature in df_with_outliers.keys():
    df_exclude_zeroes = df_with_outliers.copy()[df_with_outliers[feature] != 0]

    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(df_exclude_zeroes[feature], 25)

    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(df_exclude_zeroes[feature], 75)

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5

    ouliers_by_feature[feature] = df_exclude_zeroes[~((df_exclude_zeroes[feature] >= Q1 - step) & (df_exclude_zeroes[feature] <= Q3 + step))]

# Consider a row an overall outlier if it is an outlier for more than a quarter of all the features.
all_indices = [df_with_outliers.index.tolist() for feature, df_with_outliers in ouliers_by_feature.iteritems()]
flattened_indices = [index for index_list in all_indices for index in index_list]
outliers  = [index for index, count in Counter(flattened_indices).iteritems() if count > len(features_list) / 4]
# This process results in the following IDs being found as outliers for more than a quarter of the features.
# [128, 43, 65, 82, 95]
print "Indices for rows that include outliers for more than a quarter of the features: '{}'".format(outliers)

# Remove the outliers, if any were specified
good_data = df_with_features.copy().drop(df_with_features.index[outliers])


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# This new feature should highlight users who had email conversations with
# a POI rather than people who only sent or received emails to / from a POI.

good_data['to_poi_to_the_from_poi_power'] = good_data['from_this_person_to_poi'] ** good_data['from_poi_to_this_person']
features_list.append('to_poi_to_the_from_poi_power')
# (10.755957285334565, 'from_poi_to_the_to_poi_power') (SelectKBest)

good_data['from_poi_to_the_to_poi_power'] = good_data['from_poi_to_this_person'] ** good_data['from_this_person_to_poi']
features_list.append('from_poi_to_the_to_poi_power')
# (7.4721617882646187, 'from_poi_to_the_to_poi_power') (SelectKBest)

good_data['expenses_to_the_bonus_power'] = good_data['expenses'] ** good_data['bonus']
features_list.append('expenses_to_the_bonus_power')
# (11.693381672323559, 'expenses_to_the_bonus_power') (SelectKBest)

good_data['lti_times_expenses_to_the_bonus_power'] = good_data['long_term_incentive'] * good_data['expenses_to_the_bonus_power']
features_list.append('lti_times_expenses_to_the_bonus_power')
# (21.365611192932043, 'lti_times_expenses_to_the_bonus_power') (SelectKBest)

good_data['salary_times_expenses_to_the_bonus_power'] = good_data['salary'] * good_data['expenses_to_the_bonus_power']
features_list.append('salary_times_expenses_to_the_bonus_power')
# (8.6526886035309953, 'salary_times_expenses_to_the_bonus_power') (SelectKBest)


# Normalise the remaining features (except for 'poi').
pois = good_data['poi']
df_to_normalise = good_data.copy().drop('poi', 1)

df_norm = (df_to_normalise - df_to_normalise.min()) / (df_to_normalise.max() - df_to_normalise.min())
df_exp = df_norm.apply(np.exp)
df_norm_sm = df_exp / df_exp.sum()

normalised_df = pd.concat([pois, df_norm_sm], axis=1)

# Convert the pandas data_frame back into a dictionary so it can be
# fed into featureFormat and exported at the end of the script.
def reconstruct_dict_data(df, features_list, people):
    data_dict = {}
    for index, row in normalised_df.iterrows():
        data_dict[people[index]] = row[features_list].to_dict()
    return data_dict

my_dataset = reconstruct_dict_data(good_data, features_list, people)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# Gaussian Process
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
#                                               ExpSineSquared, DotProduct,
#                                               ConstantKernel)
# gp_clf = GaussianProcessClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()


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

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

evaluate_feature_selection = False

if evaluate_feature_selection:
    # This code was used to score and evaluate features during the feature selection
    # portion of this project.
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    feature_selector = SelectKBest(f_classif, k='all').fit(features, labels)
    sorted_features = sorted(zip(feature_selector.scores_, features_list[1:]), reverse=True)
    print sorted_features

    import numpy as np
    import matplotlib.pyplot as plt

    # save the names and their respective scores separately
    # reverse the tuples to go from most frequent to least frequent
    feature_labels = zip(*sorted_features)[1]
    score = zip(*sorted_features)[0]
    x_pos = np.arange(len(sorted_features))
    plt.bar(x_pos, score, align='center')
    plt.xticks(x_pos, feature_labels)
    plt.ylabel('Features with f_classif Score')
    plt.show()

train_and_evaluate_classifier = True

if train_and_evaluate_classifier:
    # Grid Search Paramters for Gaussian Process Classifier
    # The Kernels were copied from:
    #  http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py
    # kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
    #            1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
    #            1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
    #                                 length_scale_bounds=(0.1, 10.0),
    #                                 periodicity_bounds=(1.0, 10.0)),
    #            ConstantKernel(0.1, (0.01, 10.0))
    #                * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) ** 2),
    #            1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
    #                         nu=1.5)]
    #
    # search_parameters = {
    #     'warm_start': [True, False],
    #     'kernel': kernels,
    #     'max_iter_predict': np.logspace(1.0, 3.0, num=20).astype(int)
    # }

    # Grid Search Paramters for the Random Forest Classifier
    search_parameters = {
        # Search 1

        # 'max_depth': np.logspace(0.5, 2, num=5).astype(int),
        # 'n_estimators': np.logspace(0.5, 2, num=5).astype(int),
        # 'max_features': range(1, len(features_train[0])+1, 2)

        # training time: 3026.012 s
        # {'max_features': 3, 'n_estimators': 3, 'max_depth': 42}
        # prediction time: 0.002 s
        # accuracy_score: 0.97619047619
        # matthews_corrcoef: 0.900450337781
        # f1_score: 0.909090909091
        # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #         max_depth=42, max_features=3, max_leaf_nodes=None,
        #         min_impurity_split=1e-07, min_samples_leaf=1,
        #         min_samples_split=2, min_weight_fraction_leaf=0.0,
        #         n_estimators=3, n_jobs=1, oob_score=False, random_state=None,
        #         verbose=0, warm_start=False)
    	# Accuracy: 0.86993	Precision: 0.32046	Recall: 0.20785	F1: 0.25216	F2: 0.22357
    	# Total predictions: 14000	True positives:  307	False positives:  651	False negatives: 1170	True negatives: 11872



        # Search 2

        'max_depth': [35,37,39,40,41,42,43,44,45,47,49],
        'n_estimators': [1,2,3,4,5,6,7],
        'max_features': [1, 2, 3, 4, 5, 6, 7, 8]

        # training time: 1160.145 s
        # {'max_features': 4, 'n_estimators': 1, 'max_depth': 40}
        # prediction time: 0.001 s
        # accuracy_score: 0.952380952381
        # matthews_corrcoef: 0.772972972973
        # f1_score: 0.8
        # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #         max_depth=40, max_features=4, max_leaf_nodes=None,
        #         min_impurity_split=1e-07, min_samples_leaf=1,
        #         min_samples_split=2, min_weight_fraction_leaf=0.0,
        #         n_estimators=1, n_jobs=1, oob_score=False, random_state=None,
        #         verbose=0, warm_start=False)
    	# Accuracy: 0.84143	Precision: 0.25414	Recall: 0.25999	F1: 0.25703	F2: 0.25879
    	# Total predictions: 14000	True positives:  384	False positives: 1127	False negatives: 1093	True negatives: 11396
    }

    ssscv = StratifiedShuffleSplit(labels, 100)
    gs = GridSearchCV(rf_clf, search_parameters, cv=ssscv, scoring="f1", verbose=2)

    # The best I was able to achive with the Gaussian Process Classifier
    # clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    # Accuracy: 0.85471	Precision: 0.35833	Recall: 0.02150	F1: 0.04057	F2: 0.02648
    # Total predictions: 14000	True positives:   43	False positives:   77	False negatives: 1957	True negatives: 11923

    t0 = time()
    gs.fit(features, labels)
    print "training time:", round(time()-t0, 3), "s"

    clf = gs.best_estimator_
    print gs.best_params_

    t0 = time()
    y_pred = clf.predict(features_test)
    print "prediction time:", round(time()-t0, 3), "s"

    from sklearn.metrics import accuracy_score
    # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import f1_score
    print "accuracy_score: {}".format(accuracy_score(labels_test, y_pred))
    print "matthews_corrcoef: {}".format(matthews_corrcoef(labels_test, y_pred))
    print "f1_score: {}".format(f1_score(labels_test, y_pred))

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    dump_classifier_and_data(clf, my_dataset, features_list)
