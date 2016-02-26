#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################

### The highest accuracy we were able to get using any of these algorithms
### (Naive Bayes, SVM, Decision Tree, AdaBoost, Random Forest, KNN) was 93.6%
### Here's a fun challenge question: can you beat us? If you can, write what you
### did (algorithm+parameters) in the box.


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

from time import time
from sklearn.metrics import accuracy_score

# KNN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4)

# adaboost
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(n_estimators=100)

# Random Forest
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)


t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
y_pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

print accuracy_score(labels_test, y_pred)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
