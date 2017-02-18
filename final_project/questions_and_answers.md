# Identify Fraud from Enron Email
## Introduction to Machine Learning Project
#### Data Analyst Nanodegree (Udacity)
Project submission by Edward Minnett (ed@methodic.io).

February 18th 2017. (Revision 2)

----------

# Enron Submission Free-Response Questions

*(From the project description)*

A critical part of machine learning is making sense of your analysis process and communicating it to others. The questions below will help us understand your decision-making process and allow us to give feedback on your project. Please answer each question; your answers should be about 1-2 paragraphs per question. If you find yourself writing much more than that, take a step back and see if you can simplify your response!

When your evaluator looks at your responses, he or she will use a specific list of rubric items to assess your answers. Here is the link to that rubric: Link to the rubric Each question has one or more specific rubric items associated with it, so before you submit an answer, take a look at that part of the rubric. If your response does not meet expectations for all rubric points, you will be asked to revise and resubmit your project. Make sure that your responses are detailed enough that the evaluator will be able to understand the steps you took and your thought processes as you went through the data analysis.

Once you’ve submitted your responses, your coach will take a look and may ask a few more focused follow-up questions on one or more of your answers.  

We can’t wait to see what you’ve put together for this project!

### Question 1

Summarise for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

**Answer:** The goal of this project is to take the Enron email and financial data, including the 'Person of Interest' or 'POI' classification, and attempt to train a classifier to find people in the dataset who should be classified as a 'POI'. This problem is well suited to be solved using supervised machine learning. Not only do we have structured and labeled data to help us solve this problem, but it is also a classification problem. Both of these traits point to the use of supervised learning where we can take a corpus of training data and train a classifier to help us classify people of interest within the data.

The dataset consists of 145 records each with 20 features (including 'POI'). Out of 145 records, 18 are classified as POI and 127 are not. Out of the 20 features, there are 7 that are 0 for at least half of the records. (deferral_payments, restricted_stock_deferred, loan_advances, from_this_person_to_poi, director_fees, deferred_income, long_term_incentive).

There was only one outlier that I excluded from the data and that was the 'TOTAL' row. This row is clearly unhelpful as it is the sum of all of the features for the entire data set. Although I tried performing automatic outlier detection on the remaining data, I found this process to be counter productive. Several of the outliers that were detected were POI records. Given that there are so few POI records to begin with, removing some of these would only make the classification problem harder.

### Question 2

What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

**Answer:** I selected 4 of the original features and added one of the 5 engineered features I created. These are the 5 features along with their score from `SelectKBest`:

- **exercised_stock_options**: 26.69
- **total_stock_value**: 24.38
- **salary**: 18.21
- **bonus**: 17.35
- **to_poi_to_the_from_poi_power**: 15.42

All 5 engineered features, `'to_poi_to_the_from_poi_power', 'from_poi_to_the_to_poi_power', 'expenses_to_the_bonus_power', 'lti_times_expenses_to_the_bonus_power', 'salary_times_expenses_to_the_bonus_power'` were either one feature to the power of another feature or one feature multiplied by another feature to the power of a third feature. My reasoning for engineering new features in this manner is that not only should these new features be linearly indepent of the original features, but they should also amplify the differences between POI and NON-POI records. These features as well as many other attempts were created by trial and error.

Once the new features were created and added to the dataset, they were all scaled and normalised. The data for each feature was first scaled to a range from 0 to 1 inclusive and then normalised using the SoftMax expression to ensure all the features have the same mean value. This process helps ensure that gradients and distances between records can be calculated more consistently while training and evaluating the machine learning algorithm. This is particularly important for algorithms that calculate distances between points such as K Nearest Neighbours. The performance of classifiers with and without feature scaling and normalisation will be shown in the answer to question 3.

Once I completed this step, I iterated through all values of k from 1 to the total number of features under consideration. For each value of k, I trained a decision tree classifier using its default settings. With each classifier, I trained it using 70% of the data (training data) and then predicted the classifications for the remaining 30% (test data). From these predictions, I calculated the precision and recall scores as well as determined which of the features were selected as the k best for that iteration.

The following plot of the precision and recall scores for each value of k shows that the ideal number of features was 5.

![feature_selection_plot](https://github.com/eminnett/da-nanodegree-intro-to-machine-learning/blob/master/final_project/assets/feature_selection_plot.png "Precision and Recall Scores for each k best features")

The 5 features and their scores from `SelectKBest` are displayed at the beginning of this answer.

### Question 3

What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

**Answer:** I ended up using a Decision Tree classifier though I managed to train a K Nearest Neighbour classifier to perform equally well against the test data. I will explain why the Decision Tree was my classifier of choice in my answer to question 6. Initially, I compared the performance of KNN, SVM, Decision Tree, Random Forest, AdaBoost, and Gaussian Naive Bayes classifiers. All 6 were tested with and without feature scaling and normalisation using their default parameters. These were the results:

- FS & N = T:  (With Feature Scaling and Normalisation)
- FS & N = F: (Without Feature Scaling and Normalisation)

|                        | FS & N = T    | FS & N = T | FS & N = T   | FS & N = F     | FS & N = F  | FS & N = F    |
|------------------------|---------------|------------|--------------|----------------|-------------|---------------|
| Classifier             | Precision     | Recall     | F1 Score     | Precision      | Recall      | F1 Score      |
| K Nearest Neighbours   | 1.0           | 0.25       | 0.4          | 1.0            | 0.25        | 0.4           |
| Support Vector Machine | 0.0           | 0.0        | 0.0          | 0.0            | 0.0         | 0.0           |
| Decision Tree          | 0.667         | 0.5        | 0.5714       | 0.333          | 0.5         | 0.4           |
| Random Forest          | 0.0           | 0.0        | 0.0          | 0.0            | 0.0         | 0.0           |
| AdaBoost               | 0.25          | 0.25       | 0.25         | 0.25           | 0.25        | 0.25          |
| Gaussian Naive Bayes   | 0.5           | 0.5        | 0.5          | 0.0            | 0.0         | 0.0           |

These results clearly show that most of the classifiers benefitted from feature scaling and normalisation. I am surprised that KNN performed equally well under both conditions as I would have expected to suffer from a lack of feature scaling and normalisation.

From these results, I chose the K Nearest Neighbours and Decision Tree classifiers to explore further. Even though Gaussian Naive Bayes performed reasonably well in this first test, it doesn't have any parameters that can be tuned using hyper-parameter optimisation.

I put both classifiers through more rigorous testing using `GridSearchCV` with `StratifiedShuffleSplit` cross validation with 100 folds.

Both classifiers were able to classify the 30% test data set with precision and recall scores of 1.0 after performing the stratified shuffle split cross validation with grid search. This was quite a surprise as I wasn't expecting such extreme results for both classifiers.

The two resulting classifiers with their parameters were as follows:

```
DecisionTreeClassifier(max_features = 3, criterion = 'gini')
```
and
```
KNeighborsClassifier(n_neighbors = 1, algorithm = 'ball_tree')
```

Though they performed equally well agains the test data set, they didn't perform equally well when tested using `tester.py`. I will discuss this in greater detail in my answer to question 6.

### Question 4

What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that
was not your final choice or a different model that does utilise parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

**Answer:** Machine Learning algorithms tend to be highly parameterised. These parameters allow the practitioner to tailor the algorithm for a given problem and data set. Failing to properly tune the parameters of an algorithm will most likely result in the algorithm under-fitting or over-fitting the training data and in turn generalise poorly when confronted with new data. One strategy to perform hyper-parameter optimisation using grid search. This process takes a set of parameters each with a set of values and iteratively trains and evaluates each version of the algorithm. The grid search process will result in one set of parameters that performed the best out of all possible combinations of the given parameters and values.

As discussed in my answer to Question 3, I used the SKLearn `GridSearchCV` method for hyper-parameter optimisation for both the K Nearest Neighbour and Decision Tree classifiers.

For the K Nearest Neighbour classifier, I optimised the number of neighbours and the 'algorithm' parameter for the algorithm with the following options:

- **'algorithm'**: ['ball_tree', 'kd_tree', 'brute']
- **'n_neighbors'**: [1, 2, 3, 4, 5]

GridSearchCV found the optimal parameters to be the `ball_tree` algorithm and an `n_neighbors` values of 1.

For the Decision Tree classifier, I optimised the maximum features used by the classifier and the 'criterion' parameter with the following options:

- **'criterion'**: ['gini', 'entropy']
- **'max_features'**: [1, 2, 3, 4, 5] (max_features can't exceed the total number of features and there were only 5 features in the final data set)

GridSearchCV found the optimal parameters to be the `gini` criterion and a `max_features` value of 3.

### Question 5

What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

**Answer:** Validation when training a machine learning algorithm refers to the act of splitting a fraction of the training data set to use as a validation data set. The validation set acts like the test data set and allows you to evaluate the performs of the algorithm while training it without using the test data set. A common and potentially disastrous mistake is to use the test data set for validation. If you do this and allow the test data set to influence the training process, the algorithm will 'learn' about the test data through the choices you make while training the algorithm and will most likely overfit that test data leading it to generalise badly when confronted with new data.

The `CV` in `GridSearchCV` stands for 'Cross Validation'. Cross validation is an approach to validation in machine learning where the training data is split into a certain number of 'folds'. For each fold, a fraction of the data equal to the training data count divided by the number of folds (1/10th for the case of 10 folds). For each fold, the data is trained on the non-validation portion of the data and then evaluated on the validation portion. This process is repeated for each fold allowing the algorithm to learn from subsequent iterations. This process helps the algorithm avoid  overfitting on the training data.

I used the `StratifiedShuffleSplit` module in SKLearn to handle cross validation when I performed grid search. This stratified shuffle split has the benefit of avoiding random bias within the dataset when the proportion of classes is unbalanced. For `StratifiedShuffleSplit`, "folds are made by preserving the percentage of samples for each class" (as stated in the SKLearn documentation).

### Question 6

Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

**Answer:** In the end, I used 5 metrics for evaluating the trained algorithms: accuracy, precision, recall, F1 Score, and Mathew's Correlation Coefficient.

Accuracy is a useful score to understand the general performance of an algorithm though it can be misleading if the target classes are unequally distributed in a dataset. For example, 12.4% POI records and 87.6% NON-POI records. If an algorithm was very naive and marked all records a NON-POI, it would have an accuracy score of 0.876 which looks like a good outcome, but in this situation it really isn't.

The precision score is the number of correctly classified positive records divided by the total number of records labeled as positive. If we think of this score in terms of medical diagnostics, the precision score represents the proportion of all people diagnosed with a condition who actually had the condition.

The recall score is the number of correctly classified negative records divided by the total number of records labeled as negative. If we think of this score in terms of medical diagnostics, the recall score represents the proportion of all people diagnosed not to have a condition who actually did not have the condition.

The F1 Score is harmonic mean of precision and recall. More specifically, it is `2 * precision * recall / (precision + recall)`. Precision and recall can have values from 0 to 1 which means the F1 score does as well. This is a much more useful score than accuracy when determining the performance of an algorithm trying to classify data with unbalanced classes.

Mathew's Correlation Coefficient measures the quality of binary classifications. It takes a value between -1 and 1 where 1 indicates perfect prediction, 0 represents no better than random prediction, and -1 indicates predicting every classification incorrectly.


Even though both the K Nearest Neighbour and Decision Tree classifiers received perfect scores (a value of 1.0 for all 5 scores discussed above) for the test data set when evaluating the optimised classifiers after performing grid search, they didn't both exceed the 0.3 threshold for precision and recall when evaluating the algorithms using tester.py.

The `tester.py` scores for `KNeighborsClassifier(n_neighbors = 1, algorithm = ball_tree')` were:

- **Accuracy**: 0.8092
- **Precision**: 0.26082
- **Recall**: 0.23500
- **F1**: 0.24724
- **F2**: 0.23975

The `tester.py` scores for `DecisionTreeClassifier(max_features = 1, criterion = 'gini')` were:

- **Accuracy**: 0.81967
- **Precision**: 0.33566
- **Recall**: 0.36000
- **F1**: 0.34741
- **F2**: 0.35485

Only these final results allowed me to choose `DecisionTreeClassifier(max_features = 1, criterion = 'gini')` as the higher performance classifier.


## References

- http://scikit-learn.org
- http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py
- http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
- http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
- http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
- https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
- https://en.wikipedia.org/wiki/Feature_scaling
- https://en.wikipedia.org/wiki/Softmax_function
- https://github.com/eminnett/ml-nanodegree-student-intervention-system/blob/master/student_intervention.ipynb
- https://docs.scipy.org
- http://pandas.pydata.org/
- https://seaborn.pydata.org/api.html
- http://stats.stackexchange.com/questions/107874/how-to-deal-with-a-skewed-class-in-binary-classification-having-many-features
- https://discussions.udacity.com/t/why-precision-recall-value-is-so-low/187489/2
- https://en.wikipedia.org/wiki/Harmonic_mean
- https://www.quora.com/When-should-you-perform-feature-scaling-and-mean-normalization-on-the-given-data-What-are-the-advantages-of-these-techniques
- https://www.quora.com/Is-feature-scaling-ever-bad
- http://matplotlib.org/

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.
