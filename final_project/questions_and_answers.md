# Identify Fraud from Enron Email
## Introduction to Machine Learning Project
#### Data Analyst Nanodegree (Udacity)
Project submission by Edward Minnett (ed@methodic.io).

February 11th 2017. (Revision 1)

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

I found 5 outliers in the data ('FREVERT MARK A', 'LAVORATO JOHN J', 'LAY KENNETH L', 'BELDEN TIMOTHY N', 'SKILLING JEFFREY K'). My process for finding them was to analyse the people who fell outside of the interquartile range for each of my selected features. If an individual was an outlier for more than a quarter of the selected features, I considered them as an overall outlier and excluded them from the data. Selecting a threshold of a quarter of the features was a bit arbitrary, but after some trial and error, it felt like a good compromise. It isn't surprising that Jeff Skilling and Ken Lay were considered outliers as the two people who ran Enron, their data was far from typical for the dataset.

### Question 2

What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

**Answer:** I selected 12 of the original features `'exercised_stock_options', 'total_stock_value', 'deferred_income', 'salary', 'bonus', 'from_poi_to_this_person', 'expenses', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'director_fees', 'deferral_payments'`. My method of selection was to pass all of the features and data through `SelectKBest(f_classif, k='all').fit(features, labels)`, mapped the scores back to the feature labels and selected all of the features that ha d a score greater than 1. In addition to these features, I created 5 new ones. All 5 new features, `'to_poi_to_the_from_poi_power', 'from_poi_to_the_to_poi_power', 'expenses_to_the_bonus_power', 'lti_times_expenses_to_the_bonus_power', 'salary_times_expenses_to_the_bonus_power'` were either one feature to the power of another feature or one feature multiplied by another feature to the power of a third feature. My reasoning for engineering new features in this manner is that not only should these new features be linearly indepent of the original features, but they should also amplify the differences between POI and NON-POI records. These features as well as many other attempts were created by trial and error. I only kept the features that received a score from `SelectKBest` greater than 7. The highest score was `21.37` for `lti_times_expenses_to_the_bonus_power`. The following is a sorted list from greatest to smallest of the selected and engineered feature scores (rounded to 2 decimal places) as given by `SelectKBest(f_classif, k='all').fit(features, labels)`.

- **lti_times_expenses_to_the_bonus_power**: 21.37
- **exercised_stock_options**: 14.58
- **total_stock_value**: 13.36
- **expenses_to_the_bonus_power**: 11.69
- **deferred_income**: 11.18
- **to_poi_to_the_from_poi_power**: 10.76
- **salary**: 9.00
- **salary_times_expenses_to_the_bonus_power**: 8.65
- **from_poi_to_the_to_poi_power**: 7.47
- **bonus**: 6.62
- **expenses**: 4.03
- **from_poi_to_this_person**: 3.99
- **from_this_person_to_poi**: 3.94
- **long_term_incentive**: 2.62
- **shared_receipt_with_poi**: 1.80
- **director_fees**: 1.70
- **deferral_payments**: 1.13

Once the new features were created and added to the dataset, there were all scaled and normalised. The data for each feature was first scaled to a range from 0 to 1 inclusive and then normalised using the SoftMax expression to ensure all the features have the same mean value. This process helps ensure that gradients and distances between records can be calculated more consistently while training and evaluating the machine learning algorithm.

### Question 3

What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

**Answer:** I ended up training and evaluate a Random Forest Classifier. I initially tried KKN, AdaBoost, Decision Tree, and Gaussian Process classifiers as well. From the first exploration, I settled on exploring the Random Forest and Gaussian Process classifiers to a greater depth. From the initial exploration, they performed the best with the smallest compromise between classification power and training time. I also felt they would be well suited to utilising the non-linear engineered features. Using all of these classifiers with their default settings resulted in varying degrees of performance in terms of precision, but they all had quite poor recall.

I put both Random Forest and Gaussian Process classifiers through moor rigorous testing using `GridSearchCV` with `StratifiedShuffleSplit` cross validation with 100 folds.

The Gaussian Process classifier managed to achieve a precision of `0.35833` but only a recall of `0.02150`. Even though I could get the precision above 0.3 when running `tester.py` I couldn't get the recall anywhere close 0.3. This was achieved setting `warm_start = true` and using the RBF kernel set to `1.0 * RBF(1.0)`.

I managed to achieve better recall performance from the Random Forest classifier though it was at the sacrifice of some precision. Multiple iterations through `GridSearchCV` resulted in parameters set to `{'max_features': 4, 'n_estimators': 1, 'max_depth': 40}`. This classifier received the following scores from `tester.py`: `Accuracy: 0.84143	Precision: 0.25414	Recall: 0.25999	F1: 0.25703	F2: 0.25879`. Other algorithms and parameters scored better for accuracy and precision, but this was the highest recall score I managed to achieve.

### Question 4

What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that
was not your final choice or a different model that does utilise parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

**Answer:** Machine Learning algorithms tend to be highly parameterised. The se parameters all the the practitioner to tailor the algorithm for a given problem and data set. Failing to properly tune the parameters of an algorithm  will most likely result in the algorithm under-fitting or overfitting the training data and in turn generalise poorly when confronted with new data. One strategy to perform hyper-parameter optimisation using grid search. This process takes a set of parameters each with a set of values and iteratively trains and evaluates each version of the algorithm. The grid search process will result in one set of parameters that performed the best out of all possible combinations of the given parameters and values.

As discussed in my answer to Question 3, I used the SKLearn `GridSearchCV` method for hyper-parameter optimisation. For the Random Forest Classifier, I ended up using grid search twice to optimise the classifier parameters. The first iteration used quite broad values for `max_depth`, `n_estimators`, and `max_features`. I then took the selected parameters for the first search and narrowed the values to try and hone in on the best combination of values. The end result left me with the following parameters for the Random Forest Classifier: `{'max_features': 4, 'n_estimators': 1, 'max_depth': 40}`.

### Question 5

What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

**Answer:** Validation when training a machine learning algorithm refers to the act of splitting a fraction of the training data set to use as a validation data set. The validation set acts like the test data set and allows you to evaluate the performs of the algorithm while training it without using the test data set.  A common and potentially disastrous mistake is to use the test data set for validation. If you do this and allow the test data set to influence the training process, the algorithm will 'learn' about the test data through the choices you make while training the algorithm and will most likely overfit that test data leading it to generalise badly when confronted with new data.

The `CV` in `GridSearchCV` stands for 'Cross Validation'. Cross validation is an approach to validation in machine learning where the training data is split into a certain number of 'folds'. For each fold, a fraction of the data equal to the training data count divided by the number of folds (1/10th for the case of 10 folds). For each fold, the data is trained on the non-validation portion of the data and then evaluated on the validation portion. This process is repeated for each fold allowing the algorithm to learn from subsequent iterations. This process helps the algorithm avoid  overfitting on the training data.

I used the `StratifiedShuffleSplit` module in SKLearn to handle cross validation when I performed grid search. This stratified shuffle split has the benefit of avoiding random bias within the dataset when the proportion of classes is unbalanced. For `StratifiedShuffleSplit`, "folds are made by preserving the percentage of samples for each class" (as stated in the SKLearn documentation).

### Question 6

Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

**Answer:** In the end, I used 3 metrics for evaluating the trained algorithms: accuracy, F1 Score, and Mathew's Correlation Coefficient.

Accuracy is a useful score to understand the general performance of an algorithm though it can be misleading if the target classes are unequally distributed in a dataset. For example, 12.4% POI records and 87.6% NON-POI records. If an algorithm was very naive and marked all records a NON-POI, it would have an accuracy score of 0.876 which looks like a good outcome, but in this situation it really isn't.

The F1 Score is a weighted proportion of precision and recall. More specifically, it is `2 * precision * recall / (precision + recall)`. Precision and recall can have values from 0 to 1 which means the F1 score does as well. This is a much more useful score when determining the performance of an algorithm trying to classify data with unbalanced classes.

Mathew's Correlation Coefficient measures the quality of binary classifications. It takes a value between -1 and 1 where 1 indicates perfect prediction, 0 represents no better than random prediction, and -1 indicates predicting every classification incorrectly.

The Random Forest Classifier returned by the first grid search (with parameters set to `{'max_features': 3, 'n_estimators': 3, 'max_depth': 42}`) had received the following evaluation scores when predicting the test data:

- **accuracy_score**: 0.97619047619
- **f1_score**: 0.909090909091
- **matthews_corrcoef**: 0.900450337781

This was far better than I expected. That classifier received the following results from `tester.py`.

- **Accuracy**: 0.86993
- **Precision**: 0.32046
- **Recall**: 0.20785
- **F1**: 0.25216
- **F2**: 0.22357

A precision score above 0.3 but recall of only 0.207. I thought another grid search with narrower parameter values might improve the results. The second iteration resulted in parameters of `{'max_features': 4, 'n_estimators': 1, 'max_depth': 40}` which received the following scores against the test data set.

- **accuracy_score**: 0.952380952381
- **f1_score**: 0.8
- **matthews_corrcoef**: 0.772972972973

This algorithm performed worse against the test data set but had better recall when it was tested with `tester.py`.

- **Accuracy**: 0.84143
- **Precision**: 0.25414
- **Recall**: 0.25999
- **F1**: 0.25703		
- **F2**: 0.25879

It is hard to say whether precision or recall is more important when evaluating the performance an algorithm. It depends whether when applying the algorithm, false positives are more detrimental than false negatives, this when precision is more important, or vice versa in which case recall is more important.


## References

- http://scikit-learn.org
- http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py
- http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py
- http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
- http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
- https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
- https://en.wikipedia.org/wiki/Feature_scaling
- https://en.wikipedia.org/wiki/Softmax_function
- https://github.com/eminnett/ml-nanodegree-student-intervention-system/blob/master/student_intervention.ipynb
- https://docs.scipy.org
- http://pandas.pydata.org/
- http://stats.stackexchange.com/questions/107874/how-to-deal-with-a-skewed-class-in-binary-classification-having-many-features
- https://discussions.udacity.com/t/why-precision-recall-value-is-so-low/187489/2

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.
