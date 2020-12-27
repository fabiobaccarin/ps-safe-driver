from sklearn import linear_model as lm
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes as nb
from lightgbm import LGBMClassifier

MAX_ITER = 1e4

MODELS = [
    (
        'logistic_regression',
        lm.LogisticRegression(
            penalty='none',
            random_state=0,
            solver='saga',
            max_iter=MAX_ITER
        ),
        None,
    ),
    (
        'ridge',
        lm.LogisticRegression(
            penalty='l2',
            random_state=0,
            solver='saga',
            max_iter=MAX_ITER
        ),
        {'clf__C': [1e-3, 1e-2, 1e-1, 1e0]},
    ),
    (
        'lasso',
        lm.LogisticRegression(
            penalty='l1',
            random_state=0,
            solver='saga',
            max_iter=MAX_ITER
        ),
        {'clf__C': [1e-3, 1e-2, 1e-1, 1e0]},
    ),
    (
        'elastic_net',
        lm.LogisticRegression(
            penalty='elasticnet',
            random_state=0,
            solver='saga',
            max_iter=MAX_ITER
        ),
        {
            'clf__C': [1e-3, 1e-2, 1e-1, 1e0],
            'clf__l1_ratio': [0.2, 0.5, 0.8]
        },
    ),
    (
        'decision_tree',
        DecisionTreeClassifier(random_state=0),
        None,
    ),
    (
        'naive_bayes_bernoulli',
        nb.BernoulliNB(),
        None,
    ),
    (
        'naive_bayes_gaussian',
        nb.GaussianNB(),
        None,
    ),
    (
        'light_gbm',
        LGBMClassifier(random_state=0),
        {
            'clf__boosting_type': ['gbdt', 'dart', 'goss'],
            'clf__n_estimators': [100, 200, 300]
        },
    ),
]