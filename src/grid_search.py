"""
Grid Search 
"""
import redactometer
import pandas
import itertools
import time
import numpy as np

def gridsearch(param_grid, train, img_root, metric, censor_type="dark"):
    """Lorem Ipsum"""
    #no cross-val simple param setting
    train = train[train.redaction_type == censor_type]

    combos = [{key: value for (key, value) in zip(param_grid, values)} 
             for values in itertools.product(*param_grid.values())]

    #I believe this is not the efficient way. (Dataframes should not be iterated)

    max_score = 0
    argmax = []
    total_img = len(train[train['redaction_type'] == censor_type])
    print "number img", total_img
    print "number combos", len(combos)

    for combo in combos:
        print "combo", combo
        #boolean
        if metric == 'is_censored':
            correct = sum((len(redactometer.censor_dark(img_root + i, **combo)[1]) > 0) \
               == train.ix[i]['is_censored'] for i in train.index)
            score = float(correct)/float(total_img)

        #are we able to get the right number of blobs?
        if metric == 'total_censor':
            correct = sum(len(redactometer.censor_dark(img_root + i, **combo)[1]) \
                == train.ix[i]['total_censor'] for i in train.index)
            score = float(correct)/float(total_img)

        #score = percent correct
        print "correct", correct
        print "score", score

        #update
        if score >= max_score:
            max_score = score
            argmax.append(combo) 
    return argmax, max_score     

def complete_search(training, img_dir, metric, start, stop, step):
    #range is inclusive
    param_range = np.arange(start, stop, step)
    param_grid = {'min_width_ratio' : param_range,
                'max_width_ratio': param_range,
                'min_height_ratio': param_range,
                'max_height_ratio': param_range}
    return gridsearch(param_grid, training, img_dir, metric)

def eval(test, params, img_root, metric, censor_type="dark"):
    """Evaluation results on test set"""
    score = None
    total_img = len(test[test['redaction_type'] == censor_type])

    if metric == "is_censored":
        correct = sum((len(redactometer.censor_dark(img_root + i, **params)[1]) > 0) \
           == test.ix[i]['is_censored'] for i in test.index)
        score = float(correct)/float(total_img)

    if metric == "total_censor":
        correct = sum(len(redactometer.censor_dark(img_root + i, **params)[1]) \
                    == test.ix[i]['total_censor'] for i in test.index)
        score = float(correct)/float(len(test[test['redaction_type'] == censor_type]))
    
    return score


def train_complete(metric):
     training = pandas.DataFrame.from_csv('../data/train_dark.csv', sep="\t")
     img_dir = "../train_dark/"
     return complete_search(training, img_dir, metric, 0.1, 0.9, 0.1)

def train_grid(metric):
    param_grid = {'min_width_ratio': [0.01, 0.05, 0.10, 0.20], 
                'max_width_ratio': [0.80, 0.85, 0.90, 0.95], 
                'min_height_ratio': [0.01, 0.05, 0.10, 0.20],
                'max_height_ratio': [0.80, 0.85, 0.90, 0.95]}   
    train = pandas.DataFrame.from_csv('../data/train_dark.csv', sep="\t")
    return gridsearch(param_grid, train, "../train_dark/", metric) 



