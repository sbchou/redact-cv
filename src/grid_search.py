"""
Grid Search 
"""
import redactometer
import pandas
import itertools
import time

def gridsearch(param_grid, train, img_root, censor_type="dark"):
    """Simple paramter training for redactometer. Not as modular as it could be, and
    also could be more efficient.

    Parameters
    ----------
    param_grid : dictionary of possibly paramater values
    train : training data, a Pandas DataFrame

    Returns
    ------
    argmax: a list, in case we have multiple equivalent optimal paramater combos
    max_score: maximum correctness score from best parameters

    Example
    -------
    See test function below

    """
    #no cross-val simple param setting
    train = train[train.redaction_type == censor_type]

    combos = [{key: value for (key, value) in zip(param_grid, values)} 
             for values in itertools.product(*param_grid.values())]

    #I believe this is not the efficient way. (Dataframes should not be iterated)

    max_score = 0
    argmax = []

    for combo in combos:
        #are we able to get the right number of blobs?
        correct = sum(len(redactometer.censor_dark(img_root + i, **combo)[1]) \
                == train.ix[i]['total_censor'] for i in train.index)
       #score = percent correct
        score = correct/len(train[train['redaction_type'] == censor_type])
        print score
        if score >= max_score:
            max_score = score
            argmax.append(combo)
    print time.time()            
    return argmax, max_score     

def complete_search(training, img_dir, numsteps):
    #range is inclusive
    param_range = [x/float(numsteps) for x in range(numsteps + 1)]
    param_grid = {'min_width_ratio' : param_range,
                'max_width_ratio': param_range,
                'min_height_ratio': param_range,
                'max_height_ratio': param_range}
    return gridsearch(param_grid, training, img_dir)

def test_complete():
     training = pandas.DataFrame.from_csv('../data/train_dark.csv', sep="\t")
     img_dir = "../train_dark/"
     return complete_search(training, img_dir, 5)

def test():
    param_grid = {'min_width_ratio': [0.01, 0.05, 0.10, 0.20], 
                'max_width_ratio': [0.80, 0.90, 0.95, 1.0], 
                'min_height_ratio': [0.01, 0.05, 0.10, 0.20],
                'max_height_ratio': [0.80, 0.90, 0.95, 1.0]}   
    train = pandas.DataFrame.from_csv('../data/train_dark.csv', sep="\t")
    return gridsearch(param_grid, train, "../train_dark/") 
