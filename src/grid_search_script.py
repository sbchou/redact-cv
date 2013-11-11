"""
Grid Search Script
"""
import redactometer
import pandas
import itertools

def gridsearch():
    data  = pandas.DataFrame.from_csv('../data/img_training.csv', sep="\t")
    #no cross-val as of yet, simply split data into train and test
    folds = 3
    test = data[0:len(data)/folds]
    train = data[len(data)/folds + 1:len(data)] #for now just one part

    param_grid = {'min_width_ratio': [0.01, 0.05, 0.10, 0.20], 
                'max_width_ratio': [0.80, 0.90, 0.95, 1.0], 
                'min_height_ratio': [0.01, 0.05, 0.10, 0.20],
                'max_height_ratio': [0.80, 0.90, 0.95, 1.0]}

    #[[{key: value} for (key, value) in zip(param_grid, values)] 
    #         for values in itertools.product(*param_grid.values())]


    combos = [{key: value for (key, value) in zip(param_grid, values)} 
             for values in itertools.product(*param_grid.values())]

    #first only do dark censor
    train_dark = train[train.redaction_type == 'dark']

    root = "../training_img/"

    #I believe this is not the efficient way. (Dataframes should not be iterated)

    max_score = 0
    argmax = []

    for combo in combos:
        #import pdb; pdb.set_trace()
        #just check binary
        score = sum(bool(redactometer.censor_dark(root + train_dark.ix[i]['img'], **combo)[1]) \
                == train_dark.ix[i]['censored'] for i in train_dark.index)
        #score = sum(len(redactometer.censor_dark(root + train_dark.ix[i]['img'], **combo)[1]) \
        #        == train_dark.ix[i]['num_redactions'] for i in train_dark.index)    
        print score
        if score >= max_score:
            max_score = score
            argmax.append(combo)
                
    return argmax, max_score     









