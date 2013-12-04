import pandas, redactometer, grid_search

testing = pandas.DataFrame.from_csv('../data/test_dark.csv', sep="\t")
training = pandas.DataFrame.from_csv('../data/train_dark.csv', sep="\t")

p, s = grid_search.train_grid('is_censored')

param_grid = {'min_width_ratio': [0.01, 0.05, 0.10, 0.20], 
            'max_width_ratio': [0.80, 0.85, 0.90, 0.95], 
            'min_height_ratio': [0.01, 0.05, 0.10, 0.20],
            'max_height_ratio': [0.80, 0.85, 0.90, 0.95]} 

#train
params, score = grid_search.gridsearch(param_grid, training, '../train_dark/', 'is_censored')

#test
results = [grid_search.eval(testing, p, '../train_dark/', 'is_censored') for p in params]