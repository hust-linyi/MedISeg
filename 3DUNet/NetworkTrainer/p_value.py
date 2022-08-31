import numpy as np
import pandas as pd
import os
from scipy import stats

def t_test():
    result_dir = '/newdata/ianlin/Experiment/LIVER'
    baseline_fold = 'baseline'
    folds = [f for f in os.listdir(result_dir) if f != baseline_fold]
    folds.sort()
    result_baseline = pd.read_csv(os.path.join(result_dir, baseline_fold, 'test_results_all.csv')).values[:-1].flatten().tolist()
    for fold in folds:
        try:
            result = pd.read_csv(os.path.join(result_dir, fold, 'test_results_all.csv')).values[:-1].flatten().tolist()
            # p_value = stats.ttest_ind(result_baseline, result,equal_var=False)
        except:
            try:
                result = pd.read_csv(os.path.join(result_dir, fold, 'test_results_all.csv')).values[:-1].flatten().tolist()
            except:
                continue
        p_value = stats.ttest_rel(result_baseline, result)
        print(fold, p_value)

t_test()