import json
from scipy import stats
import pprint
import numpy.testing as npt
import numpy as np
from utils import load_json

def concat_dictionaries(d1, d2):
    for k, v in d2.items():
        if k in d1:
            d1[f'{k}_2'] = v
        else:
            d1[k] = v
    return d1

def naive_eval(dict):
    """
    Returns the number of occurences where image n got the n:th best score in the dict.

    Works for descending and ascending score systems.
    """
    asc = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}
    des = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)}

    asc_correct = 0
    des_correct = 0
    i = 0
    for k, _ in asc.items():
        if k == f"{i}.png":
            asc_correct += 1
        i += 1
    i = 0
    for k, _ in des.items():
        if k == f"{i}.png":
            des_correct += 1
        i += 1
    return max(asc_correct, des_correct) / len(dict.items())

def output_results(path, output_path):
    results = load_json(path)
    
    model_results = {}
    for k, v in results.items():
        share = naive_eval(v)
        model_results[k] = share

    sorted_model_results = {k: v for k, v in sorted(model_results.items(), key=lambda item: item[1], reverse=True)}
    
    with open(output_path, "w") as f:
        json.dump(sorted_model_results, f, indent=4)

def concatenate_scores_files(path_1, path_2, output_path):
    f1 = load_json(path_1)
    f2 = load_json(path_2)
    combined = concat_dictionaries(f1, f2)
    sorted_model_results = {k: v for k, v in sorted(combined.items(), key=lambda item: item[1], reverse=True)}

    with open(output_path, "w") as f:
        json.dump(sorted_model_results, f, indent=4)
    
def concatenate_results_files(path_1, path_2, output_path):
    f1 = load_json(path_1)
    f2 = load_json(path_2)

    for k, v in f2.items():
        if k in f1:
            f1[f'{k}_2'] = v
        else:
            f1[k] = v

    with open(output_path, "w") as f:
        json.dump(f1, f, indent=4)

def rank_score(path, output_path, scipy_metric):
    """
    Inputs a path to the file with the raw results,
    a path to where the rank scores should end up, and
    a metric (spearman rank correlation coefficient, kendalltau).

    NOTE: Spearman rank correlation coefficient requires >500 samples to be accurate.
    Read more at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    """
    d = load_json(path)
    kendall = {}
    asc = [i for i in range(100)]
    desc = [i for i in range(99, -1, -1)]
    for metric, values in d.items():
        sorted_values = {k: v for k, v in sorted(values.items(), key=lambda item: item[1], reverse=True)}
        keys = sorted_values.keys()
        ranking = map ((lambda s: int(s[:-4])), list(keys))
        ranking_list = list(ranking)
        if scipy_metric == 'kendalltau':
            asc_val = stats.kendalltau(x=asc, y=ranking_list)
            desc_val = stats.kendalltau(x=desc, y=ranking_list)
        elif scipy_metric == 'sprc':
            asc_val = stats.spearmanr(a=asc, b=ranking_list)
            desc_val = stats.spearmanr(a=desc, b=ranking_list)
        else:
            raise ValueError("You must select an existing metric!")
        
        npt.assert_allclose(np.abs(asc_val), np.abs(desc_val), atol=1e-3)

        kendall[metric] = asc_val
    pprint.pprint(kendall)

    with open(output_path, "w") as f:
        json.dump(kendall, f, indent=4)

def main():
    #paths = ['output/albumentations/foggy/foggy_results.json']
             #'output/albumentations/foggy/results.json']
    #paths = ['output/albumentations/rainy/other_metrics_results.json']
    #output_paths = ['output/albumentations/rainy/naive_eval_other_metrics.json']
    #output_paths = ['output/albumentations/foggy/foggy_naive_eval.json']
                    #'output/albumentations/foggy/naive_eval.json']
    #for i in range(len(paths)):
        #output_results(paths[i], output_paths[i])
    #kendalltau("output/albumentations/rainy/combined_results.json",
     #          "output/albumentations/rainy/rainy_kendall.json")
    rank_score('output/albumentations/rainy/combined_results.json',
         'output/albumentations/rainy/rainy_sprc.json', 'sprc')

if __name__ == '__main__':
    main()