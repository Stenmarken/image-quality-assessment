import json
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

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

def main():
    paths = ['output/albumentations/foggy/results.json',
             'output/albumentations/rainy/results.json']
    output_paths = ['output/albumentations/foggy/naive_eval.json',
                    'output/albumentations/rainy/naive_eval.json']
    
    for i in range(len(paths)):
        output_results(paths[i], output_paths[i])

if __name__ == '__main__':
    main()