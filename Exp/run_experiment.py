"""
Runs an experiment: searches for hyperparameters and then trains the final model multiple times
"""

import argparse
import os
import glob
import json

import yaml
from sklearn.model_selection import ParameterGrid
import numpy as np

from Exp.run_model import run 
from Misc.config import RESULTS_PATH

relevant_keys = ["loss_train", "loss_val", "loss_test", "result_val", "result_test", "runtime_hours", "epochs", "runtime_hours"] 

def tune_params(args, hyperparams_path, errors_path):
    with open(args.grid_file, 'r') as file:
        grid_raw = yaml.safe_load(file)
    grid = ParameterGrid(grid_raw)
    print(f"Grid contains {len(grid)} combinations")

    prev_params = []
    mode = None
    previously_tested_params = glob.glob(os.path.join(hyperparams_path, "*.json"))
    nr_tries = args.candidates - len(previously_tested_params)
    for c in range(nr_tries):
        # Set seed randomly, because model training sets seeds and we want different parameters every time
        np.random.seed()

        param = np.random.choice(grid, 1)
        while param in prev_params:
            param = np.random.choice(grid, 1)

        prev_params.append(param)
        param_dict = {
            "--dataset": args.dataset
                    }
        for key, value in param[0].items():
            param_dict["--" + str(key)] = str(value)

        try:
            result_dict = run(param_dict)

            output_path = os.path.join(hyperparams_path, f"params_{len(glob.glob(os.path.join(hyperparams_path, '*')))}.json")
            storage_dict = {"params": param_dict,
                "result_val": result_dict["result_val"]
                }

            if mode is None:
                mode = result_dict["mode"]
            with open(output_path, "w") as file:
                json.dump(storage_dict, file, indent=4)
            
            print(output_path)

            if len(prev_params) >= len(grid):
                break
        except Exception as e:
            # Detected an error: store the params and the error
            error_dict = {"params": param_dict, "error": str(e)}
            output_path = os.path.join(errors_path, f"params_{len(glob.glob(os.path.join(errors_path, '*')))}.json")
            with open(output_path, "w") as file:
                json.dump(error_dict, file, indent=4)

    return mode

def main():
    parser = argparse.ArgumentParser(description='An experiment.')
    parser.add_argument('-grid', dest='grid_file', type=str,
                    help="Path to a .yaml file that contains the parameters grid.")
    parser.add_argument('-dataset', type=str)
    parser.add_argument('--candidates', type=int, default=20,
                    help="Number of parameter combinations to try per fold.")
    parser.add_argument('--repeats', type=int, default=10,
                    help="Number of times to repeat the final model training")
    parser.add_argument('--max_time', type=float, default=0,
                    help="The maximum amount of time to train for, in hours (default: 0, meaning that it gets ignored)")
    args = parser.parse_args()
    

    benchmark_runtime = args.max_time != 0



    if benchmark_runtime:
        results_path = os.path.join(RESULTS_PATH, f"RT_{args.dataset}_{os.path.split(args.grid_file)[-1]}") 
    else:
        results_path = os.path.join(RESULTS_PATH, f"{args.dataset}_{os.path.split(args.grid_file)[-1]}") 

    hyperparams_path = os.path.join(results_path, "Hyperparameters")
    final_eval_path = os.path.join(results_path, "FinalResults")
    errors_path = os.path.join(results_path, "Errors")

    if not os.path.isdir(results_path):
        os.mkdir(results_path)
        os.mkdir(hyperparams_path)
        os.mkdir(final_eval_path)
        os.mkdir(errors_path)

    if not benchmark_runtime:
        print("Searching for hyperparameters")
        mode = tune_params(args, hyperparams_path, errors_path)

        print("Finished search")
        # Select best parameters
        best_params, best_result_val = None, None
        for param_file in glob.glob(os.path.join(hyperparams_path, "*.json")):
            with open(param_file) as file:
                dict = yaml.safe_load(file)

                if best_result_val is None or \
                        (mode == "min" and dict["result_val"] < best_result_val) or \
                        (mode == "max" and dict["result_val"] > best_result_val):
                    best_params = dict["params"]
                    best_result_val = dict["result_val"]

        print(f"Best params have score {best_result_val:.4f} and are:\n{best_params}")
        print("Evaluating the final params")
    else:
        with open(args.grid_file, 'r') as file:
            best_params = yaml.safe_load(file)


    prev_evaluation = glob.glob(os.path.join(final_eval_path, "*.json"))
    nr_tries = args.repeats - len(prev_evaluation)
    for iteration in range(nr_tries):
        print(iteration)
        best_params["--seed"] = iteration
        if benchmark_runtime:
            best_params["--max_time"] = args.max_time
            best_params["--dataset"] = args.dataset

        result_dict = run(best_params)

        output_dict = {}
        for key in relevant_keys:
            output_dict[key] = result_dict[key]
        output_dict["params"] = best_params

        output_path = os.path.join(final_eval_path, f"eval_{len(glob.glob(os.path.join(final_eval_path, '*')))}.json")
        print(output_path)
        with open(output_path, "w") as file:
            json.dump(output_dict, file, indent=4)
    
    eval_paths = list(glob.glob(os.path.join(final_eval_path, '*.json')))
    final_results = {}
    for key in relevant_keys:
        final_results[key] = []

    for eval_path in eval_paths:
        with open(eval_path) as file:
            result_dict = json.load(file)
        for key in final_results.keys():
            final_results[key].append(result_dict[key])

    output = {}
    for key in final_results.keys():
        avg, std = np.average(final_results[key]), np.std(final_results[key])
        output[f"{key}-avg"] = avg
        output[f"{key}-std"] = std
        print(f"{key}:\t{avg:.4f}±{std:4f}")

    output_path = os.path.join(results_path, "final.json")  
    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

if __name__ == "__main__":
    main()