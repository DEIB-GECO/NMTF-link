import warnings
import sys
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import matplotlib
import time
import statistics
import os
from scripts import Network, bold

warnings.filterwarnings('ignore')
matplotlib.use('agg')

current = os.getcwd()
try:
    _, filename_1, filename_2 = sys.argv
except:
    print("Usage: NMTF-link input_folder configuration_file")
    sys.exit(1)

conf_file = os.path.join(current, filename_1, filename_2)
input_dir = os.path.join(current, filename_1)

with open(conf_file) as f:
    for line in f:
        if line.strip().startswith("#metric"):
            s = line.strip().split("\t")
            if s[1].upper() == "APS":
                metric = "aps"
            elif s[1].upper() == "AUROC":
                metric = "auroc"
            elif s[1].upper() == "PEARSON":
                metric = "pearson"
            elif s[1].upper() == "R2":
                metric = "r2"
            elif s[1].upper() == "RMSE":
                metric = "rmse"
            else:
                print(f"Metric '{s[1]}' not supported")
                exit(-1)
            print(f"\nEvaluation metric is {bold(metric.upper())}")

        elif line.strip().startswith("#number.of.iterations"):
            try:
                s = line.strip().split("\t")
                max_iter = int(s[1])
                print(f"Maximum number of iteration is {bold(str(max_iter))}")
            except ValueError:
                print("Number of iterations should be integer")
                exit(-1)
        elif line.strip().startswith("#stop.criterion"):
            s = line.strip().split("\t")
            if s[1].lower() == "maximum_metric":
                stop_criterion = 'maximum_metric'
            elif s[1].lower() == "maximum_iterations":
                stop_criterion = 'maximum_iterations'
            elif s[1].lower() == "relative_error":
                stop_criterion = 'relative_error'
            else:
                print(f"Option '{s[1]}' not supported")
                exit(-1)
        elif line.strip().startswith("#score.threshold"):
            s = line.strip().split("\t")
            if 0 <= float(s[1]) <= 1:
                threshold = float(s[1])
            else:
                print("Threshold for retrieval should be between 0 and 1")
                exit(-1)

metric_vals = np.zeros(max_iter // 10)

if stop_criterion == 'maximum_metric':
    best_iter = 0
    best_iter_arr = []  # contains the iterations with best performance from each of 5 validation runs (j cycle)
    # cycle to find the stop criterion value
    for j in range(5):
        V = []
        difference = []
        verbose = True if j == 0 else False

        network = Network(conf_file, input_dir, verbose)
        initial_error = network.get_error()
        print(bold(f"Run number {str(j + 1)}"))
        print(f"initial error: {initial_error}")
        for i in range(max_iter):
            network.update()
            V.append(network.validate(metric))
            if i % 10 == 0:
                metric_vals[i // 10] = V[-1]
                result = [metric_vals]
            print(f"iteration {str(i + 1)}, {metric} = {V[-1]}")
            if metric == 'rmse':
                if i > 1 and best_iter == 0:
                    difference.append(abs((V[-1] - V[-2]) / V[-2]))
                    if (difference[-1] < 0.001) and (difference[-1] == min(difference)):  # to verify
                        best_iter = i
            else:
                if V[-1] == max(V):
                    best_iter = i

        X = np.arange(1, max_iter, 10)
        df = pd.DataFrame([metric_vals], columns=X).melt()
        sns.lineplot(x="variable", y="value", data=df, ci='sd')
        plt.xlabel('Iteration')

        if metric == 'aps':
            plt.ylabel('Average Precision Score (APS)')
            if j == 4:
                plt.savefig('results/aps_' + network.init_strategy + '_' + stop_criterion + '.png')
        elif metric == 'auroc':
            plt.ylabel('Area Under ROC Curve')
            if j == 4:
                plt.savefig('results/auroc_' + network.init_strategy + '_' + stop_criterion + '.png')
        elif metric == 'pearson':
            plt.ylabel('Pearson Correlation')
            if j == 4:
                plt.savefig('results/pearson_' + network.init_strategy + '_' + stop_criterion + '.png')
        elif metric == 'r2':
            plt.ylabel('Coefficient of Determination (R^2)')
            if j == 4:
                plt.savefig('results/r2_' + network.init_strategy + '_' + stop_criterion + '.png')
        elif metric == 'rmse':
            plt.ylabel('Root Mean Square Error (RMSE)')
            if j == 4:
                plt.savefig('results/rmse_' + network.init_strategy + '_' + stop_criterion + '.png')
        best_iter_arr.append(best_iter)
        best_iter = 0
        time.sleep(2)  # used since otherwise random initialization gives the same result multiple times

    res_best_iter = statistics.median(best_iter_arr)
    plt.axvline(x=res_best_iter, color='k', label='selected stop iteration', linestyle='dashed')
    plt.legend(loc=4)
    # plt.ylim(0, 1)
    plt.savefig('results/' + metric + '_' + network.init_strategy + '_' + stop_criterion + '.png')
    plt.close("all")

    network = Network(conf_file, input_dir, mask=0, verbose=False)
    initial_error = network.get_error()
    print(f"Final run without masking, stop at iteration: {str(res_best_iter)}")
    # cycle to find new predictions on unmasked matrix
    error_f = []
    epsilon = 0
    for i in range(res_best_iter):
        network.update()
        error_f.append(network.get_error())
        if i > 1:
            epsilon = abs((error_f[-1] - error_f[-2]) / error_f[-2])
        print(f"iteration {i + 1}, relative error = {epsilon}")


if stop_criterion == 'relative_error':
    best_epsilon_arr = []
    # cycle to find the stop criterion value
    for j in range(5):
        epsilon = 0
        error = []
        V = []
        if j > 0:
            verbose = False
        elif j == 0:
            verbose = True
        network = Network(conf_file, input_dir, verbose)
        initial_error = network.get_error()
        print(bold(f"Run number {str(j + 1)} of the algorithm"))
        print(f"initial error: {initial_error}")
        eps_iter = []
        for i in range(max_iter):
            network.update()
            error.append(network.get_error())
            V.append(network.validate(metric))
            if i % 10 == 0:
                metric_vals[i // 10] = network.validate(metric)
                result = [metric_vals]
            if i > 1:
                epsilon = abs((error[-1] - error[-2]) / error[-2])
                if epsilon < 0.001:
                    eps_iter.append(i)
                    print(eps_iter)
            print(f"iteration {i + 1}, relative error = {epsilon}")

        X = np.arange(1, max_iter, 10)
        df = pd.DataFrame([metric_vals], columns=X).melt()
        sns.lineplot(x="variable", y="value", data=df, ci='sd')
        plt.xlabel('Iteration')
        if metric == 'aps':
            plt.ylabel('Average Precision Score (APS)')
            if j == 4:
                plt.savefig('results/aps_' + network.init_strategy + '_' + stop_criterion + '.png')
        elif metric == 'auroc':
            plt.ylabel('Area Under ROC Curve')
            if j == 4:
                plt.savefig('results/auroc_' + network.init_strategy + '_' + stop_criterion + '.png')
        elif metric == 'pearson':
            plt.ylabel('Pearson Correlation')
            if j == 4:
                plt.savefig('results/pearson_' + network.init_strategy + '_' + stop_criterion + '.png')
        elif metric == 'r2':
            plt.ylabel('Coefficient of Determination (R^2)')
            if j == 4:
                plt.savefig('results/r2_' + network.init_strategy + '_' + stop_criterion + '.png')
        elif metric == 'rmse':
            plt.ylabel('Root Mean Square Error (RMSE)')
            if j == 4:
                plt.savefig('results/rmse_' + network.init_strategy + '_' + stop_criterion + '.png')
        best_epsilon_arr.append(eps_iter[0])
        time.sleep(2)  # used since otherwise random initialization gives the same result multiple times

    res_best_epsilon = statistics.median(best_epsilon_arr)
    plt.axvline(x=res_best_epsilon, color='k', label='selected stop iteration', linestyle='dashed')
    #plt.ylim(0, 1)
    plt.legend(loc=4)
    plt.savefig('results/' + metric + '_' + network.init_strategy + '_' + stop_criterion + '.png')
    plt.close("all")

    network = Network(conf_file, input_dir, mask=0, verbose=False)
    initial_error = network.get_error()
    prev_error = initial_error
    print(f"Final run without masking, stop at iteration: {str(res_best_epsilon)}")
    # cycle to find new predictions on unmasked matrix
    error_f = []
    epsilon = 0
    for i in range(res_best_epsilon):
        network.update()
        error_f.append(network.get_error())
        if i > 1:
            epsilon = abs((error_f[-1] - error_f[-2]) / error_f[-2])
        print(f"iteration {i + 1}, relative error = {epsilon}")



elif stop_criterion == 'maximum_iterations':
    network = Network(conf_file, input_dir, mask=0, verbose=True)
    initial_error = network.get_error()
    print(bold("Unique run of the algorithm without masking"))
    print(f"initial error: {initial_error}")
    error = []
    for i in range(max_iter):
        network.update()
        error.append(network.get_error())
        if i % 10 == 0:
            metric_vals[i // 10] = network.validate(metric)
            result = [metric_vals]
        if i > 1:
            epsilon = abs((error[-1] - error[-2]) / error[-2])
            print(f"iteration {i + 1}, relative error = {epsilon}")



# reconstruction of the matrix from factor matrices at the final point and output of result
rebuilt_association_matrix = np.linalg.multi_dot(
    [network.get_main().G_left, network.get_main().S, network.get_main().G_right.transpose()])
new_relations_matrix = rebuilt_association_matrix - network.get_main().original_matrix
n, m = new_relations_matrix.shape
outF = open("results/myOutFile.txt", "w")
for i in range(n):
    for j in range(m):
        if new_relations_matrix[i, j] > threshold:
            line = network.get_main().left_sorted_term_list[i] + "  " + network.get_main().right_sorted_term_list[
                j] + "  " + str(new_relations_matrix[i, j])
            outF.write(line)
            outF.write("\n")
outF.close()