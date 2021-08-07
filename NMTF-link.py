import warnings

warnings.filterwarnings('ignore')
import sys
from scripts import Network
import numpy as np
import matplotlib
from utils import EvaluationMetric, StopCriterion

matplotlib.use('agg')
import pylab as plt
import time
import statistics
import os

current = os.getcwd()

_, filename_1, filename_2 = sys.argv
dirname_1 = os.path.join(current, filename_1, filename_2)
dirname_2 = os.path.join(current, filename_1)

# Baseline parameters
default_threshold = 0.1
threshold=default_threshold
metric = 'aps'
max_iter =200
stop_criterion= 'calculate'

def plot_iteration(max_it, met_val):
    X = np.arange(1, max_it, 10)
    plt.plot(X, met_val)

def complete_plot(m):
    plt.xlabel('Iteration')
    if m == EvaluationMetric.APS:
        plt.ylabel('Average Precision Score (APS)')
        plt.ylim(0,1)
    elif m == EvaluationMetric.AUROC:
        plt.ylabel('Area Under ROC Curve')
        plt.ylim(0, 1)
    elif m == EvaluationMetric.RMSE:
        plt.ylabel('RMSE')

def predict(num_iterations, th):
    network = Network(dirname_1, dirname_2, mask=0, verbose=False)

    for i in range(int(num_iterations)):
        network.update()
        print(f"iteration {i+1}, error = {network.get_error()}")

    rebuilt_association_matrix = np.linalg.multi_dot(
        [network.get_main().G_left, network.get_main().S, network.get_main().G_right.transpose()])
    new_relations_matrix = rebuilt_association_matrix - network.get_main().original_matrix
    n, m = new_relations_matrix.shape
    with open("results/myOutFile.txt", "w") as outF:
        for i in range(n):
            for j in range(m):
                if new_relations_matrix[i, j] > th:
                    line = network.get_main().left_sorted_term_list[i] + "  " + network.get_main().right_sorted_term_list[
                        j] + "  " + str(new_relations_matrix[i, j])
                    outF.write(line)
                    outF.write("\n")


with open(dirname_1) as f:
    for line in f:
        if line.strip().startswith("#metric"):
            _,metric_name = line.strip().split("\t")
            metric = EvaluationMetric(metric_name.upper())

        if line.strip().startswith("#number.of.iterations"):
            try:
                s, max_iter_value = line.strip().split("\t")
                max_iter = int(max_iter_value)
            except ValueError:
                print(f"Invalid number of iteration {max_iter_value}, set default value {max_iter}", file=sys.stderr)

        if line.strip().startswith("#stop.criterion"):
            _,criterion_name = line.strip().split("\t")
            stop_criterion = StopCriterion(criterion_name.upper())

        if line.strip().startswith("#score.threshold"):
            _, th_value = line.strip().split("\t")
            try:
                threshold = float(th_value)
                if not (0 <= threshold <= 1):
                    raise ValueError()
            except ValueError:
                threshold = default_threshold

print(f"metric : {metric.value}")
print(f"number of iterations : {max_iter}")
print(f"stop criterion : {stop_criterion.value}")
print(f"threshold : {threshold}")

metric_vals = np.zeros(max_iter // 10)

if stop_criterion == StopCriterion.MAXIMUM_METRIC:
    best_iter = 0
    best_iter_arr = []  # contains the iterations with best performance from each of 5 validation runs (j cycle)
    # cycle to find the stop criterion value
    for j in range(20):
        V = []
        if j > 0:
            verbose = False
        elif j == 0:
            verbose = True
        network = Network(dirname_1, dirname_2, verbose)
        initial_error = network.get_error()
        print('\033[1m' + "Run number " + str(j + 1) + " of the algorithm" + '\033[0m')
        #print("initial error: {}".format(initial_error))
        for i in range(max_iter):
            network.update()
            V.append(network.validate(metric))
        print(f"iteration {i + 1}, {metric.value} = {V[-1]}")
        best_iter_arr.append(V.index(min(V)) if (metric==EvaluationMetric.RMSE or metric==EvaluationMetric.LOG_RMSE) else V.index(max(V)))
        best_iter = 0
        time.sleep(2)  # used since otherwise random initialization gives the same result multiple times

    #res_best_iter = statistics.median(best_iter_arr)
    #print(res_best_iter)
    #predict(res_best_iter, threshold)
    res_best_iter = 24
    predict(res_best_iter, threshold)
    print(f"iteration {res_best_iter}, {metric.value} = {V[int(res_best_iter)]}")

elif stop_criterion == StopCriterion.RELATIVE_ERROR:
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
        network = Network(dirname_1, dirname_2, verbose)
        initial_error = network.get_error()
        print('\033[1m' + "Run number " + str(j + 1) + " of the algorithm" + '\033[0m')
        print("initial error: {}".format(initial_error))
        eps_iter = []
        for i in range(max_iter):
            network.update()
            error.append(network.get_error())
            V.append(network.validate(metric))
            if i % 10 == 0:
                metric_vals[i // 10] = network.validate(metric)
            if i > 1:
                epsilon = abs((error[-1] - error[-2]) / error[-2])
                if epsilon < 0.001:
                    eps_iter.append(i)
            print(f"iteration {i + 1}, relative error = {epsilon}")

        plot_iteration(max_iter, metric_vals)
        time.sleep(2)  # used since otherwise random initialization gives the same result multiple times
        best_epsilon_arr.append(eps_iter[0])

    complete_plot(metric)

    res_best_epsilon = statistics.median(best_epsilon_arr)
    plt.axvline(x=res_best_epsilon, color='k', label='selected stop iteration', linestyle='dashed')
    plt.legend(loc=4)
    plt.savefig('results/' + metric.value + '_' + network.init_strategy + '_' + stop_criterion.value + '.png')
    plt.close("all")

    print('\033[1m' + "Final run without masking, stop at iteration: " + str(res_best_epsilon) + '\033[0m')
    predict(res_best_epsilon, threshold)

elif stop_criterion == StopCriterion.MAXIMUM_ITERATIONS:
    network = Network(dirname_1, dirname_2, mask=0, verbose=True)
    initial_error = network.get_error()
    print('\033[1m' + "Unique run of the algorithm without masking" + '\033[0m')
    predict(max_iter, threshold)