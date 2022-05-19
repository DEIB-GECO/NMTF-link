import warnings
import sys
from scripts import Network
import numpy as np
import matplotlib
from utils import EvaluationMetric, StopCriterion
import pylab as plt
import time
import statistics
import os
import yaml

warnings.filterwarnings('ignore')
matplotlib.use('agg')

# current directory
current = os.getcwd()

_, filename_1, filename_2 = sys.argv
dirname_1 = os.path.join(current, filename_1, filename_2)
dirname_2 = os.path.join(current, filename_1)

# Baseline parameters
default_threshold = 0.1
threshold = default_threshold
metric = 'aps'
max_iter = 200
stop_criterion = 'calculate'

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

    for i in range(num_iterations):
        network.update()
        print(f"iteration {i}, error = {network.get_error()}")

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


with open(dirname_1, 'r') as f:
    graph_topology = yaml.load(f, Loader=yaml.FullLoader)

    metric = EvaluationMetric(graph_topology["metric"].upper())
    stop_criterion = StopCriterion(graph_topology["stop.criterion"].upper())

    try:
        vmax_iter = graph_topology["number.of.iterations"]
        vmax_iter = int(vmax_iter)
    except ValueError:
        print(f"Invalid number of iteration {vmax_iter}, set default value {max_iter}", file=sys.stderr)

    try:
        threshold = graph_topology["score.threshold"]
        threshold = int(threshold)
        if not (0 <= threshold <= 1):
            raise ValueError()
    except ValueError:
        print(f"Invalid threshold {threshold}, set default value {default_threshold}")
        threshold = default_threshold

print("\nmetric :", metric.value)
print("number of iterations : ", max_iter)
print("stop criterion : ", stop_criterion.value)
print("threshold : ", threshold)

metric_vals = np.zeros(max_iter // 10)

if stop_criterion == StopCriterion.MAXIMUM_METRIC:
    best_iter = 0
    best_iter_arr = []  # contains the iterations with the best performance from each of 5 validation runs (j cycle)

    # cycle to find the stop criterion value
    for j in range(5):
        V = []
        if j > 0:
            verbose = False
        elif j == 0:
            verbose = True

        network = Network(dirname_1, dirname_2, verbose)

        initial_error = network.get_error()
        print("Run number " + str(j + 1) + " of the algorithm")
        start = time.time()
        print("initial error: {}".format(initial_error))
        for i in range(max_iter):
            startIteration = time.time();
            network.update()
            if i % 10 == 0:
                metric_vals[i // 10] = network.validate(metric)
            V.append(network.validate(metric))
            endIteration = time.time();
            print(f"iteration {i + 1}, {metric.value} = {V[-1]}, time = {endIteration - startIteration}")
        end = time.time()
        plot_iteration(max_iter, metric_vals)
        best_iter_arr.append(V.index(min(V)) if metric == EvaluationMetric.RMSE else V.index(max(V)))
        best_iter = 0
        print("Total time: ", end - start)
        time.sleep(2)  # used since otherwise random initialization gives the same result multiple times

    complete_plot(metric)

    res_best_iter = statistics.median(best_iter_arr)
    plt.axvline(x=res_best_iter, color='k', label='selected stop iteration', linestyle='dashed')
    plt.legend(loc=4)
    plt.savefig(f'results/{metric.value}_{network.init_strategy}_{stop_criterion.value}.png')
    plt.close("all")

    predict(res_best_iter, threshold)

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
        print("\nRun number " + str(j + 1) + " of the algorithm")
        start = time.time()
        print("initial error: {}".format(initial_error))
        eps_iter = []
        for i in range(max_iter):
            startIteration = time.time()
            network.update()
            error.append(network.get_error())
            V.append(network.validate(metric))
            if i % 10 == 0:
                metric_vals[i // 10] = network.validate(metric)
            if i > 1:
                epsilon = abs((error[-1] - error[-2]) / error[-2])
                if epsilon < 0.001:
                    eps_iter.append(i)
            endIteration = time.time();
            print(f"iteration {i + 1}, relative error = {epsilon}, time = {endIteration - startIteration}")
        end = time.time()
        print("Total time: ", end - start)
        plot_iteration(max_iter, metric_vals)

        time.sleep(2)  # used since otherwise random initialization gives the same result multiple times
        best_epsilon_arr.append(eps_iter[0])

    complete_plot(metric)

    res_best_epsilon = statistics.median(best_epsilon_arr)
    plt.axvline(x=res_best_epsilon, color='k', label='selected stop iteration', linestyle='dashed')
    plt.legend(loc=4)
    plt.savefig('results/' + metric.value + '_' + network.init_strategy + '_' + stop_criterion.value + '.png')
    plt.close("all")

    print("\nFinal run without masking, stop at iteration: " + str(res_best_epsilon))
    predict(res_best_epsilon, threshold)

elif stop_criterion == StopCriterion.MAXIMUM_ITERATIONS:
    network = Network(dirname_1, dirname_2, mask=0, verbose=True)
    initial_error = network.get_error()
    print("\nUnique run of the algorithm without masking")
    predict(max_iter, threshold)