import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

stop_criterion = 'maximum_metric'
init_strategies = 'skmeans'
integ_strategies = 'intersection'
metrics = 'LOG_RMSE'
mask = 'fully_random'
n_iter = 25
ks = range(50, 100, 10)

errors = {}
for k1 in ks:
    for k2 in ks:
        rel_error = []
        iterations = []
        with open('case_study_4/graph_topology_config_k.tsv', 'w') as f:
            f.write("#parameters\n"
                    f"#integration.strategy	{integ_strategies}\n"
                    f"#initialization	{init_strategies}\n"
                    f"#metric	{metrics}\n"
                    f"#number.of.iterations	{n_iter}\n"
                    f"#type.of.masking	{mask}\n"
                    "#stop.criterion	maximum_metric\n"
                    f"#score.threshold	0.001\n"
                    f"#k	cell_line	{k1}\n"
                    f"#k	drug	{k2}\n"
                    "#nodes_left	nodes_right	filename	main\n"
                    "cell_line	drug	drug_cell_log10.txt   1\n")

        os.system("python NMTF-link.py  case_study_4  graph_topology_config_k.tsv > results/output.txt")

        with open("results/output.txt", "r") as f:
            lines = f.readlines()
            last_line = lines#[-1]#[-2:-1]
            with open("results/all_output.txt", "a") as f2:
                f2.write(f'###### K1:{k1}, K2: {k2}')
                f2.writelines(lines)
        l = last_line
        l = l[-1].strip().split(',')
        print(f'k1:{k1}, k2: {k2}, {l}')
        l[0] = int(l[0].replace('iteration ', ''))
        error = float(l[1].split(' ')[-1])
        errors[(k1,k2)] = error#l[1]


print('############ MIN:   ' , min(errors, key=errors.get), errors[min(errors, key=errors.get)])