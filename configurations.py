import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

stop_criterion = 'maximum_metric'
init_strategies = ['random', 'skmeans', 'kmeans']
metrics = ['RMSE', 'R2', 'PEARSON']
n_iter = range(100,500,100)
masks = ['fully_random', 'per_row_random']
#thresholds = np.arange(0, 0.1, 0.001)

#for strategy in init_strategies:
#    for metric in metrics:
#        for mask in masks:
threshold = 0.1
#strategy = 'skmeans'
#metric = 'RMSE'
#mask = 'fully_random'

for strategy in init_strategies:
    for metric in metrics:
        for mask in masks:
            rel_error = []
            for n in tqdm(n_iter):
                #for threshold in thresholds:
                with open('case_study_4/graph_topology_config.tsv','w') as f:
                    f.write("#parameters\n"
                            "#integration.strategy	intersection\n"
                            f"#initialization	{strategy}\n"
                            f"#metric	{metric}\n"
                            f"#number.of.iterations	{n}\n"
                            f"#type.of.masking	{mask}\n"
                            "#stop.criterion	maximum_metric\n"
                            f"#score.threshold	{threshold}\n"
                            "#nodes_left	nodes_right	filename	main\n"
                            "cell_line	drug	drug_cell.txt   1\n")


                os.system("python NMTF-link.py  case_study_4  graph_topology_config.tsv > results/output.txt")

                with open("results/output.txt", "r") as f:
                    x = f.readlines()
                l = x[-1].strip().split(',')
                l[0] = int(l[0].replace('iteration ', ''))
                l[1] = float(l[1].replace(' relative error = ', ''))
                rel_error.append(l[1])

            df = pd.DataFrame([rel_error], columns=n_iter).melt()
            print(df)
            sns.lineplot(x = 'variable', y='value', data=df, ci='sd', label=n)

        plt.xlabel('Threshold')
        plt.ylabel('Relative Error')

        plt.legend(loc=4)
        plt.savefig('results/configurations/' + metric + '_' + strategy + '_' + stop_criterion + '_' + mask +'.png')
#plt.close("all")
