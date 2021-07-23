import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

stop_criterion = 'maximum_metric'
init_strategies = ['random', 'skmeans']  # , 'kmeans']
metrics = ['RMSE', 'R2']  # , 'PEARSON']
n_iter = range(100, 300, 100)
masks = ['fully_random']  # , 'per_row_random']
normalizations = ['standard_scaler']#, 'min_max', 'max', 'unit_form', 'none']
# thresholds = np.arange(0, 0.1, 0.001)

# for strategy in init_strategies:
#    for metric in metrics:
#        for mask in masks:
threshold = 0.1
# strategy = 'skmeans'
# metric = 'RMSE'
# mask = 'fully_random'
for normalization in normalizations:
    for mask in masks:
        for strategy in init_strategies:
            for metric in metrics:
                rel_error = []
                iterations = []
                for n in tqdm(n_iter):
                    # for threshold in thresholds:
                    with open('case_study_4/graph_topology_config.tsv', 'w') as f:
                        f.write("#parameters\n"
                                "#integration.strategy	intersection\n"
                                f"#initialization	{strategy}\n"
                                f"#metric	{metric}\n"
                                f"#number.of.iterations	{n}\n"
                                f"#type.of.masking	{mask}\n"
                                f"#normalization	{normalization}\n"
                                "#stop.criterion	maximum_metric\n"
                                f"#score.threshold	{threshold}\n"
                                "#nodes_left	nodes_right	filename	main\n"
                                "cell_line	drug	drug_cell.txt   1\n")

                    os.system("python NMTF-link.py  case_study_4  graph_topology_config.tsv > results/output.txt")

                    with open("results/output.txt", "r") as f:
                        x = f.readlines()
                    l = x[-1].strip().split(',')
                    l[0] = int(l[0].replace('iteration ', ''))
                    iterations.append(l[0])
                    l[1] = float(l[1].replace(' relative error = ', ''))
                    rel_error.append(l[1])

                df = pd.DataFrame([rel_error], columns=n_iter).melt()
                df1 = pd.DataFrame([rel_error], columns=iterations).melt()
                df1 = df1.sort_values('variable')
                #print(df1)
                #sns.lineplot(x='variable', y='value', data=df, ci='sd', label=mask + '_' + strategy + '_' + metric)
                sns.lineplot(x='variable', y='value', data=df1, ci='sd', label=mask + '_' + strategy + '_' + metric)

plt.xlabel('N_Iterations')
plt.ylabel('Relative Error')

plt.legend(loc=4)
plt.savefig('results/configurations/try.png')  # + metric + '_' + strategy + '_' + stop_criterion + '_' + mask +'.png')
# plt.close("all")
