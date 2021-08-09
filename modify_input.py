import numpy as np

with open("carolina/input.graph.tsv") as f:
    with open("carolina/input.graph.norm.tsv", "w") as of:
        for line in f:
            a,b,c=line.strip().split("\t")
            new_c = np.sqrt(float(c))
            of.write(f'{a}\t{b}\t{new_c}\n')