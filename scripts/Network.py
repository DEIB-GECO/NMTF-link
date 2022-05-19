import warnings

warnings.filterwarnings('ignore')
import sys

sys.path.insert(0, 'scripts/')
from AssociationMatrix import AssociationMatrix, EvaluationMetric
import numpy as np
import os
import yaml


class Network():

    def __init__(self, graph_topology_file, dirfilename, verbose, mask=1):
        self.graph_topology_file = graph_topology_file
        self.init_strategy = "random"
        self.integration_strategy = lambda x, y: x.intersection(y)
        self.type_of_masking = 1

        self.association_matrices = []
        self.datasets = {}
        self.dataset_ks = {}
        self.files = {}

        with open(self.graph_topology_file) as f:
            graph_topology = yaml.load(f, Loader=yaml.FullLoader)

            integration_strategy = graph_topology["integration.strategy"]
            if integration_strategy == "union":
                self.integration_strategy = lambda x, y: x.union(y)
            elif integration_strategy == "intersection":
                self.integration_strategy = lambda x, y: x.intersection(y)
            else:
                print("Option '{}' not supported".format(integration_strategy))
                exit(-1)

            initialization = graph_topology["initialization"]
            if (initialization == "random") or (initialization == "kmeans") or (initialization == "skmeans"):
                self.init_strategy = initialization
            else:
                print("Option '{}' not supported".format(initialization))
                exit(-1)
            if verbose == True:
                print("Initialization strategy is " + self.init_strategy + "\n")

            type_of_masking = graph_topology["type.of.masking"]
            if type_of_masking == "fully_random":
                self.type_of_masking = 0
            elif type_of_masking == "per_row_random":
                self.type_of_masking = 1
            else:
                print("Option '{}' not supported".format(type_of_masking))
                exit(-1)

            if "k" in graph_topology:
                for data in graph_topology["k"]:
                    dsname = data["dsname"]
                    k = data["k"]
                    self.dataset_ks[dsname.upper()] = int(k)

            # For each category of nodes, compute the intersection or union between the different matrices
            for element in graph_topology["graph.datasets"]:
                filename = element["filename"]
                filename = os.path.join(dirfilename, filename)
                ds1_name = element["nodes.left"].upper()
                ds2_name = element["nodes.right"].upper()
                main = int(element["main"])
                ds1_entities = set()
                ds2_entities = set()

                with open(filename) as af:
                    for edge in af:
                        s_edge = edge.strip().split("\t")
                        ds1_entities.add(s_edge[0])
                        ds2_entities.add(s_edge[1])

                self.files[filename] = [main, ds1_name, ds2_name]

                if ds1_name in self.datasets:
                    self.datasets[ds1_name] = self.integration_strategy(self.datasets[ds1_name], ds1_entities)
                else:
                    self.datasets[ds1_name] = ds1_entities
                if ds2_name in self.datasets:
                    self.datasets[ds2_name] = self.integration_strategy(self.datasets[ds2_name], ds2_entities)
                else:
                    self.datasets[ds2_name] = ds2_entities

        # sort the nodes, such that each matrix receives the same ordered list of nodes
        for k in self.datasets.keys():
            self.datasets[k] = list(sorted(list(self.datasets[k])))

        if verbose == True:
            print('All specified nodes\' categories: ' + "{}".format(
                str(list(self.datasets.keys()))) + "\n")

        for file in self.files.keys():
            self.association_matrices.append(
                AssociationMatrix(file,                                 #file = files.key = filename
                                  self.files[file][1],                  #files[file][1] = ds1_name
                                  self.files[file][2],                  #file[file][2] = ds2_name
                                  self.datasets[self.files[file][1]],
                                  self.datasets[self.files[file][2]],
                                  self.files[file][0],                  #files[file][0] = main
                                  mask,
                                  self.type_of_masking,
                                  verbose))

        for m1 in self.association_matrices:
            for m2 in self.association_matrices:
                if m1 != m2:
                    if m1.leftds == m2.leftds:
                        m1.dep_own_left_other_left.append(m2)
                    elif m1.rightds == m2.rightds:
                        m1.dep_own_right_other_right.append(m2)
                    elif m1.rightds == m2.leftds:
                        m1.dep_own_right_other_left.append(m2)
                    elif m1.leftds == m2.rightds:
                        m1.dep_own_left_other_right.append(m2)

# NOT CLEAR how k is used
        for k in self.datasets.keys():
            if not k.strip().startswith("+"):
                rank = self.select_rank(k)
            for am in self.association_matrices:
                if am.leftds == k:
                    am.k1 = int(rank)
                elif am.rightds == k:
                    am.k2 = int(rank)

        for am in self.association_matrices:
            am.initialize(self.init_strategy, verbose)
            # am.compute_tf_idf()

        for am in self.association_matrices:
            am.create_update_rules()

    # method to calculate rank for each datatype. In case of k-means and shrerical k-means initialization represents number of clusters.
    def select_rank(self, ds_name):
        if ds_name in self.dataset_ks:
            rank = self.dataset_ks[ds_name]

        else:
            if self.init_strategy == "kmeans" or self.init_strategy == "skmeans":
                el_num = len(self.datasets[ds_name])
                if el_num > 200:
                    el_num = int(el_num / 5)
                for am in self.association_matrices:  # rank should be less then the number of unique elements in rows and in columns of any matrix where datatype is present
                    if am.leftds == ds_name:
                        el_num = min([el_num, np.count_nonzero(np.sum(np.unique(am.association_matrix, axis=1), 0)),
                                      np.count_nonzero(np.sum(np.unique(am.association_matrix, axis=0), 1)),
                                      len(self.datasets[am.rightds])])
                    elif am.rightds == ds_name:
                        el_num = min([el_num, np.count_nonzero(np.sum(np.unique(am.association_matrix, axis=1), 0)),
                                      np.count_nonzero(np.sum(np.unique(am.association_matrix, axis=0), 1)),
                                      len(self.datasets[am.leftds])])
                rank = el_num
            elif self.init_strategy == "random":
                rank = 100
        return rank

    def get_error(self):
        return sum([am.get_error() for am in self.association_matrices])

    def update(self):
        for am in self.association_matrices:
            am.update()

    def validate(self, metric=EvaluationMetric.APS):
        for am in self.association_matrices:
            if am.main == 1 and am.validation == 1:
                return am.validate(metric)

    def get_main(self):
        for am in self.association_matrices:
            if am.main == 1:
                return am