import warnings

warnings.filterwarnings('ignore')
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
import sklearn.metrics as metrics
import copy

from contextlib import contextmanager
import os
import sys
sys.path.insert(0, 'scripts/')
from utils import EvaluationMetric


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def parse_line(line):
    s = line.strip().split("\t")
    if len(s)==2:
        return tuple(s+[1])
    elif len(s)==3:
        return tuple([s[0],s[1],float(s[2])])
    else:
        raise Exception()


class AssociationMatrix():
    def __init__(self, filename, leftds, rightds, left_sorted_terms, right_sorted_terms, main, mask, type_of_masking,
                 verbose):
        self.filename = filename
        self.leftds = leftds
        self.rightds = rightds
        self.intra_data_matrices = []
        self.dep_own_right_other_right = []
        self.dep_own_right_other_left = []
        self.dep_own_left_other_right = []
        self.dep_own_left_other_left = []
        self.left_sorted_term_list = left_sorted_terms
        self.right_sorted_term_list = right_sorted_terms
        self.k1 = 0
        self.k2 = 0
        self.main = main
        self.rightds_intra = None
        self.leftds_intra = None
        self.validation = mask
        if self.main == 1:
            self.type_of_masking = type_of_masking

        with open(self.filename, "r") as f:
            data_graph = [parse_line(element) for element in f.readlines()]
        self.edges = [el
                      for el in data_graph
                      if el[0] in set(self.left_sorted_term_list) and el[1] in set(self.right_sorted_term_list)]

        graph = nx.Graph()
        graph.add_nodes_from(list(self.left_sorted_term_list), bipartite=0)
        graph.add_nodes_from(list(self.right_sorted_term_list), bipartite=1)
        graph.add_weighted_edges_from(self.edges)

        self.association_matrix = nx.algorithms.bipartite.matrix.biadjacency_matrix(graph,
                                                                                    self.left_sorted_term_list,
                                                                                    self.right_sorted_term_list)
        self.association_matrix = self.association_matrix.toarray()
        self.original_matrix = copy.deepcopy(self.association_matrix)  # for all to use in select_rank
        if self.main == 1 and self.validation == 1:  # so this is the matrix which we try to investigate
            self.mask_matrix()
            # self.compute_tf_idf() #not affecting original matrix
        self.G_left = None
        self.G_left_primary = False
        self.G_right = None
        self.G_right_primary = False
        self.S = None

    def initialize(self, initialize_strategy, verbose):
        if initialize_strategy == "random":
            if verbose == True:
                print("Association matrix filename: " + self.filename)
                print("Used parameters: " + '\033[1m' + " k\u2081 = " + str(self.k1) + " and" + " k\u2082 = " + str(
                    self.k2) + '\033[0m')
                print("Non-zero elements of the association matrix = " + '\033[1m' + "{}".format(
                    np.count_nonzero(self.association_matrix)) + '\033[0m')
            if self.G_left is None:
                self.G_left = np.random.rand(self.association_matrix.shape[0], self.k1)
                self.G_left_primary = True
            if self.G_right is None:
                self.G_right = np.random.rand(self.association_matrix.shape[1], self.k2)
                self.G_right_primary = True
        elif initialize_strategy == "oldkmeans":
            if verbose == True:
                print("Association matrix filename: " + self.filename)
                print("Used parameters: " + '\033[1m' + " k\u2081 = " + str(self.k1) + " and" + " k\u2082 = " + str(
                    self.k2) + '\033[0m')
                print("Non-zero elements of the association matrix = " + '\033[1m' + "{}".format(
                    np.count_nonzero(self.association_matrix)) + '\033[0m')
            if self.G_left is None:
                with suppress_stdout():
                    km = KMeans(n_clusters=self.k1).fit(self.association_matrix)
                    self.G_left = np.zeros((self.association_matrix.shape[0], self.k1))
                    for row in range(self.association_matrix.shape[0]):
                        for col in range(self.k1):
                            self.G_left[row, col] = np.linalg.norm(
                                self.association_matrix[row] - km.cluster_centers_[col])
                    self.G_left_primary = True
            if self.G_right is None:
                with suppress_stdout():
                    km = KMeans(n_clusters=self.k2).fit(self.association_matrix.transpose())
                    self.G_right = np.zeros((self.association_matrix.shape[1], self.k2))
                    for row in range(self.association_matrix.shape[1]):
                        for col in range(self.k2):
                            self.G_right[row, col] = np.linalg.norm(
                                self.association_matrix.transpose()[row] - km.cluster_centers_[col])
                    self.G_right_primary = True
        elif initialize_strategy == "kmeans":
            if verbose == True:
                print("Association matrix filename: " + self.filename)
                print("Used parameters: " + '\033[1m' + " k\u2081 = " + str(self.k1) + " and" + " k\u2082 = " + str(
                    self.k2) + '\033[0m')
                print("Non-zero elements of the association matrix = " + '\033[1m' + "{}".format(
                    np.count_nonzero(self.association_matrix)) + '\033[0m')
            if self.G_left is None:
                with suppress_stdout():
                    km = KMeans(n_clusters=self.k1, n_init=10).fit_predict(self.association_matrix.transpose())
                    self.G_left = np.array(
                        [np.mean([self.association_matrix[:, i] for i in range(len(km)) if km[i] == p], axis=0) for p in
                         range(self.k1)]).transpose()
                    self.G_left_primary = True
            if self.G_right is None:
                with suppress_stdout():
                    km = KMeans(n_clusters=self.k2, n_init=10).fit_predict(self.association_matrix)
                    self.G_right = np.array(
                        [np.mean([self.association_matrix[i] for i in range(len(km)) if km[i] == p], axis=0) for p in
                         range(self.k2)]).transpose()
                    self.G_right_primary = True
        elif initialize_strategy == "skmeans":
            if verbose == True:
                print("Association matrix filename: " + self.filename)
                print("Used parameters: " + '\033[1m' + " k\u2081 = " + str(self.k1) + " and" + " k\u2082 = " + str(
                    self.k2) + '\033[0m')
                print("Non-zero elements of the association matrix = " + '\033[1m' + "{}".format(
                    np.count_nonzero(self.association_matrix)) + '\033[0m')
            # with suppress_stdout():
            if self.G_left is None:
                with suppress_stdout():
                    #skm = SphericalKMeans(n_clusters=self.k1)
                    skm = SphericalKMeans(n_clusters=5)
                    skm = skm.fit(self.association_matrix.transpose())
                    # Factor matrices are initialized with the center coordinates
                    self.G_left = skm.cluster_centers_.transpose()
                    self.G_left_primary = True
            if self.G_right is None:
                with suppress_stdout():
                    skm = SphericalKMeans(n_clusters=self.k2).fit(self.association_matrix)
                    # Factor matrices are initialized with the center coordinates
                    self.G_right = skm.cluster_centers_.transpose()
                    self.G_right_primary = True

        for am in self.dep_own_left_other_left:
            if am.G_left is None:
                am.G_left = self.G_left
        for am in self.dep_own_left_other_right:
            if am.G_right is None:
                am.G_right = self.G_left
        for am in self.dep_own_right_other_left:
            if am.G_left is None:
                am.G_left = self.G_right
        for am in self.dep_own_right_other_right:
            if am.G_right is None:
                am.G_right = self.G_right
        if verbose == True:
            print(self.leftds, self.rightds, self.association_matrix.shape)
            print("Shape Factor Matrix left " + str(self.G_left.shape))
            print("Shape Factor Matrix right " + str(self.G_right.shape) + "\n")
        self.S = np.linalg.multi_dot([self.G_left.transpose(), self.association_matrix, self.G_right])

    def compute_tf_idf2(self):
        self.tf_rows = np.true_divide(np.count_nonzero(self.original_matrix, axis=1), np.count_nonzero(
            self.original_matrix))  # axis - Axis or tuple of axes along which to count non-zeros.
        self.tf_columns = np.true_divide(np.count_nonzero(self.original_matrix, axis=0),
                                         np.count_nonzero(self.original_matrix))

    def compute_tf_idf(self):
        self.tf = np.true_divide(self.association_matrix,
                                 np.repeat(np.count_nonzero(self.association_matrix, axis=1)[:, np.newaxis],
                                           self.association_matrix.shape[1], axis=1) + 0.00000001)
        # axis - Axis or tuple of xes along which to count non-zeros.
        self.idf = np.repeat(np.log(np.true_divide(np.count_nonzero(self.association_matrix),
                                                   np.count_nonzero(self.association_matrix, axis=0) + 0.00000001))[
                             np.newaxis, :], self.association_matrix.shape[0], axis=0)
        self.association_matrix = np.multiply(self.association_matrix, self.tf, self.idf)

    # method to mask the matrix. Used in validation phase to be able later in validate method to assess performance of the algorithm. self.type_of_masking is set in the settings file and read in the method open of the class network. This parameter can be "fully_random"(0) or "per_row_random"(1). In the first case masking elements in the created mask are distrubited uniformly randomly and in the second case mask has same number of masking elements per row, distributed randomly within the row.
    def mask_matrix(self):
        self.M = np.zeros_like(self.association_matrix)
        if self.type_of_masking == 0:
            a = np.ones(self.association_matrix.shape, dtype=self.association_matrix.dtype)
            n = self.association_matrix.size * 0.05
            a = a.reshape(a.size)
            a[:int(n)] = 0
            np.random.shuffle(a)
            a = a.reshape(self.association_matrix.shape)
            self.association_matrix = np.multiply(self.association_matrix, a)
            self.M = a
        else:
            for i in range(0, self.association_matrix.shape[0] - 1):
                nc = self.association_matrix.shape[1]  # nc is row size ( number of columns)
                a = np.ones(nc, dtype=int)  # get array of dimension of 1 row
                n = self.association_matrix.shape[1] * 0.05
                a[:int(n)] = 0
                np.random.shuffle(a)
                self.association_matrix[i, :] = np.multiply(self.association_matrix[i, :], a)
                self.M[i, :] = a

    # method to produce performance metrics (APS, AUROC). Produces output only if the matrix is the matrix for which predictions are searched and the network is in validation mode.
    def validate(self, metric=EvaluationMetric.APS):
        with suppress_stdout():
            if self.main == 1 and self.validation == 1:
                self.rebuilt_association_matrix = np.linalg.multi_dot([self.G_left, self.S, self.G_right.transpose()])
                n, m = self.rebuilt_association_matrix.shape
                # R12_found = np.linalg.multi_dot([self.G1, self.S12, self.G2.transpose()])#equals to rebuilt_association_matrix
                R12_2 = []
                R12_found_2 = []

                # We first isolate the validation set and the corresponding result
                for i in range(n):
                    for j in range(m):
                        if self.M[i, j] == 0:
                            R12_2.append(self.original_matrix[i, j])
                            R12_found_2.append(self.rebuilt_association_matrix[i, j])
                # We can asses the quality of our output with APS or AUROC score
                if metric == EvaluationMetric.AUROC:
                    fpr, tpr, threshold = metrics.roc_curve(R12_2, R12_found_2)
                    return metrics.auc(fpr, tpr)
                elif metric == EvaluationMetric.APS:
                    return metrics.average_precision_score(R12_2, R12_found_2)
                else:
                    print("NOT GUD", file=sys.stderr)
                    print(f"metric = {type(metric)}", file=sys.stderr)
                    print(f"metric = {type(EvaluationMetric.APS)}", file=sys.stderr)
                    print(f"metric = {metric}", file=sys.stderr)
                    AAAA

    def get_error(self):
        self.rebuilt_association_matrix = np.linalg.multi_dot([self.G_left, self.S, self.G_right.transpose()])
        return np.linalg.norm(self.association_matrix - self.rebuilt_association_matrix, ord='fro') ** 2

    def create_update_rules(self):
        if self.G_right_primary:
            def update_G_r():
                num = np.linalg.multi_dot([self.association_matrix.transpose(), self.G_left, self.S])
                den = np.linalg.multi_dot(
                    [self.G_right, self.G_right.transpose(), self.association_matrix.transpose(), self.G_left, self.S])
                for am in self.dep_own_right_other_right:
                    num += np.linalg.multi_dot([am.association_matrix.transpose(), am.G_left, am.S])
                    den += np.linalg.multi_dot(
                        [am.G_right, am.G_right.transpose(), am.association_matrix.transpose(), am.G_left, am.S])
                for am in self.dep_own_right_other_left:
                    num += np.linalg.multi_dot([am.association_matrix, am.G_right, am.S.transpose()])
                    den += np.linalg.multi_dot(
                        [am.G_left, am.G_left.transpose(), am.association_matrix, am.G_right, am.S.transpose()])
                div = np.sqrt(np.divide(num, den + 0.00000001))
                return np.multiply(self.G_right, div)

            self.update_G_right = update_G_r

        if self.G_left_primary:
            def update_G_l():
                num = np.linalg.multi_dot([self.association_matrix, self.G_right, self.S.transpose()])
                den = np.linalg.multi_dot(
                    [self.G_left, self.G_left.transpose(), self.association_matrix, self.G_right, self.S.transpose()])

                for am in self.dep_own_left_other_left:
                    num += np.linalg.multi_dot([am.association_matrix, am.G_right, am.S.transpose()])
                    den += np.linalg.multi_dot(
                        [am.G_left, am.G_left.transpose(), am.association_matrix, am.G_right, am.S.transpose()])
                for am in self.dep_own_left_other_right:
                    num += np.linalg.multi_dot([am.association_matrix.transpose(), am.G_left, am.S])
                    den += np.linalg.multi_dot(
                        [am.G_right, am.G_right.transpose(), am.association_matrix.transpose(), am.G_left, am.S])
                div = np.sqrt(np.divide(num, den + 0.00000001))
                return np.multiply(self.G_left, div)

            self.update_G_left = update_G_l

        def update_S():
            num = np.linalg.multi_dot([self.G_left.transpose(), self.association_matrix, self.G_right])
            den = np.linalg.multi_dot(
                [self.G_left.transpose(), self.G_left, self.S, self.G_right.transpose(), self.G_right])
            div = np.sqrt(np.divide(num, den + 0.00000001))
            return np.multiply(self.S, div)

        self.update_S = update_S

    def update(self):
        if self.G_right_primary:
            self.G_right = self.update_G_right()
        if self.G_left_primary:
            self.G_left = self.update_G_left()
        self.S = self.update_S()
