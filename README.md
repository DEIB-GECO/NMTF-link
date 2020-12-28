# NMTF-link Tool

This tool is designed for computing **link predictions** using the **Non Negative Matrix Tri-Factorization (NMTF)** method. This is generalization for already pre-existing code in [DEIB-GECO/NMTF-DrugRepositioning](https://github.com/DEIB-GECO/NMTF-DrugRepositioning) allowing the use of networks of any topology. 
**This is a command line tool that uses a setting file**. 

An example setting file called [graph_topology.tsv](https://github.com/DEIB-GECO/NMTF-link/case_study_1/graph_topology.tsv) is located in the subfolders "case_study_1" or "case_study_2" of the main directory. An example of [myOutFile.txt](https://github.com/DEIB-GECO/NMTF-link/results/myOutFile.txt) file containing new possible relations among objects of the datasets is located in the results folder of the main directory.

# Table of contents
- [Parameters to be specified by the user in the setting file](#parameters-to-be-specified-by-the-user-in-the-setting-file)
- [Other entries to be specified by the user in the setting file](#other-entries-to-be-specified-by-the-user-in-the-setting-file)
- [Setting file example](#setting-file-example)
  * [Interpretation of the setting file](#interpretation-of-the-setting-file)
- [Contents](#contents)
- [Usage](#usage)

# Parameters to be specified by the user in the setting file
In the [graph_topology.tsv](https://github.com/DEIB-GECO/NMTF-link/case_study_1/graph_topology.tsv) file the user needs to specify the following parameters:

- **metric**, performance evaluation metric.

  *Options*: "APS" (Average Precision Score) and "AUROC" (Area Under the ROC Curve).

- **initialization**, method to initialize the factor matrices, which are the three matrices that factorize each association matrix. 

  *Options*: "random", "kmeans" and "skmeans".

- **integration.strategy**, mode of integration of datasets.
When one dataset is present in several association matrices, there are two ways to integrate its elements. The options are either to use only the objects, which are shared by all association matrices (intersection) or to use all its objects, which are present in at least one matrix (union).

  *Options*: "intersection" and "union".

- **number.of.iterations**, number of maximum iterations for each run of the algorithm. 

  *Options*: any integer value.

- **stop.criterion**, stop criterion strategies for link predictions using the NMTF method. 

  *Options*: "maximum_metric", "relative_error" and "maximum_iterations".
    
    - "maximum_metric" option runs the algorithm 5 times with masking, chooses the iteration with best average evaluation metric and after runs one more time, without evaluation, untile chosen iteration and outputs results. It also outputs evaluation plots to the main directory.
    - "relative_error" option runs the algorithm 5 times with masking, chooses the iteration with relative error < 0.001 and after runs one more time, without evaluation, until chosen iteration and outputs results. It also outputs evaluation plots to the main directory.
    - "maximum_iterations" option runs the chosen number of iterations without masking and outputs result for the last iteration. 

- **type.of.masking**, to evaluate the NMTF predictions, there is the need of choosing the masking strategy applied on the selected association matrix. 
It can either have a completely randomized distribution of masking elements or have the same number of masking elements per row randomly distributed within each row. 

  *Options*: "fully_random" and "per_row_random".

- **threshold.for.retrieval**, value of threshold for retrieved results. 

  *Options*: any value between 0 and 1.

# Other entries to be specified by the user in the setting file

In the setting file, the user has also to specify the datasets which will be used to create the network. The structure of the datafile line is (entries separated by whitespace):
- name of left category of nodes (e.g., in tenth line of the setting file the left category of nodes is **users**)
- name of right category of nodes (e.g., in tenth line of the setting file the right category of nodes is **genres**)
- filename (e.g., in tenth line of the setting file the filename is **UsersToGenres.txt**, which has to be in the specified data directory)
- 1 or 0 value in the **main** column representing whether or not this datafile represents the matrix which will be searched for new links.

# Setting file example

Example of [graph_topology.tsv](https://github.com/DEIB-GECO/NMTF-link/case_study_1/graph_topology.tsv) file:

| #general parameters | | | |
| ------------- | ------------- | ------------- | ------------- |
| **#metric**  | APS |  |  |
| **#initialization**  | random |  |  |
| **#integration.strategy**  | intersection |  |  |
| **#number.of.iterations** | 200 |  |  |
| **#stop.criterion**  | relative_error |  |  |
| **#type.of.masking** | fully_random |  |  |
| **#threshold.for.retrieval** | 0.5 |  |  |
| **#ds1** |  |  | **main** |
| users |	genres |	UsersToGenres.txt |	0 |
| users |	movies |	UsersToMovies.txt	| 1 |
| movies	| actors	| MoviesToActors.txt |	0 |
| movies	| genres	| MoviesToGenres.txt	| 0 |

## Interpretation of the setting file

We used APS (Average Precision Score) as evaluation metric, a random initialization strategy for the factor matrices, 200 maximum iterations for each run of the algorithm, a stop criterion based on the relative error of the loss function and a random masking strategy for the evaluation process. The mode of integration of shared datasets is by intersection, i.e., only objects shared by all association matrices are considered. The new link predictions have an NMTF score above 0.5 (the threshold for retrieval).

The example contains 4 nodes' categories (**users, genres, actors and movies**). Users to genres links are reported in [UsersToGenres.txt](https://github.com/DEIB-GECO/NMTF-link/case_study_1/UsersToGenres.txt), each link represents the preference of movie genres for a specific user. Users to movies links are the list of watched movies for each user reported in [UsersToMovies.txt](https://github.com/DEIB-GECO/NMTF-link/case_study_1/UsersToMovies.txt). [MoviesToActors.txt](https://github.com/DEIB-GECO/NMTF-link/case_study_1/MoviesToActors.txt) file contains information on which actors worked on a specific movie, i.e., there is a link when an actor worked on that movie. Movies to genres links ([MoviesToGenres.txt](https://github.com/DEIB-GECO/NMTF-link/case_study_1/MoviesToGenres.txt)) classify the genre of each movie in the network.
The output file ([myOutFile.txt](https://github.com/DEIB-GECO/NMTF-link/results/myOutFile.txt)) contains new movie suggestions for the users, this is selected by the element equal to 1 in the column **main**.

# Contents

This repository contains all data, scripts and example results related to the NMTF-link tool. In particular, you will find:

1. folders which stores the case study number 1 ([case_study_1](https://github.com/DEIB-GECO/NMTF-link/case_study_1)), the case study number two ([case_study_2](https://github.com/DEIB-GECO/NMTF-link/case_study_2)) and the [results](https://github.com/DEIB-GECO/NMTF-link/results), i.e., the evaluation plots and the new link predictions;
2. .py files, [NMTF-link.py](https://github.com/DEIB-GECO/NMTF-link/NMTF-link.py), [Network.py](https://github.com/DEIB-GECO/NMTF-link/Network.py) and [AssociationMatrix.py](https://github.com/DEIB-GECO/NMTF-link/AssociationMatrix.py) which create classes used in other files. In particular, NMTF-link.py is used to create the network, execute and output the results, Network.py creates the network of association matrices and AssociationMatrix.py computes each association matrix.
3. A Jupyter notebook, [NMTF-link_Example](https://github.com/DEIB-GECO/NMTF-link/NMTF-link_Example.ipynb), explaining in detail an example of how to run the algorithm and comparisons between different outputs achieved by changing some parameters of the setting file. This example can be tried also online using this link: [NMTF_link_Example](https://colab.research.google.com/drive/1JWuYjppKcUiNm0bJsHTjQzYoSK6MJ7Pm?usp=sharing)

# Usage

To run the tool user should call it from terminal following way:
```
python3 NMTF-link.py
```
