# NMTF-link tool 

<!-- This tool is designed for computing **link predictions** using the **Non-negative Matrix Tri-Factorization (NMTF)** method. This is the generalization for already pre-existing code in [DEIB-GECO/NMTF-DrugRepositioning](https://github.com/DEIB-GECO/NMTF-DrugRepositioning) allowing the use of networks of any topology. -->
This tool is designed for computing network **link predictions** using a **Non-negative Matrix Tri-Factorization (NMTF)** based method generalized to allow its application on networks of any topology. 

**It is a command-line tool that uses a setting file**. 
Two example setting files called [graph_topology.tsv](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/graph_topology.tsv) are located in the folders "[case_study_1](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/)" and "[case_study_2](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_2/)" of the main directory. Examples of the [myOutFile.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/results/case_study_1/myOutFile_random_relative_error.txt) file containing new possible relations among objects of the input datasets are located in the [results](https://github.com/DEIB-GECO/NMTF-link/blob/master/results/) folder of the main directory.

To run the software:
```
python NMTF-link.py case_study_1 graph_topology.tsv
```
# Contents
This repository contains all data, scripts and example results related to the NMTF-link tool. In particular, you can find:

1. a [requirements.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/requirements.txt) file with the software requirements to run the tool;
2. the .py file [NMTF-link.py](https://github.com/DEIB-GECO/NMTF-link/blob/master/NMTF-link.py) for the tool execution, which is used to create the network, compute the predictions and output the results;
3. the folder [scripts](https://github.com/DEIB-GECO/NMTF-link/blob/master/scripts/) containing two .py files: [Network.py](https://github.com/DEIB-GECO/NMTF-link/blob/master/scripts/Network.py) and [AssociationMatrix.py](https://github.com/DEIB-GECO/NMTF-link/blob/master/scripts/AssociationMatrix.py), which create classes used in the [NMTF-link.py](https://github.com/DEIB-GECO/NMTF-link/blob/master/NMTF-link.py) file. In particular, [Network.py](https://github.com/DEIB-GECO/NMTF-link/blob/master/scripts/Network.py) creates the network of association matrices and [AssociationMatrix.py](https://github.com/DEIB-GECO/NMTF-link/blob/master/scripts/AssociationMatrix.py) computes each association matrix;
4. folders that store the case study number 1 ([case_study_1](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/)), the case study number 2 ([case_study_2](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_2/)) and their [results](https://github.com/DEIB-GECO/NMTF-link/blob/master/results/), i.e., the evaluation plots and the new link predictions;
5. a Jupyter notebook, [NMTF-link_Example.ipynb](https://github.com/DEIB-GECO/NMTF-link/blob/master/NMTF-link_Example.ipynb), explaining in detail an example of how to run the algorithm and the comparisons between different outputs obtained by changing some parameter values in the setting file. This example can be run also online at: [online NMTF_link_Example](https://colab.research.google.com/drive/1JWuYjppKcUiNm0bJsHTjQzYoSK6MJ7Pm?usp=sharing).

# Table of contents
- [Installation](#installation)
- [NMTF method](#nmtf-method)
- [Parameters to be specified by the user in the setting file](#parameters-to-be-specified-by-the-user-in-the-setting-file)
- [Other entries to be specified by the user in the setting file](#other-entries-to-be-specified-by-the-user-in-the-setting-file)
- [Setting file example](#setting-file-example)
  * [Interpretation of the setting file](#interpretation-of-the-setting-file)
- [Input network format](#input-network-format)
  

# Installation

After cheching the software requirements in the [requirements.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/requirements.txt) file, to run the NMTF-link tool the user can call it from the terminal as follows:
```
git clone https://github.com/DEIB-GECO/NMTF-link
cd NMTF-link
pip3 install -r requirements.txt
python3 NMTF-link.py
```
Additionally, an online executable example of the NMTF-link use and outputs is available here:
[online NMTF_link_Example](https://colab.research.google.com/drive/1JWuYjppKcUiNm0bJsHTjQzYoSK6MJ7Pm?usp=sharing).

# NMTF method
The NMTF-link tool implements a generalized NMTF-based method that permits an effortless factorization of all association matrices in a multipartite network. 
By simultaneously decomposing each association matrix into three factor matrices, this NMTF-based method can output new predictions for a user-specified subnetwork of the multipartite network. 
At the starting point of the algorithm, each factor matrix needs to be initialized using a user-selected **initialization strategy**. 
The NMTF is an iterative approach; thus, it updates the factor matrices until a specified **stop criterion** is reached. 
Then, the new link predictions are computed by multiplying the estimated factor matrices of the chosen subnetwork. 
To evaluate our method, the user can use the Average Precision Score (**APS**) or the Area Under the ROC (**AUROC**) curve on masked elements of the selected association matrix (choosing the **masking strategy** to be used). 

# Parameters to be specified by the user in the setting file
In the [graph_topology.tsv](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/graph_topology.tsv) file the user needs to specify the following parameters:

- **integration.strategy**, mode of integration of datasets.
When a dataset is present in several association matrices (e.g., the drugs are present in three of the four association matrices of the multipartite network in the use case 2), there are two ways to integrate its elements: either using only its objects that are shared by all association matrices (intersection option), or using all its objects, which are present in at least one association matrix (union option).

  *Options*: "intersection" or "union".
  
- **initialization**, method to initialize the factor matrices, which are the three matrices that factorize each association matrix. 

  *Options*: "random", "kmeans" or "skmeans".

- **metric**, performance evaluation metric.

  *Options*: "APS" (Average Precision Score) or "AUROC" (Area Under the ROC curve).

- **number.of.iterations**, number of maximum iterations for each run of the algorithm. 

  *Options*: any integer value.

- **type.of.masking**, to evaluate the NMTF predictions, there is the need of choosing the masking strategy to be applied on the selected association matrix. 
It can either have a completely randomized distribution of masking elements, or have the same number of masking elements per row, randomly distributed within each row. 

  *Options*: "fully_random" or "per_row_random".

- **stop.criterion**, stop criterion strategies for link prediction using the NMTF method. 

  *Options*: "maximum_metric", "relative_error" or "maximum_iterations".
    
    - "maximum_metric" option runs the algorithm 5 times with masking, chooses the iteration with best average evaluation metric, runs one more time (without masking and evaluation) until the chosen iteration and outputs the results; it also outputs evaluation plots to the main directory.
    - "relative_error" option runs the algorithm 5 times with masking, chooses the first iteration of each run with relative error < 0.001, runs one more time (without masking and evaluation) until the chosen iteration and outputs the results; it also outputs evaluation plots to the main directory.
    - "maximum_iterations" option runs the chosen number of iterations without masking and outputs the result for the last iteration. 

- **score.threshold**, minimum NMTF score value for the novel links predicted. 

  *Options*: any value between 0 and 1.

# Other entries to be specified by the user in the setting file

In the setting file, e.g. [graph_topology.tsv](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/graph_topology.tsv), the user has also to specify the datasets to be used to create the network. The structure of the dataset line is (entries separated by whitespace):
- name of left category of nodes (e.g., in the column **nodes_left** of the [example setting file of use case 1](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/graph_topology.tsv), the first category of nodes is **users**)
- name of right category of nodes (e.g., in the column **nodes_right** of the [example setting file of use case 1](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/graph_topology.tsv), the first category of nodes is **genres**)
- filename (e.g., in the column **filename** of the [example setting file of use case 1](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/graph_topology.tsv), the first file is **UsersToGenres.txt**, which has to be in the specified data directory)
- 1 or 0 value in the **main** column, indicating whether or not the datafile represents the matrix for which to compute new links.

# Setting file example

Example of [graph_topology.tsv](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/graph_topology.tsv) configuration file, from the use case number 1. The lines starting with a _hash mark_ (#) represent execution parameters. The final lines describe the topology of the multi-layer network.

| #parameters | | | |
| ------------- | ------------- | ------------- | ------------- |
| **#integration.strategy**  | intersection |  |  |
| **#initialization**  | random |  |  |
| **#metric**  | APS |  |  |
| **#number.of.iterations** | 200 |  |  |
| **#type.of.masking** | fully_random |  |  |
| **#stop.criterion**  | relative_error |  |  |
| **#score.threshold** | 0.5 |  |  |
| **#nodes_left** | **nodes_right** | **filename** | **main** |
| users |	genres |	UsersToGenres.txt |	0 |
| users |	movies |	UsersToMovies.txt	| 1 |
| movies	| actors	| MoviesToActors.txt |	0 |
| movies	| genres	| MoviesToGenres.txt	| 0 |

## Interpretation of the setting file

We used the random initialization strategy for the factor matrices, the APS (Average Precision Score) as evaluation metric, 200 maximum iterations for each run of the algorithm, the fully random masking strategy for the evaluation process and the stop criterion based on the relative error of the loss function. 
The used mode of integration of shared datasets is by intersection, i.e., only objects shared by all association matrices are considered. The new link predictions have an NMTF score (score threshold) above 0.5.

The example contains 4 node categories (**users, genres, actors and movies**). Users to genres links are reported in [UsersToGenres.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/UsersToGenres.txt); each link represents the preference of movie genres for a specific user. Users to movies links are the list of watched movies for each user, reported in [UsersToMovies.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/UsersToMovies.txt). The [MoviesToActors.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/MoviesToActors.txt) file contains information on which actors worked on a specific movie, i.e., there is a link when an actor worked on a movie. Movies to genres links ([MoviesToGenres.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/MoviesToGenres.txt)) classify the genre of each movie in the network.

The element equal to 1 in the column **main** of the [setting file](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/graph_topology.tsv) indicates the file describing the association matrix for which to compute the predictions, i.e., in the example, new movie suggestions for the users; they are stored in the output file ([myOutFile.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/results/case_study_1/myOutFile_random_relative_error.txt)). 

## Input network format
Each input file (e.g., [UsersToGenres.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/UsersToGenres.txt), [UsersToMovies.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/UsersToMovies.txt), [MoviesToActors.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/MoviesToActors.txt) ,([MoviesToGenres.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/MoviesToGenres.txt)) is a tab-separated file containing all link information for each bipartite layer of the overall input network architecture. For example, the below first five rows of [UsersToMovies.txt](https://github.com/DEIB-GECO/NMTF-link/blob/master/case_study_1/UsersToMovies.txt) show that the first column contains user names and the second column contains movie titles as follows:

| **UsersToMovies.txt** | |
|-----------------------| ------------- |
| Anna                  |	The Lighthouse |
| Anna                  |	Doctor Sleep|
| Anna                  |	Ma |
| Anna                  |	Spider-Man |
| Anna                  |	Oceans 8 |
