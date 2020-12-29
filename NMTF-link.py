import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'scripts/')
from Network import Network
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import pylab as plt
import seaborn as sns
import copy
import time
import statistics
import os 

current = os.getcwd()
filename_1 = input("Please specify the directory of your data and setting file: ")
filename_2 = input("Please specify the name of your setting file: ")

dirname_1 = os.path.join(current,filename_1,filename_2)
dirname_2 = os.path.join(current,filename_1)

#Baseline parameters 
#threshold=0
#metric = 'aps'
#max_iter =200
#stop_criterion= 'calculate'


with open(dirname_1) as f:
    for line in f:
       if line.strip().startswith("#metric"):
          s = line.strip().split("\t")
          if s[1] == "APS":
               metric="aps"
               print("\nEvaluation metric is " + '\033[1m' + "APS" + '\033[0m')
          elif s[1] == "AUROC":
               metric="auroc"
               print("\nEvaluation metric is " + '\033[1m' + "AUROC" + '\033[0m')
          else:
              print("Option '{}' not supported".format(s[1]))
              exit(-1)
       if line.strip().startswith("#number.of.iterations"):
          try: 
            s = line.strip().split("\t")
            max_iter = int(s[1])
            print("Maximum number of iteration is " + '\033[1m' + str(max_iter) + '\033[0m')
          except ValueError:
              print("Number of iterations should be integer")
              exit(-1)
       if line.strip().startswith("#stop.criterion"):
          s = line.strip().split("\t")
          if s[1] == "maximum_metric":
               stop_criterion= 'maximum_metric'
          elif s[1] == "maximum_iterations":
               stop_criterion= 'maximum_iterations'
          elif s[1] == "relative_error":
               stop_criterion= 'relative_error'
          else:
              print("Option '{}' not supported".format(s[1]))
              exit(-1)
       if line.strip().startswith("#likelyhood.threshold"):
            s = line.strip().split("\t")
            if 0<=float(s[1])<=1:
              threshold = float(s[1])
            else:
              print("Threshold for retrieval should be between 0 and 1")
              exit(-1)

metric_vals = np.zeros(max_iter//10)

if stop_criterion== 'maximum_metric':
    best_iter = 0
    best_iter_arr =[]#contains the iterations with best performance from each of 5 validation runs (j cycle)
    #cycle to find the stop criterion value
    for j in range(5):
      V=[]
      if j>0:
        verbose=False
      elif j==0:
        verbose=True
      network = Network(dirname_1,dirname_2,verbose)
      initial_error = network.get_error()
      print('\033[1m' + "Run number " + str(j+1) + " of the algorithm" + '\033[0m')
      print("initial error: {}".format(initial_error))
      for i in range(max_iter):
         network.update()
         if i % 10 == 0:
             metric_vals[i//10] = network.validate(metric)
             result = [metric_vals]
         V.append(network.validate(metric))
         print("iteration {}, ".format(i+1) + metric + " = {}".format(V[-1]))
         if V[-1] == max(V):
            best_iter=i
      X = np.arange(1, max_iter, 10)
      df = pd.DataFrame([metric_vals], columns = X).melt()
      sns.lineplot(x="variable", y="value", data=df, ci='sd')        
      plt.xlabel('Iterations')
      if metric == 'aps':
         plt.ylabel('Average Precision Score (APS)')
         if j==4:
             plt.savefig('results/aps_' + network.init_strategy + '_' + stop_criterion + '.png')
      elif metric == 'auroc':
         plt.ylabel('Area Under ROC Curve')
         if j==4:
             plt.savefig('results/auroc_' + network.init_strategy + '_' + stop_criterion + '.png')
      best_iter_arr.append(best_iter)
      best_iter = 0
      time.sleep(2) #used since otherwise random initialization gives the same result multiple times
    
    res_best_iter = statistics.median(best_iter_arr)    
    plt.axvline(x=res_best_iter, color='k', label='selected stop iteration',linestyle = 'dashed')
    plt.legend()
    plt.savefig('results/'+ metric +'_' + network.init_strategy + '_' + stop_criterion + '.png')
    plt.close("all")
    
    network = Network(dirname_1,dirname_2, mask = 0,verbose=False)
    initial_error = network.get_error()
    print('\033[1m' + "Final run without masking" + '\033[0m')
    #cycle to find new predictions on unmasked matrix
    error_f = []
    epsilon = 0
    for i in range(res_best_iter):
         network.update()
         error_f.append(network.get_error())
         if i>1:
             epsilon = abs((error_f[-1] - error_f[-2])/error_f[-2])
         print("iteration {}, relative error = {}".format(i+1, epsilon))
        
    #reconstruction of the matrix from factor matrices at the final point and output of result
    rebuilt_association_matrix = np.linalg.multi_dot([network.get_main().G_left, network.get_main().S, network.get_main().G_right.transpose()])
    new_relations_matrix= rebuilt_association_matrix - network.get_main().original_matrix
    n, m = new_relations_matrix.shape
    outF = open("results/myOutFile.txt", "w")
    for i in range(n):
      for j in range(m):
        if new_relations_matrix[i, j] >  threshold:
           line = network.get_main().left_sorted_term_list[i] + "  " + network.get_main().right_sorted_term_list[j] + "  " + str(new_relations_matrix[i, j])
           outF.write(line)
           outF.write("\n")
    outF.close()

if stop_criterion== 'relative_error':
    best_epsilon_arr =[]
    #cycle to find the stop criterion value
    for j in range(5):
      epsilon = 0
      error = []
      V = []
      if j>0:
        verbose=False
      elif j==0:
        verbose=True
      network = Network(dirname_1,dirname_2, verbose)
      initial_error = network.get_error()
      print('\033[1m' + "Run number " + str(j+1) + " of the algorithm" + '\033[0m')
      print("initial error: {}".format(initial_error))
      eps_iter=[]
      for i in range(max_iter):
         network.update()
         error.append(network.get_error())
         V.append(network.validate(metric))
         if i % 10 == 0:
             metric_vals[i//10] = network.validate(metric)
             result = [metric_vals]
         if i>1:
             epsilon=abs((error[-1] - error[-2])/error[-2])
             if epsilon<0.001:
                eps_iter.append(i)
         print("iteration {}, relative error = {}".format(i+1, epsilon))
      
      X = np.arange(1, max_iter, 10)
      df = pd.DataFrame([metric_vals], columns = X).melt()
      sns.lineplot(x="variable", y="value", data=df, ci='sd')
      plt.xlabel('Iterations')
      if metric == 'aps':
         plt.ylabel('Average Precision Score (APS)')
         if j==4:
             plt.savefig('results/aps_' + network.init_strategy + '_' + stop_criterion + '.png')
      elif metric == 'auroc':
         plt.ylabel('Area Under ROC Curve')
         if j==4:
             plt.savefig('results/auroc_' + network.init_strategy + '_' + stop_criterion + '.png')
      best_epsilon_arr.append(eps_iter[0])
      time.sleep(2) #used since otherwise random initialization gives the same result multiple times
    
    res_best_epsilon = statistics.median(best_epsilon_arr)
    plt.axvline(x=res_best_epsilon, color='k', label='selected stop iteration',linestyle = 'dashed')
    plt.legend()
    plt.savefig('results/'+ metric +'_' + network.init_strategy + '_' + stop_criterion + '.png')
    plt.close("all")
    
    network = Network(dirname_1,dirname_2, mask = 0, verbose=False)
    initial_error = network.get_error()
    prev_error=initial_error
    print('\033[1m' + "Final run without masking, stop at iteration: " + str(res_best_epsilon) + '\033[0m')
    #cycle to find new predictions on unmasked matrix
    error_f=[]
    epsilon=0
    for i in range(res_best_epsilon):
         network.update()
         error_f.append(network.get_error())
         if i>1:
             epsilon=abs((error_f[-1] - error_f[-2])/error_f[-2])
         print("iteration {}, relative error = {}".format(i+1, epsilon))
        
    #reconstruction of the matrix from factor matrices at the final point and output of result
    rebuilt_association_matrix = np.linalg.multi_dot([network.get_main().G_left, network.get_main().S, network.get_main().G_right.transpose()])
    new_relations_matrix= rebuilt_association_matrix - network.get_main().original_matrix
    n, m = new_relations_matrix.shape
    outF = open("results/myOutFile.txt", "w")
    for i in range(n):
      for j in range(m):
        if new_relations_matrix[i, j] >  threshold:
           line = network.get_main().left_sorted_term_list[i] + "  " + network.get_main().right_sorted_term_list[j] + "  " + str(new_relations_matrix[i, j])
           outF.write(line)
           outF.write("\n")
    outF.close()

elif stop_criterion== 'maximum_iterations':
      network = Network(dirname_1,dirname_2, mask = 0, verbose=True)
      initial_error = network.get_error()
      print('\033[1m' + "Unique run of the algorithm without masking" + '\033[0m')
      print("initial error: {}".format(initial_error))
      error = []
      for i in range(max_iter):
         network.update()
         error.append(network.get_error())
         if i % 10 == 0:
             metric_vals[i//10] = network.validate(metric)
             result = [metric_vals]
         if i>1:
             epsilon=abs((error[-1] - error[-2])/error[-2])
         print("iteration {}, relative error = {}".format(i+1, epsilon))
        
      #reconstruction of the matrix from factor matrices and output of result
      rebuilt_association_matrix = np.linalg.multi_dot([network.get_main().G_left, network.get_main().S, network.get_main().G_right.transpose()])
      new_relations_matrix= rebuilt_association_matrix - network.get_main().original_matrix
      n, m = new_relations_matrix.shape
      outF = open("results/myOutFile.txt", "w")
      for i in range(n):
        for j in range(m):
          if new_relations_matrix[i, j] >  threshold:
             line = network.get_main().left_sorted_term_list[i] + "  " + network.get_main().right_sorted_term_list[j] + "  " + str(new_relations_matrix[i, j])
             outF.write(line)
             outF.write("\n")
      outF.close()





