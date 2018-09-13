import numpy as np
import matplotlib.pyplot as plt
import parse_matlab_data
import time
import pandas


path_to_data = '/Users/stevecharczynski/workspace/python_ready_data'
conditions  = np.loadtxt(path_to_data + '/conditions.csv', delimiter=',', dtype='int')

c1 = np.zeros((500, 2435))
c2 = np.zeros((500, 2435))
c3 = np.zeros((500, 2435))
c4 = np.zeros((500, 2435))



for i in range(conditions.shape[0]):
    for v in range(conditions.shape[1]):
        if conditions[i][v] == 1:
            c1[i][v] = 1
        elif conditions[i][v] == 2:
            c2[i][v] = 1
        elif conditions[i][v] == 3:
            c3[i][v] = 1
        elif conditions[i][v] == 4:
            c4[i][v] = 1
        elif conditions[i][v] == 0:
            pass
        else:
            print ("error")

conditions = [c1, c2, c3, c4]

print (type(conditions))