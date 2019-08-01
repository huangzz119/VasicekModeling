import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.linalg
import time

np.random.seed(1)

def loss_percent_function(sim_subsample, obligors_number, low_cholesky, threshold, w):
    random_norm = np.random.normal(size=(sim_subsample, obligors_number))  # size: the shape of output
    x = np.transpose(np.matmul(low_cholesky, np.transpose(random_norm)))  # asset return value

    temp_indicator = x < threshold  # return true or false

    loss_portfolio = np.sum(w * temp_indicator, axis=1)  # add with row value, portfolio loss
    loss_percent = loss_portfolio / np.sum(w)

    return loss_percent

def Monte_Carlo_method(obligors_number, pd, rho, ead, lgd):
    loss_percent_collector = np.zeros((sim_subsample,loop_number))   #return a new array of given shape and type, filled with zeros
    corr_matrix = np.full((obligors_number,obligors_number),rho)  #return a new array of given shape and type, filled with fill_value
    np.fill_diagonal(corr_matrix,1) #fill the main diagonal of the given array of any dimensionality
    low_cholesky = scipy.linalg.cholesky(corr_matrix, lower = True) #compute the Cholesky decomposition of a matrix, A=LL*, A=U*U
    w = ead * lgd   #effective exposure
    threshold = norm.ppf(pd)   #ppf: percent point function(inverse of cdf - percentiles), probability of default

    sim_outstanding = sim_total
    loop = 0

    while sim_outstanding > 0:
        sim_temp_subsample = np.min((sim_outstanding, sim_subsample))
        sim_outstanding = sim_outstanding - sim_temp_subsample
        loss_percent = loss_percent_function(sim_temp_subsample, obligors_number, low_cholesky, threshold, w)
        loss_percent_collector[:,loop] = loss_percent    #the value of loop column turn into l_perc
        loop = loop + 1

    return loss_percent_collector.flatten()   # flat into one array

if __name__ == '__main__':
     start_time = time.time()

     sim_total = 4000000
     sim_subsample = 10000
     loop_number = int(sim_total/sim_subsample)

     percent = np.linspace(0, 1, sim_total)     #range between 0 and 1, divided into sim_total numbers
     factor = 1 / np.arange(1, sim_total+1)

     n_obligor = [1000, 100, 1001, 100]
     pd = [0.01, 0.01, 0.005, 0.01]
     rho = [0.2, 0.2, 0.2, 0.5]
     ead = [np.ones(1000), np.arange(1, 101), np.append(100, np.ones(1000)), np.repeat(np.array(np.arange(1, 6)), 20) ** 2]
     loss_percent_simulation = []

     for i in np.arange(len(n_obligor)):
         lgd = np.ones(n_obligor[i])
         temp_loss_percent = Monte_Carlo_method(n_obligor[i], pd[i], rho[i], ead[i], lgd)
         loss_percent_simulation.append(temp_loss_percent)
         print(temp_loss_percent)

     np.set_printoptions(threshold=np.nan)    #print all values of array

     running_time = time.time() - start_time
     print("running time: ", running_time)


# plot VaR
     plt.figure(0, figsize=(20, 15))
     plt.yscale("log")
     plt.ylim((0.0001, 0.1))
     plt.xlim((0.05, 0.25))

     for temp in np.arange(len(n_obligor)):
         plt.plot(np.sort(loss_percent_simulation[temp], axis=None), 1 - percent, label = 'VaR example'+str(temp+1))

     plt.xlabel('Loss percentage')
     plt.ylabel('Tail probability')
     plt.grid(which='both', linestyle=':', linewidth=1)
     plt.legend()
     plt.show()


# plot ES
     plt.figure(1, figsize=(20, 15))
     plt.yscale("log")
     plt.ylim((0.0001, 0.1))
     plt.xlim((0.05, 0.25))

     for temp in np.arange(len(n_obligor)):
         plt.plot(np.cumsum(sorted(loss_percent_simulation[temp],reverse=True))*factor, percent, label='ES example'+str(temp+1))

     plt.xlabel('Loss percentage')
     plt.ylabel('Tail probability')
     plt.grid(which='both', linestyle=':', linewidth=1)
     plt.legend()
     plt.show()



