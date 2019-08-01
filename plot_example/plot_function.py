import numpy as np
from scipy.stats import norm
import scipy.linalg


np.random.seed(1)

sim_subsample = 4000000
sim_total = 40000000
loop_number = int(sim_total/sim_subsample)
def loss_function(sim_subsample, n_obligor, low_cholesky, threshold, w, alpha):
    random_norm = np.random.normal(size=(sim_subsample, n_obligor))  # size: the shape of output
    x = np.transpose(np.matmul(low_cholesky, np.transpose(random_norm)))  # asset return value

    temp_indicator = x < threshold  # return true or false

    loss_portfolio = np.sum(w * temp_indicator, axis=1)  # add with row value, portfolio loss
    loss_portfolio = sorted(loss_portfolio, reverse=True)

    # var
    var = np.percentile(loss_portfolio, alpha*100)
    es = np.zeros(len(alpha))

    # for es
    bool = np.transpose(np.tile(loss_portfolio, (len(var), 1))) > np.tile(var, (sim_subsample, 1))
    num_bool = np.sum(bool,axis = 0)

    for i in np.arange(len(num_bool)):
        tail = loss_portfolio[:num_bool[i]]
        es[i] = sum(tail)/len(tail)

    return var, es
def Monte_Carlo_method(n_obligor, pd, rho, ead, lgd, alpha):
    var_c= np.zeros((len(alpha),loop_number))
    es_c = np.zeros((len(alpha),loop_number))

    corr_matrix = np.full((n_obligor,n_obligor),rho)  #return a new array of given shape and type, filled with fill_value
    np.fill_diagonal(corr_matrix,1) #fill the main diagonal of the given array of any dimensionality
    low_cholesky = scipy.linalg.cholesky(corr_matrix, lower = True) #compute the Cholesky decomposition of a matrix, A=LL*, A=U*U
    w = ead * lgd   #effective exposure
    threshold = norm.ppf(pd)   #ppf: percent point function(inverse of cdf - percentiles), probability of default

    sim_outstanding = sim_total
    loop = 0

    while sim_outstanding > 0:

        print("--------------------subsample "+ str(loop+1) + "-----------------------")

        sim_temp_subsample = np.min((sim_outstanding, sim_subsample))
        sim_outstanding = sim_outstanding - sim_temp_subsample
        var, es = loss_function(sim_temp_subsample, n_obligor, low_cholesky, threshold, w, alpha)
       # loss_percent_collector[:,loop] = loss_portfolio/np.sum(w)    #the value of loop column turn into l_perc

        var_c[:,loop] = var
        es_c[:,loop] = es
        loop = loop + 1

    return var_c, es_c   # flat into one array


def vesicek_method(alpha, pd, rho, ead, lgd):
    eff_exposure =  ead * lgd
    alpha_matrix = [[i] * len(ead) for i in alpha]
    temp_var = eff_exposure * norm.cdf((norm.ppf(pd) - np.sqrt(rho) * norm.ppf(alpha_matrix)) / np.sqrt(1 - rho))
    var = temp_var.sum(axis=1)
    return var
