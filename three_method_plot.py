import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statistics import mean, stdev
import scipy.linalg
import time
import pandas as pan


############################# Portfolio Information ##############################
#### example 1 #########
n_obligor = 100
pd = 0.01
rho = 0.2
lgd = np.ones(n_obligor)
ead = np.ones(n_obligor)

#### example 2 #########

n_obligor = 100
pd = 0.1
rho = 0.2
lgd = np.ones(n_obligor)
ead = np.arange(1, 101)

#### example 3 #########
n_obligor = 1001
pd = 0.005
rho = 0.2
lgd = np.ones(n_obligor)
ead = np.append(100, np.ones(1000))

#### example 4 #########
n_obligor = 100
pd = 0.01
rho = 0.5
lgd = np.ones(n_obligor)
ead = np.repeat(np.array(np.arange(1, 6)), 20) ** 2

#####Functions #########################################################################################################################
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

    return np.sort(loss_percent_collector.flatten(), axis=None)

def Monte_Carlo_interval(x_loss_percent_all, y_probability_MC):
    x_loss_percent_MC = sorted(list(set( x_loss_percent_all)) )   # possible values
    y_probability_MC_high = []
    y_probability_MC_low = []
    std_error_collector = []
    for i in np.arange(len(x_loss_percent_MC)):
        temp_index = list( np.where (x_loss_percent_all == x_loss_percent_MC[i])[0] )
        temp_y_value = y_probability_MC[temp_index]
        if len(temp_index) == 1:
            std_error = 0.000001
            temp_interval_upper = mean(temp_y_value) + (1.96*0.000001) / np.sqrt(len(temp_index))
            temp_interval_low = mean(temp_y_value) - (1.96*0.000001) / np.sqrt(len(temp_index))
        else:
            std_error = stdev(temp_y_value)/np.sqrt(len(temp_index))
            temp_interval_upper = mean(temp_y_value) + (1.96*stdev(temp_y_value)) / np.sqrt(len(temp_index))
            temp_interval_low = mean(temp_y_value) - (1.96*stdev(temp_y_value)) / np.sqrt(len(temp_index))
        y_probability_MC_high.append(temp_interval_upper)
        y_probability_MC_low.append(temp_interval_low)
        std_error_collector.append(std_error)

    return x_loss_percent_MC, y_probability_MC_high, y_probability_MC_low,std_error_collector

"""
mean = np.mean(loss_percent_collector, axis = 0)
var = np.var(loss_percent_collector, axis = 0)
std_error = np.sqrt(var/loss_percent_collector.shape[0])
std = np.std(loss_percent_collector, axis = 0)

ind = np.lexsort((std_error,mean))
sort_mean = np.array([mean[i] for i in ind])
sort_std_error = np.array([std_error[i] for i in ind])

CI_up = sort_mean + 1.96*sort_std_error
CI_low = sort_mean - 1.96*sort_std_error

diff = (loss_percent_collector-sort_mean)>0   # loss > mean
loss_percent = np.sum(diff,axis=0)/loss_percent_collector.shape[0]
"""

def vesicek_method(alpha, pd, rho, ead, lgd):
    eff_exposure =  ead * lgd
    alpha_matrix = [[i] * len(ead) for i in alpha]
    temp_var = eff_exposure * norm.cdf((norm.ppf(pd) - np.sqrt(rho) * norm.ppf(alpha_matrix)) / np.sqrt(1 - rho))
    var = temp_var.sum(axis=1)
    return var/sum(eff_exposure)


def tail_prob_for_one_cf(x_loss_percent, common_factor, n_obligor, pd, rho, ead, lgd):

    eff_exposure = ead * lgd  # array 1x5 values
    cdtion_pd = norm.cdf(
        (norm.ppf(pd) - np.sqrt(rho) * common_factor) / np.sqrt(1 - rho))  # conditional probability of default

    #here is value at risk
    x = x_loss_percent * sum(eff_exposure)

    # first_deriv_cdtion_cgf_discrete_t
    t_value = np.linspace(-1, 10, 100000)
    t_matrix = np.array([[i] * n_obligor for i in t_value])
    # first derivative condition CGF calculate
    temp_first_num = eff_exposure * cdtion_pd * np.exp(eff_exposure * t_matrix)
    temp_first_den = 1 - cdtion_pd + cdtion_pd * np.exp(eff_exposure * t_matrix)
    first_deriv_discrete_t = (temp_first_num / temp_first_den).sum(axis=1)
    # saddlepoint_calculate

    x_matrix = np.array([[i] * len(first_deriv_discrete_t) for i in x])
    diff = first_deriv_discrete_t - x_matrix

    temp_indicator = diff < 0.000001
    new_t_value = temp_indicator * t_value  # turn some t to zero
    saddlepoint = []
    for i in np.arange(len(x)):
        x_nonzero = new_t_value[i].nonzero()
        temp_sp = new_t_value[i][x_nonzero][-1]
        saddlepoint.append(temp_sp)
    saddlepoint = np.array(saddlepoint)

    saddlepoint_matrix = np.array( [[i] * n_obligor for i in saddlepoint])

    def cdtion_cgf(saddlepoint_matrix):
        temp_cgf = 1 - cdtion_pd + cdtion_pd * np.exp(eff_exposure * saddlepoint_matrix)
        return np.log(temp_cgf).sum(axis=1)

    def sed_deriv_cdtion_cgf(saddlepoint_matrix):
        temp_sed_num = (1 - cdtion_pd) * np.square(eff_exposure) * cdtion_pd * np.exp(eff_exposure * saddlepoint_matrix)
        temp_sed_den = np.square( 1 - cdtion_pd + cdtion_pd * np.exp(eff_exposure * saddlepoint_matrix) )
        return (temp_sed_num / temp_sed_den).sum(axis=1)

    z_l = np.sign(saddlepoint) * np.sqrt(2 * ( x * saddlepoint - cdtion_cgf(saddlepoint_matrix) ))
    z_w = saddlepoint * np.sqrt(sed_deriv_cdtion_cgf(saddlepoint_matrix))
    tail_prob = 1 - norm.cdf(z_l) + norm.pdf(z_l) * ( (1 / z_w) - (1 / z_l) )

    df = pan.DataFrame({"loss_percent":x_loss_percent, "loss_value":x, "saddlepoint":saddlepoint, "tail_prob":tail_prob, "z_l":z_l,"z_w":z_w,"1/z_w-1/z_l":1/z_w-1/z_l})
    #df.to_csv("Common factor_" + str(common_factor)+".csv", sep="\t", encoding="utf-8")

    return tail_prob, df

"""
    plt.figure(2, figsize=(20, 15))
    plt.yscale("log")
    plt.plot(x_loss_percent, tail_prob, label='Common Factor: '+str(common_factor))
    # plt.ylim((0.001, 0.1))
    # plt.xlim((0.1, 0.7))
    plt.title("Example 2", loc="left", fontsize=25, fontweight=5, color="black")
    plt.xlabel('Loss percentage', fontsize=30)
    plt.ylabel('Condition Tail probability', fontsize=30)
    plt.grid(which='both', linestyle=':', linewidth=1)
    plt.legend(fontsize=30)
    plt.savefig("Condition for Common Factor" + str(common_factor)+".png")
    plt.show()

    print ("saddlepoint and tail probability in this common factor:")
    print (df.head())
"""

def tail_prob_combine(common_factors_collector, weights_collector, x_loss_percent, n_obligor, pd, rho, ead, lgd):
    weights_tail_probability_collector = []

    # change the common factor to interval [-5,5]
    interval_change_collector = np.array(common_factors_collector)
    prob_dens_collector = norm.pdf(interval_change_collector)
    df_collector = []

    for i in np.arange(len(interval_change_collector)):
        common_factor = interval_change_collector[i]
        prob_dens = prob_dens_collector[i]
        weights = weights_collector[i]

        print("------common factor " + str( i + 1) + "-------------------------------------------------")
        print("common factor is: ", common_factor)
        print("prob density of common factor is: ", prob_dens)
        print("weight is: ", weights)

        tail_prob_condition, df = tail_prob_for_one_cf(x_loss_percent, common_factor, n_obligor, pd, rho, ead, lgd)
        df_collector.append(df)

        tail_prod_den = tail_prob_condition*prob_dens
        weight_tail_prob_cond = (tail_prod_den * weights)

        weights_tail_probability_collector.append(weight_tail_prob_cond.tolist())


    tail_prob_uncondition = np.sum(weights_tail_probability_collector, axis=0)  # add by column
    print("-----------uncondition tail probability--------------------------------------")
   # print(tail_prob_uncondition)
    return tail_prob_uncondition,df_collector



############################################## For Monte Carlo ###################

sim_subsample = 40000
sim_total = 400000
loop_number = int(sim_total/sim_subsample)

y_probability_MC = np.linspace(1, 0, sim_total)
x_loss_percent_all = Monte_Carlo_method(n_obligor, pd, rho, ead, lgd)  # from small to large
#x_loss_percent_MC, y_probability_MC_high, y_probability_MC_low, std_error_collector = Monte_Carlo_interval(x_loss_percent_all, y_probability_MC)

############################################ For Vesicek #########################

y_probability_Vesicek = np.linspace(0, 0.5, 10000)
x_loss_percent_Vesicek = vesicek_method(y_probability_Vesicek, pd, rho, ead, lgd)

########################################### For Saddlepoint #######################

common_factors_collector = np.random.normal(0, 1, 1)
weights_collector = norm.pdf(common_factors_collector)

# interval [-5,5]
common_factors_collector20 = [  -0.993128599,
    -0.963971927,
    -0.912234428,
    -0.839116972,
    -0.746331906,
    -0.636053681,
    -0.510867002,
    -0.373706089,
    -0.227785851,
    -0.076526521,
    0.076526521,
    0.227785851,
    0.373706089,
    0.510867002,
    0.636053681,
    0.746331906,
    0.839116972,
    0.912234428,
    0.963971927,
    0.993128599 ]
weights_collector20 = [0.017614007,
0.04060143,
0.062672048,
0.083276742,
0.10193012,
0.118194532,
0.131688638,
0.142096109,
0.149172986,
0.152753387,
0.152753387,
0.149172986,
0.142096109,
0.131688638,
0.118194532,
0.10193012,
0.083276742,
0.062672048,
0.04060143,
0.017614007]

common_factors_collector50 = [-0.998866404,
-0.994031969,
-0.985354084,
-0.972864385,
-0.956610955,
-0.936656619,
-0.913078557,
-0.88596798,
-0.855429769,
-0.821582071,
-0.784555833,
-0.744494302,
-0.701552469,
-0.655896466,
-0.607702927,
-0.557158305,
-0.504458145,
-0.449806335,
-0.393414312,
-0.335500245,
-0.276288194,
-0.216007237,
-0.15489059,
-0.093174702,
-0.031098338,
0.031098338,
0.093174702,
0.15489059,
0.216007237,
0.276288194,
0.335500245,
0.393414312,
0.449806335,
0.504458145,
0.557158305,
0.607702927,
0.655896466,
0.701552469,
0.744494302,
0.784555833,
0.821582071,
0.855429769,
0.88596798,
0.913078557,
0.936656619,
0.956610955,
0.972864385,
0.985354084,
0.994031969,
0.998866404]
weights_collector50 = [
0.002908623,
0.006759799,
0.010590548,
0.014380823,
0.018115561,
0.021780243,
0.025360674,
0.028842994,
0.032213728,
0.035459836,
0.038568757,
0.041528463,
0.044327504,
0.046955051,
0.049400938,
0.051655703,
0.053710622,
0.055557745,
0.057189926,
0.05860085,
0.059785059,
0.060737971,
0.0614559,
0.061936067,
0.062176617,
0.062176617,
0.061936067,
0.0614559,
0.060737971,
0.059785059,
0.05860085,
0.057189926,
0.055557745,
0.053710622,
0.051655703,
0.049400938,
0.046955051,
0.044327504,
0.041528463,
0.038568757,
0.035459836,
0.032213728,
0.028842994,
0.025360674,
0.021780243,
0.018115561,
0.014380823,
0.010590548,
0.006759799,
0.002908623]


x_loss_percent_Saddlepoint1 = np.linspace(0.05, 0.35, 100)

y_probability_Saddlepoint1,df_collector = tail_prob_combine(common_factors_collector20, weights_collector20, x_loss_percent_Saddlepoint1, n_obligor, pd, rho, ead, lgd)

########################################### Plot ##################################

plt.figure(2, figsize=(20, 15))
plt.yscale("log")
#plt.plot(x_loss_percent_Saddlepoint1, tail_prob)
#plt.ylim((0.0001, 0))
plt.xlim((0.05, 0.25))

#plt.plot(x_loss_percent_Saddlepoint1,df_collector[1].tail_prob.values,"b.-", label = "Condition at common factor: "+str(common_factors_collector20[1]))
#plt.plot(x_loss_percent_Saddlepoint1,df_collector[8].tail_prob.values,"g-.",label = "Condition at common factor: "+str(common_factors_collector20[8]))
#plt.plot(x_loss_percent_Saddlepoint1,df_collector[12].tail_prob.values, "m--",label = "Condition at common factor: "+str(common_factors_collector20[12]))
#plt.plot(x_loss_percent_Saddlepoint1,df_collector[18].tail_prob.values, "y:",label = "Condition at common factor: "+str(common_factors_collector20[18]))

plt.plot(x_loss_percent_Saddlepoint1, y_probability_Saddlepoint1, label='Saddlepoint Method')
plt.plot(x_loss_percent_Vesicek,y_probability_Vesicek, label='Vasicek Method' )
plt.plot(x_loss_percent_all, y_probability_MC, label = "Monte Carlo result")
#plt.plot(x_loss_percent_MC, y_probability_MC_high, label='Monte Carlo Intervel high')
#plt.plot(x_loss_percent_MC, y_probability_MC_low, label='Monte Carlo Intervel low')
plt.xlabel('Loss percentage', fontsize = 30)
plt.ylabel('Tail probability', fontsize = 30)
#plt.title("Example 2 for Saddlepoint method", loc="left", fontsize=35, fontweight=5, color="black")
plt.grid(which='both', linestyle=':', linewidth=1)
plt.legend(fontsize=20)

plt.savefig("exap4" + ".png")
plt.show()
