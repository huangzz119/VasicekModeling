from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# related to eff_exposure, it related to ead and lgd

def cdtion_cgf(saddlepoint_matrix):
    temp_cgf = 1 - cdtion_pd + cdtion_pd * np.exp(eff_exposure * saddlepoint_matrix)
    return np.log(temp_cgf).sum(axis=1)

def sed_deriv_cdtion_cgf(saddlepoint_matrix):
    temp_sed_num = (1 - cdtion_pd) * np.square(eff_exposure) * cdtion_pd * np.exp(eff_exposure * saddlepoint_matrix)
    temp_sed_den = np.square( 1 - cdtion_pd + cdtion_pd * np.exp(eff_exposure * saddlepoint_matrix) )
    return (temp_sed_num / temp_sed_den).sum(axis=1)

if __name__ == '__main__':

    n_obligor = [100, 100, 1001, 100]
    pd = [0.01, 0.01, 0.005, 0.01]   # probability of default, all obligors have same pd
    rho = [0.2, 0.2, 0.2, 0.5]     # correlation between obligors, all same
    ead = [np.ones(100), np.arange(1, 101), np.append(100, np.ones(1000)),
           np.repeat(np.array(np.arange(1, 6)), 20) ** 2]
    lgd = [np.ones(100), np.ones(100), np.ones(1001), np.ones(100)]
    #common_factors = np.random.normal(1)
    common_factors = -0.9

    for temp in np.arange(len(n_obligor)):
        temp = int(np.arange(len(n_obligor))[1])

        print("-----------the "+str(temp+1)+" obligor------------------------------------")
        cdtion_pd = norm.cdf((norm.ppf(pd[temp]) - np.sqrt(rho[temp]) * common_factors) / np.sqrt(1 - rho[temp]))
        eff_exposure = ead[temp]* lgd[temp]  # effective exposure
        t_plot = np.linspace(-5, 5, 10000)

        # method 2
        t_matrix = np.array([[i] * n_obligor[temp] for i in t_plot])
        temp_first_num = eff_exposure * cdtion_pd * np.exp(eff_exposure * t_matrix)
        temp_first_den = 1 - cdtion_pd + cdtion_pd * np.exp(eff_exposure * t_matrix)
        first_deriv_discrete_t = (temp_first_num / temp_first_den).sum(axis=1)

        cdtion_cgf_value = cdtion_cgf(t_matrix)
        sed_deriv_value = sed_deriv_cdtion_cgf(t_matrix)

        x1 = np.repeat(10, len(t_plot))
        x2 = np.repeat(35, len(t_plot))

        plt.figure(0, figsize=(20, 15))
        plt.plot(t_plot, first_deriv_discrete_t, label='K`(t)')
        plt.plot(t_plot, first_deriv_discrete_t-x1, label = ' K`(t) - x1, x1=10')
        plt.plot(t_plot, first_deriv_discrete_t-x2, label = 'K`(t) - x2, x2=35')

        #plt.plot(t_plot, cdtion_cgf_value)
        #plt.plot(t_plot,sed_deriv_value)

       # plt.hlines(0, -5, 5, linestyles='dashed')
        plt.xlabel('t',  fontsize=20)
        plt.ylabel('K`(t)', fontsize=20)
        plt.title("Example " +str ( temp + 1 ) , loc="left", fontsize=25, fontweight=5, color="black")
        plt.legend(fontsize = 20)
        #plt.savefig("First_Deriv_Plot_for Example " +str ( temp + 1 ) + ".png")
        plt.show()
