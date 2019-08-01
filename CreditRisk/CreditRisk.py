import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
import datetime
import sys
import pandas as pan

# Credit Swiss Financial Product Reference Portfolio
# Exposure
E = np.array([358475, 1089819, 1799710, 1933116, 2317327, 2410929, 2652184, 2957685, 3137989, 3204044, 4727724, 4830517,
              4912097, 4928989, 5042312, 5320364, 5435457, 5517586, 5764596, 5847845, 6466533, 6480322, 7727651, 15410906, 20238895])

# Mean Default rate
P = np.array([0.3, 0.3, 0.1, 0.15, 0.15, 0.15, 0.3, 0.15, 0.05, 0.05, 0.015, 0.05, 0.05, 0.3, 0.1, 0.075, 0.05, 0.03, 0.075, 0.03, 0.3, 0.3, 0.016, 0.1, 0.075])
# Standard deviation of default rate
s = P/ 2
# Factor weights
w = np.array([[0.5, 0.25, 0.25, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 0.75, 0.25, 0.5, 0.75],
[0.3, 0.25, 0.25, 0.05, 0.1, 0.2, 0.1, 0.25, 0.25, 0.1, 0.1, 0.2, 0.25, 0.1, 0.25, 0.1, 0.2, 0.1, 0.25, 0.1, 0.25, 0.1, 0.25, 0.2, 0.1],
[0.1, 0.25, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.25, 0.05, 0.1, 0.1, 0.25, 0.1, 0.3, 0.05, 0.1, 0.1, 0.2, 0.1, 0.2, 0.05, 0.2, 0.1, 0.1],
[0.1, 0.25, 0.3, 0.1, 0.3, 0.2, 0.55, 0.3, 0.25, 0.1, 0.3, 0.2, 0.25, 0.55, 0.2, 0.1, 0.2, 0.3, 0.3, 0.55, 0.3, 0.1, 0.3, 0.2, 0.05]])

#用于画图
def plot_pdf(data,title):
    #record the plot start
    print('plotting... starting time: ' + str(datetime.datetime.now()))

    plt.figure()

    """
    在python console中默认为交互模式,在交互模式下：

    plt.plot(x) 或plt.imshow(x) 是直接出图像，不需要plt.show()
    如果在脚本中使用ion(),命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。
    要想防止这种情况，需要在plt.show()之前加上ioff()命令。
    
    在阻塞模式下：
    打开一个窗口以后必须关掉才能打开下一个新的窗口。这种情况下，默认是不能像Matlab一样同时开很多窗口进行对比的。
    plt.plot(x)或plt.imshow(x)是直接出图像，需要plt.show()后才能显示图像
    """

    plt.ioff()

    # 用非参数估计：核函数的方法来画 loss of portfolio的概率密度函数，by gaussian kernel density estimate
    bw = 0.01  #bandwidth parameters for kde, 用在核函数中的参数，事先设定的volumn
    kde = st.gaussian_kde(data,bw_method=bw)
    t_range = np.linspace(-1,10**8,1000)
    plt.plot(t_range,kde(t_range),lw=1,color='green',alpha=0.7)
    plt.xlim(0,10**8)
    plt.xlabel('Loss')
    plt.ylabel('probability density')

    #选取百分之99的分位数作为VaR， value at risk
    VaR = np.percentile(data,99)
    plt.axvline(VaR, color='orange', linestyle='solid')
    #  .2f print only the first 2 digits after the point.
    plt.text(VaR,10**-7,
             " VaR at 99% = " + \
             "{:,.0f}".format(VaR) + \
             "\n (" + "{:.2f}".format(VaR / E.sum()*100) \
             + "% of total exposure)",
             color='orange',
             horizontalalignment = 'left',
             verticalalignment='center')
    plt.grid(True)
    plt.title(title, y=1.05)
    plt.show()

    return

# 用于记录开始和结束的时间
def timestamp(name, mark):
    # class style
    print('Procedure ' + style.DARKCYAN + name + style.END + ' '+ mark + '... at ' + str(datetime.datetime.now()))
    return

# 用于输出VaR和ES
def printLoss(data):
    # calculate the VaR and ES
    VaR = np.percentile(data,99)
    n_tail = len(data[data>=VaR])
    ES = np.sum(data[data>=VaR]/n_tail)
    # .0f: 不用输出小数部分
    print('99% VaR = ' + "{:,.0f}".format(VaR))
    print('ES = ' + "{:,.0f}".format(ES))
    return

# 得到行列的方法：
# 当 x 是[[],[],[]], x.shape[1]:得到列数， x.shape[0]:得到行数
# 当 x 是[], len(x)得到x的长度

def printAttrib(data_port, data_obligor):

    # Compute the q-th percentile of the data along the specified axis
    VaR = np.percentile(data_port,99)

    # 损失大于VaR的个数
    n_tail = len(data_port[data_port>=VaR])
    mask = data_port >= VaR

    # calculate Gaussian Kernel
    h = len(data_port)**-0.2 * np.std(data_port)
    # the number of obligor, shape[1]:column value
    d = data_obligor.shape[1]
    kernel = 1 / np.sqrt(2*np.pi) * np.exp(-1/2*((data_port-VaR)/h)**2) #len(kernel)= 1000

    # increase the dimension of the existing array by one more dimension,
    # make it as row vector by inserting an axis along first dimension: arr[np.newaxis, :]
    # make it as column vector by inserting an axis along second dimension: arr[:, np.newaxis]

    # repeat(a, repeats, axis=1): repeat a repeats times along axis = 1,  thus = shape[1000,25]
    # 每个obligor的损失乘上对应的pdf值

    VaR_attrib = data_obligor * np.repeat(kernel[:,np.newaxis], d, axis = 1) / np.sum(kernel)
    VaR_attrib = np.sum(VaR_attrib,axis=0) #得到的是每个obligor的总loss，在这nSim的模拟中

    #保留损失大于VaR的那些loss
    ES_attrib = data_obligor * np.repeat(mask[:,np.newaxis],d,axis=1) / n_tail
    ES_attrib = np.sum(ES_attrib,axis=0)

    attrib = np.zeros((d,2))
    attrib[:,0] = VaR_attrib
    attrib[:,1] = ES_attrib
    pan.options.display.float_format = '{:,.0f}'.format

    df = pan.DataFrame(attrib,
                       index = np.arange(d)+1,
                       columns = ['attributed VaR', 'attributed ES'])
    # Exposure of all obligors
    df_E = pan.DataFrame(E,
                         index = np.arange(d)+1,
                         columns = ['Exposure'])
    df = pan.concat([df_E, df], axis=1)
    df.loc['Total']= df.sum()
    print(df)
    return


# from binomial distribution
# random.binomial(n, p, size): n trials and p probability of success, testing size times
def IMF_Figure1(nSim,printGraphON,printLossON,printAttribON):

    # sys._getframe().f_code.co_name: 获取当前的函数名
    timestamp(sys._getframe().f_code.co_name,'start')

    np.random.seed(1)

    # b = tile(a,(m,n)):即是把a数组里面的元素复制n次放进一个数组c中，然后再把数组c复制m次放进一个数组b中
    # random.binomial(n, p): n trials and p probability of success
    d = np.random.binomial(1,np.tile(P,(nSim,1)))

    # the exposure in the trails
    L_obligor = d * E

    # add exposure together, and there is nSim values, this is the loss of portfolio
    L_port = L_obligor.sum(axis=1)

    # data.flatten(order='C'): means to flatten in row-major order
    if printGraphON:#plot pdf graph
        graph_title = 'Fixed Probabilities, Bernoulli Defaults (' + str(nSim) +' trials)'
        plot_pdf(L_port.flatten(),graph_title)

    if printLossON:
        print('nSim = ' + "{:,.0f}".format(nSim))
        printLoss(L_port)
    if printAttribON:
        print('nSim = ' + "{:,.0f}".format(nSim))
        printAttrib(L_port,L_obligor)

    timestamp(sys._getframe().f_code.co_name,'end')
    print()
    return

# from poisson distribution
# random.poisson(lam = 1.0, size): draw each lam with size times
def IMF_Figure2(nSim,printGraphON,printLossON,printAttribON):
    timestamp(sys._getframe().f_code.co_name,'start')
    np.random.seed(1)
    # choose from poisson distribution
    d = np.random.poisson(lam=np.tile(P,(nSim,1)))
    L_obligor = d * E
    L_port = L_obligor.sum(axis=1)
    if printGraphON: #plot pdf graph
        graph_title = 'Fixed Probabilities, Poisson Defaults (' + str(nSim) +' trials)'
        plot_pdf(L_port.flatten(),graph_title)
    if printLossON:
        print('nSim = ' + "{:,.0f}".format(nSim))
        printLoss(L_port)
    if printAttribON:
        print('nSim = ' + "{:,.0f}".format(nSim))
        printAttrib(L_port,L_obligor)
    timestamp(sys._getframe().f_code.co_name,'end')
    print()
    return

# Ratio of Bernoulli VaR to Poisson VaR
def IMF_Figure3(nSim):
    timestamp(sys._getframe().f_code.co_name,'start')

    ratio = np.zeros(100)
    i =0
    for pd in np.linspace(0,0.3,100):
        i =i + 1
        np.random.seed(1)
        # from binomial
        # the success probability of this binomial is pd, and generate rv with size (nSim, E.size)
        d_ber = np.random.binomial(1,pd,(nSim,E.size))
        L_ber = d_ber * E  # shape(1000,25)
        L_port_ber = L_ber.sum(axis=1)  # 将每次simulation 的损失相加，就是 loss of portfolio

        #from poisson, with lamda = pd, generate rv with same size
        d_poi = np.random.poisson(lam=pd,size=(nSim,E.size))
        L_poi = d_poi * E
        L_port_poi = L_poi.sum(axis=1)

        # 来自binomial的VaR/来自poisson的VaR, 给ratio赋值
        if np.percentile(L_port_poi,99) != 0:
            ratio[i-1] = np.percentile(L_port_ber,99)/np.percentile(L_port_poi,99)
    plt.figure(1)
    plt.plot(np.linspace(0,3000,100),ratio, color='green',alpha=0.7)
    plt.title('Ratio of Bernoulli VaR to Poisson VaR', y =1.05)
    plt.grid()
    plt.ylim((0.8,1.05))
    plt.show()
    timestamp(sys._getframe().f_code.co_name,'end')
    return

# OMG gamma distribution with the binomial
# Random Probabilities with draws of Factors and draws of Defaults
def IMF_Figure4(nSim_g,nSim_D,printGraphON,printLossON,printAttribON):
    timestamp(sys._getframe().f_code.co_name,'start')
    np.random.seed(1)

    # Factor weights, 一共有4种factors
    A = np.transpose(w)

    # Standard deviation of default rate and Mean Default rate
    b = s**2/P**2

    # minimize 0.5 * ||A x - b||**2
    # lsq_linear(A, b, bounds=(-inf, inf), lsmr_tol='auto', verbose=0): A:design matrix, b:target vector
    # ‘auto’: the tolerance will be adjusted based on the optimality of the current iterate, which can speed up the optimization process, but is not always reliable.
    # verbose=0: work silently (default).
    res = opt.lsq_linear(A, b, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
    # res.x: solution found, shape(4)
    alpha = np.repeat(1/(res.x)[np.newaxis,:],nSim_g,axis=0)
    beta = np.repeat((res.x)[np.newaxis,:],nSim_g,axis=0)

    #random.gamma(shape , scale, size): parameter: shape and scale, testing size times
    factor = np.random.gamma(alpha,beta)

    # w means factor weights, P means default rate,
    # matmul(x1,x2): Matrix product of two arrays (100,4)*(4,25) = (100,25)
    # calculate the new prob of default
    pd = np.matmul(factor,w) * P
    pd[pd>1] = 1

    # OMG
    d = np.random.binomial(1,np.repeat(pd[np.newaxis,:,:],nSim_D,axis=0))
    L_obligor = d * E
    L_port = L_obligor.sum(axis=2)

    if printGraphON:  #plot pdf graph
        graph_title = 'Random Probabilities with '+ str(nSim_g) +' draws of Factors \n x '+ str(nSim_D) +' draws of Defaults'
        plot_pdf(L_port.flatten(),graph_title)
    if printLossON:
        print('nSim = ' + "{:,.0f}".format(nSim_g) + ' x '+ "{:,.0f}".format(nSim_D))
        printLoss(L_port)
    if printAttribON:
        print('nSim = ' + "{:,.0f}".format(nSim_g) + ' x '+ "{:,.0f}".format(nSim_D))
        printAttrib(L_port.flatten(),L_obligor.flatten().reshape((nSim_g*nSim_D,L_obligor.shape[2])))
    timestamp(sys._getframe().f_code.co_name,'end')
    print()
    return

# gamma distribution with different binomial
# Random Probabilities trials
def IMF_Figure4b(nSim,printGraphON,printLossON,printAttribON):
    timestamp(sys._getframe().f_code.co_name,'start')
    np.random.seed(1)
    A = np.transpose(w)
    b = s**2/P**2
    res = opt.lsq_linear(A, b, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
    alpha = np.repeat(1/res.x[np.newaxis,:],nSim,axis=0)
    beta = np.repeat(res.x[np.newaxis,:],nSim,axis=0)

    factor = np.random.gamma(alpha,beta)

    pd = np.matmul(factor,w) * P
    pd[pd>1] = 1

    # diff in here random.binomial
    d = np.random.binomial(1,pd)
    L_obligor = d * E
    L_port = L_obligor.sum(axis=1)
    if printGraphON: #plot pdf graph
        graph_title = 'Random Probabilities (' + str(nSim) +' trials)'
        plot_pdf(L_port.flatten(),graph_title)
    if printLossON:
        print('nSim = ' + "{:,.0f}".format(nSim))
        printLoss(L_port)
    if printAttribON:
        print('nSim = ' + "{:,.0f}".format(nSim))
        printAttrib(L_port,L_obligor)
    timestamp(sys._getframe().f_code.co_name,'end')
    print()
    return

# Bernoulli Defaults, Uncorrelated Factors
def IMF_Figure5_Bernoulli(nSim,printGraphON,printLossON,printAttribON):
    timestamp(sys._getframe().f_code.co_name,'start')
    np.random.seed(1)

    sigma_k = np.matmul(w,s)/np.matmul(w,P)
    var_k = sigma_k**2
    alpha = np.repeat(1/var_k[np.newaxis,:],nSim,axis=0)
    beta = np.repeat(var_k[np.newaxis,:],nSim,axis=0)
    factor = np.random.gamma(alpha,beta)
    pd = np.matmul(factor,w) * P
    pd[pd>1] = 1
    d = np.random.binomial(1,pd)
    L_obligor = d * E
    L_port = L_obligor.sum(axis=1)
    if printGraphON: #plot pdf graph
        graph_title = 'Bernoulli Defaults, Uncorrelated Factors'
        plot_pdf(L_port.flatten(),graph_title)
    if printLossON:
        print('nSim = ' + "{:,.0f}".format(nSim))
        printLoss(L_port)
    if printAttribON:
        print('nSim = ' + "{:,.0f}".format(nSim))
        printAttrib(L_port,L_obligor)
    timestamp(sys._getframe().f_code.co_name,'end')
    print()
    return

# Correlated Sectors, Inter-Sector Covariance
def IMF_Figure7(nSim,cov_values,printGraphON,printLossON,printAttribON):
    timestamp(sys._getframe().f_code.co_name,'start')

    for i, cov in enumerate(cov_values):
        print('Covariance = ' + str(cov))
        np.random.seed(1)

        A = np.transpose(w)
        b = s**2/P**2
        res = opt.lsq_linear(A, b, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
        alpha = np.repeat(1/res.x[np.newaxis,:],nSim,axis=0)
        G = np.random.gamma(1/cov,cov,nSim)
        alpha_rnd =alpha * np.repeat(G[:,np.newaxis],w.shape[0],axis=1)

        beta = np.repeat(res.x[np.newaxis,:],nSim,axis=0)

        factor = np.random.gamma(alpha_rnd,beta)

        pd = np.matmul(factor,w) * P
        pd[pd>1] = 1
        d = np.random.binomial(1,pd)
        L_obligor = d * E
        L_port = L_obligor.sum(axis=1)
        if printGraphON:   #plot pdf graph
            graph_title = 'Correlated Sectors, Inter-Sector Covariance = ' +"{:,.1f}".format(cov)
            plot_pdf(L_port.flatten(),graph_title)
        if printLossON:
            print('nSim = ' + "{:,.0f}".format(nSim))
            printLoss(L_port)
        if printAttribON:
            print('nSim = ' + "{:,.0f}".format(nSim))
            printAttrib(L_port,L_obligor)
        timestamp(sys._getframe().f_code.co_name,'end')
        print()
    return

def IMF_Figure4b_seed(nSim):
    timestamp(sys._getframe().f_code.co_name,'start')
    attrib_VaR = np.zeros((len(E),30))
    attrib_ES = np.zeros((len(E),30))
    for i, seed in enumerate(np.arange(30)):
        np.random.seed(seed)
        A = np.transpose(w)
        b = s**2/P**2
        res = opt.lsq_linear(A, b, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
        alpha = np.repeat(1/res.x[np.newaxis,:],nSim,axis=0)
        beta = np.repeat(res.x[np.newaxis,:],nSim,axis=0)
        factor = np.random.gamma(alpha,beta)
        pd = np.matmul(factor,w) * P
        pd[pd>1] = 1
        d = np.random.binomial(1,pd)
        L_obligor = d * E
        L_port = L_obligor.sum(axis=1)
        print('nSim = ' + "{:,.0f}".format(nSim) + ', seed = ' + str(seed))
        # printLoss
        data = L_port
        VaR = np.percentile(data,99)
        n_tail = len(data[data>=VaR])
        ES = np.sum(data[data>=VaR]/n_tail)
        print('99% VaR = ' + "{:,.0f}".format(VaR))
        print('ES = ' + "{:,.0f}".format(ES))
        # printAttrib
        data_port, data_obligor = L_port, L_obligor
        n_tail = len(data_port[data_port>=VaR])
        mask = data_port >= VaR
        # calculate Gaussian Kernel
        h = len(data_port)**-0.2 * np.std(data_port)
        d = data_obligor.shape[1]
        kernel = 1 / np.sqrt(2*np.pi)*np.exp(-1/2*((data_port-VaR)/h)**2)
        VaR_attrib = data_obligor * np.repeat(kernel[:,np.newaxis], d, axis = 1) / np.sum(kernel)
        VaR_attrib = np.sum(VaR_attrib,axis=0)
        ES_attrib = data_obligor * np.repeat(mask[:,np.newaxis],d,axis=1) / n_tail
        ES_attrib = np.sum(ES_attrib,axis=0)
        #attrib = np.zeros((d,2))
        attrib_VaR[:,i] = VaR_attrib
        attrib_ES[:,i] = ES_attrib
        pan.options.display.float_format = '{:,.0f}'.format
        df1 = pan.DataFrame(attrib_VaR, index = np.arange(d)+1, columns = np.arange(30))
        df2 = pan.DataFrame(attrib_ES, index = np.arange(d)+1, columns = np.arange(30))
        df1.loc['Total']= df1.sum()
        df2.loc['Total']= df2.sum()
        df1['stdev']=df1.std(axis=1)
        df2['stdev']=df2.std(axis=1)
        print(df1)
        print(df2)

    timestamp(sys._getframe().f_code.co_name,'end')
    print()
    return

class style:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Main
IMF_Figure1(1000,True,True,True)
IMF_Figure2(1000,True,True,True)
IMF_Figure3(1000)
IMF_Figure4(100,100,True,True,True)
IMF_Figure4b(1000,True,True,True)
IMF_Figure5_Bernoulli(1000,True,True,True)
IMF_Figure7(1000,np.linspace(0.1,0.5,5),True,True,True)
IMF_Figure4b_seed(1000)
