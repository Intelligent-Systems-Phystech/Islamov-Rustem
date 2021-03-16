import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import pylab

def random_k(x, k):
    dim = x.shape[0]
    answer = np.zeros(dim)
    positions = list(range(dim))
    np.random.shuffle(positions)
    for i in positions[:k]:
        answer[i] = x[i]*dim/k
    return answer

def random_sparsification(x, p):
    d = x.shape[0]
    binom = np.zeros((d+1, d+1))
    for i in range(d+1):
        binom[i,0] = 1
        binom[i,i] = 1
    for i in range(2,d+1):
        for j in range(1, i):
            binom[i,j] = binom[i-1,j]+binom[i-1,j-1]
    comp_g = np.zeros(d)
    k = 0
    for i in range(d):
        if bernoulli.rvs(p):
            comp_g[i] = x[i]/p
            k += 1
    bits = 64*k + np.log2(binom[d, k])
    bits = int(np.ceil(bits))
            
    return comp_g, bits

def positive_part(x):
    for i, c in enumerate(x):
        if c < 0:
            x[i] = 0
    return x

def function_plot_builder(labels, X, Y, title, x_label, y_label, filename, x_lim=None, y_lim=None):
    plt.figure(figsize=[10,5])
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=15)
    for i in range(len(labels)):
        plt.semilogy(X[i], Y[i], label=labels[i])
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid()
    plt.minorticks_on()
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.show()
    
def bits_plot_builder(labels, X, Y, title, x_label, y_label, filename, x_lim=None, y_lim=None):
    figure = plt.figure(figsize=[10, 5])
    axes = figure.add_subplot (1, 1, 1)
    axes.set_yscale('log', subsy = [2,4,6,8])
    axes.set_xscale('log', basex = 2, subsx = [2,4,6,8])
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=15)
    for i in range(len(labels)):
        pylab.plot(X[i], Y[i], label=labels[i])
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid()
    plt.minorticks_on()
    plt.legend(loc='best')
    plt.savefig(filename, dpi=1200)
    plt.show()
    
    
def bits_plot_builder1(labels, X, Y, title, x_label, y_label, filename, x_lim=None, y_lim=None):
    figure = plt.figure(figsize=[10, 5])
    axes = figure.add_subplot (1, 1, 1)
    axes.set_yscale('log', subsy = [2,4,6,8])
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.title(title, fontsize=15)
    for i in range(len(labels)):
        pylab.plot(X[i], Y[i], label=labels[i])
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid()
    plt.minorticks_on()
    plt.legend(loc='best')
    plt.savefig(filename, dpi=1200)
    plt.show()
    

def random_sparse(x, arg):
    r = int(arg.r)
    dim = np.shape(x)[0]
    xi_r = np.zeros(dim, dtype=int)
    loc_ones = np.random.choice(dim, r, replace=False)
    xi_r[loc_ones] = 1
    x_compress = dim / r * x * xi_r
    return x_compress



def random_dithering(x, arg):
    s = int(arg.s)
    dim = np.shape(x)[0]
    xx = np.random.uniform(0.0, 1.0, dim)
    xnorm = np.linalg.norm(x)
    if xnorm > 0:
        xsign = np.sign(x)
        x_int = np.floor(s * np.abs(x) / xnorm + xx)
        x_cpmpress = xnorm / s * xsign * x_int
    else:
        x_cpmpress = np.zeros(dim)
    return x_cpmpress


def natural_compression(x, arg):
    dim = np.shape(x)[0]
    xabs = np.abs(x)
    xsign = np.sign(x)
    x_compress = x
    for i in range(dim):
        if x[i] != 0.0:
            xlog = np.log2(xabs[i])
            xdown = np.exp2(np.floor(xlog))
            xup = np.exp2(np.ceil(xlog))
            p = (xup - xabs[i]) / xdown
            if np.random.uniform(0.0, 1.0) <= p:
                x_compress[i] = xsign[i] * xdown
            else:
                x_compress[i] = xsign[i] * xup
    return x_compress


def no_compression(x, arg):
    return x


compression_dic = {
    'rand_sparse': random_sparse,
    'rand_dithering': random_dithering,
    'natural_comp': natural_compression,
    'no_comp': no_compression
}


def loss_logistic(X, y, w, arg):
    assert len(y) == X.shape[0]
    assert len(w) == X.shape[1]
    z = y * np.dot(X, w)
    tmp = np.minimum(z, 0)
    loss = np.log((np.exp(tmp) + np.exp(tmp - z)) / np.exp(tmp))
    loss_sum = np.sum(loss) / len(y)
    reg = (np.linalg.norm(w) ** 2) * arg.lamda / 2

    return loss_sum + reg

def sigmoid(z):
    tmp = np.minimum(z, 0)
    res = np.exp(tmp) / (np.exp(tmp) + np.exp(tmp - z))
    return res

def regularizer2(w, lamda):
    res = np.linalg.norm(w) ** 2
    return res * lamda / 2

def prox2(x, eta):
    newx = x / (1 + eta)
    return newx

def grad(X, y, w, arg):
    m = len(y)
    z = y * np.dot(X, w)
    tmp0 = np.minimum(z, 0)
    tmp1 = np.exp(tmp0 - z) / ((np.exp(tmp0) + np.exp(tmp0 - z)))
    tmp2 = - tmp1 * y
    res = np.dot(X.T, tmp2) / m + arg.lamda * w
    return res

def compute_bit(dim, arg):
    if arg.comp_method == 'rand_sparse':
        bit = 32 * arg.r
    elif arg.comp_method == 'rand_dithering':
        bit = 2.8 * dim + 32
    elif arg.comp_method == 'natural_comp':
        bit = 9 * dim
    else:
        bit = 32 * dim
    return bit


def compute_omega(dim, arg):
    if arg.comp_method == 'rand_sparse':
        omega = dim / arg.r - 1
    elif arg.comp_method == 'rand_dithering':
        omega = 1  # level s=sqrt(dim)
    elif arg.comp_method == 'natural_comp':
        omega = 1 / 8
    else:
        omega = 0
    return omega

    