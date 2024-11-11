import cytnx
import numpy as np
import matplotlib.pyplot as plt
import cProfile
from scipy.optimize import curve_fit
import time

def ut_print(ut, print_numpy=True):
    ut.print_diagram()
    if print_numpy:
        print(ut.get_block().numpy())

def get_M(T):
    W = np.array([[np.exp(1/T), np.exp(-1/T)],
                  [np.exp(-1/T), np.exp(1/T)]])
    W = cytnx.from_numpy(W)
    S, U, Vd = cytnx.linalg.Svd(W)
    M = U @ cytnx.linalg.Diag(S.Pow(0.5))
    M = cytnx.UniTensor(M, rowrank=1)
    M.set_name('M')
    Md = cytnx.linalg.Diag(S.Pow(0.5)) @ Vd
    Md = cytnx.UniTensor(Md, rowrank=1)
    Md.set_name('Md')
    return M, Md

def get_delta(h):
    delta = cytnx.zeros([2, 2, 2, 2])
    delta[0, 0, 0, 0] = 1
    delta[1, 1, 1, 1] = 1
    delta = cytnx.UniTensor(delta, rowrank=2)
    delta.set_name('delta')
    return delta

def get_T_baret(T, h=0):
    ut_M, ut_Md = get_M(T)
    ut_delta = get_delta(h)
    Ising_net = cytnx.Network('./Networks/Ising_square.net')
    Ising_net.PutUniTensors(['delta', 'M0.d', 'M1.d', 'M2', 'M3'],
                            [ut_delta, ut_Md, ut_Md, ut_M, ut_M])
    T_bare = Ising_net.Launch()
    T_bare.set_name('T_bare')
    return T_bare

def merge_y(Tup, Tdn, combine=False, trace=False):
    if trace:
        TupTdn_net = cytnx.Network('./Networks/merge_y_trace.net')
    else:
        TupTdn_net = cytnx.Network('./Networks/merge_y.net')
    TupTdn_net.PutUniTensors(['Tup', 'Tdn'], [Tup, Tdn])
    TupTdn = TupTdn_net.Launch()

    if combine:
        # TupTdn.combineBonds([1, 2])
        # TupTdn.combineBonds([3, 4])
        # TupTdn.relabels_(['1','2','3','4'])
        TupTdn.print_diagram()
        TupTdn.combineBonds(['1', '2'])
        TupTdn.print_diagram()
        TupTdn.combineBonds(['3', '4'])
    return TupTdn

def merge_y_truncate(Tup, Tdn, dcut):
    if ((Tup.shape()[1] * Tdn.shape()[1]) < dcut):
        return merge_y(Tup, Tdn, True)
    TupTdn_pure_net = cytnx.Network('./Networks/merge_y_pure.net')
    TupTdn_pure_net.PutUniTensors(['Tup', 'Tdn', 'Tupd', 'Tdnd'],
                                  [Tup, Tdn, Tup, Tdn])
    TupTdn_pure = TupTdn_pure_net.Launch()
    _, U, __ = cytnx.linalg.Svd_truncate(TupTdn_pure, dcut)
    TupTdn_net = cytnx.Network('./Networks/merge_y_truncate.net')
    TupTdn_net.PutUniTensors(['Tup', 'Tdn', 'UL', 'UR'],
                             [Tup, Tdn, U, U])
    TupTdn = TupTdn_net.Launch()
    return TupTdn

def merge_x(TL, TR, combine=False, trace=False):
    if trace:
        n = TL.shape()[0]
        I = np.eye(n)
        I = cytnx.from_numpy(I)
        I = cytnx.UniTensor(I, rowrank=1)
        TLTR_net = cytnx.Network('./Networks/merge_x_trace.net')
        TLTR_net.PutUniTensors(['TL', 'TR', 'IL', 'IR'], [TL, TR, I, I])
        TLTR = TLTR_net.Launch()
    else:
        TLTR_net = cytnx.Network('./Networks/merge_x.net')
        TLTR_net.PutUniTensors(['TL', 'TR'], [TL, TR])
        TLTR = TLTR_net.Launch()
    if combine and not trace:
        # TLTR.combineBonds([0, 1])
        # TLTR.combineBonds([4, 5])
        TLTR.print_diagram()
        shape = TLTR.shape()
        TLTR.reshape_(shape[0]*shape[1], shape[2]*shape[3])
    return TLTR

def merge_x_truncate(TL, TR, dcut):
    if ((TL.shape()[0] * TR.shape()[0]) < dcut):
        return merge_x(TL, TR, True)
    TLTR_pure_net = cytnx.Network('./Networks/merge_x_pure.net')
    TLTR_pure_net.PutUniTensors(['TL', 'TR', 'TLd', 'TRd'],
                                [TL, TR, TL, TR])
    TLTR_pure = TLTR_pure_net.Launch()
    try:
        _, U, __ = cytnx.linalg.Svd_truncate(TLTR_pure, dcut)
    except:
        TLTR_pure.print_diagram()
        _, U, __ = cytnx.linalg.Svd(TLTR_pure, dcut)
    TLTR_net = cytnx.Network('./Networks/merge_x_truncate.net')
    TLTR_net.PutUniTensors(['TL', 'TR', 'Uup', 'Udn'], [TL, TR, U, U])
    TLTR = TLTR_net.Launch()
    return TLTR

def trace(T):
    n = T.shape()[0]
    I = np.eye(n)
    I = cytnx.from_numpy(I)
    I = cytnx.UniTensor(I, rowrank=1)
    trace_net = cytnx.Network('./Networks/trace.net')
    trace_net.PutUniTensors(['T', 'I'], [T, I])
    return trace_net.Launch()

MaxL = 4
Temp1 = np.linspace(2.26,4,100)
W1 = np.ones((MaxL+1,len(Temp1),2))
E1 = np.ones((MaxL+1,len(Temp1),2))
Tc = 2/np.log(1+np.sqrt(2))

for i,temp in enumerate(Temp1):
    #print(temp)
    T_bare = get_T_baret(temp)
    TL = T_bare
    for j in range(2,MaxL+1):
        TL_Trace = merge_y(TL,TL,True,True)
        w,v = cytnx.linalg.Eigh(TL_Trace.get_block())
        W1[j,i,:] = w[-2:].numpy()
        TL = merge_y(TL,TL,True)
E1 = -np.log(W1)
Corr_len1 = 1/(E1[:,:,-2]-E1[:,:,-1])