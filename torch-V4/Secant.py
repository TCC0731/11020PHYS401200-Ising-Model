import numpy as np
import torch
import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process
import time
import pickle
import argparse

def get_W(beta, J=1, h=0):
    """
    Calculate the matrix W based on beta, J, and h.
    
    Parameters:
        beta (float): Inverse temperature.
        J (float): Interaction strength (default is 1).
        h (float): External field strength (default is 0).

    Returns:
        torch.Tensor: Matrix W.
    """
    sq_cosh = np.sqrt(np.cosh(beta * J))
    sq_sinh = np.sqrt(np.sinh(beta * J))
    W = torch.tensor([
        [sq_cosh, sq_sinh],
        [sq_cosh, -sq_sinh]
    ])
    return W


#可能Wh是正確的包含h的作法
def get_Wh(beta, J, h):
    """
    Calculate the matrix Wh incorporating the external field h.
    
    Parameters:
        beta (float): Inverse temperature.
        J (float): Interaction strength.
        h (float): External field strength.

    Returns:
        torch.Tensor: Matrix Wh.
    """
    cosh = np.cosh(beta * J)
    sinh = np.sinh(beta * J)

    Wh = torch.tensor([
        [np.sqrt(cosh) * np.exp(beta * h / 2), np.sqrt(sinh) * np.exp(beta * h / 2)],
        [np.sqrt(cosh) * np.exp(-beta * h / 2), -np.sqrt(sinh) * np.exp(-beta * h / 2)]
    ])
    return Wh


# 上左右下
def get_T_bare(beta, J=1, h=0, get_W=get_W):
    """
    Calculate the bare transfer matrix T_bare.
    
    Parameters:
        beta (float): Inverse temperature.
        J (float): Interaction strength (default is 1).
        h (float): External field strength (default is 0).
        get_W (callable): Function to calculate W or Wh (default is get_W).

    Returns:
        torch.Tensor: Bare transfer matrix T_bare.
    """
    W = get_W(beta, J=J, h=h)
    T_bare = torch.einsum('ai,aj,ak,al->ijkl', W, W, W, W)
    return T_bare

def merge_y(Tup, Tdn, combine=False, trace=False):
    if trace:
        Tmerge = torch.einsum('abcd,defa->becf', Tup, Tdn)
        
        if combine:
            shape = Tmerge.shape
            Tmerge = Tmerge.contiguous().view(shape[0] * shape[1], shape[2] * shape[3])
    else:
        Tmerge = torch.einsum('abcd,defg->abecfg', Tup, Tdn)
        
        if combine:
            shape = Tmerge.shape
            Tmerge = Tmerge.contiguous().view(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5])
            
    return Tmerge


def merge_y_truncate(Tup, Tdn, dcut):
    if ((Tup.shape[1] * Tdn.shape[1]) < dcut):
        return merge_y(Tup, Tdn, True)
    Tmerge_pure = torch.einsum('aAbc,cBef,aCbd,dDef->ABCD', Tup, Tdn, Tup, Tdn)
    Tmerge_pure_shape = Tmerge_pure.shape
    Tmerge_pure = Tmerge_pure.contiguous().view(
        Tmerge_pure_shape[0] * Tmerge_pure_shape[1],Tmerge_pure_shape[2] * Tmerge_pure_shape[3]
    )
    U, S, V = torch.linalg.svd(Tmerge_pure)
    U = U[:,:dcut]
    U = U.view(Tmerge_pure_shape[0], Tmerge_pure_shape[1], dcut)
    Tmerge = torch.einsum('Aace,ebdD,cdC,abB->ABCD', Tup, Tdn, U, U)
    return Tmerge

def merge_x(TL, TR, combine=False, trace=False):
    if trace:
        Tmerge = torch.einsum('aAca,ecBe->AB', TL, TR)
    else:
        Tmerge = torch.einsum('BCaF,AaDE->ABCDEF', TL, TR)
        
        if combine:
            shape = Tmerge.shape
            Tmerge = Tmerge.contiguous().view(shape[0] * shape[1], shape[2], shape[3], shape[4] * shape[5])
            
    return Tmerge

def merge_x_truncate(TL, TR, dcut):
    if ((TL.shape[0] * TR.shape[0]) < dcut):
        return merge_x(TL, TR, True)
    Tmerge_pure = torch.einsum('abcd,ecfg,hbid,jifg->aehj', TL, TR, TL, TR)
    Tmerge_pure_shape = Tmerge_pure.shape
    Tmerge_pure = Tmerge_pure.contiguous().view(
        Tmerge_pure_shape[0] * Tmerge_pure_shape[1],Tmerge_pure_shape[2] * Tmerge_pure_shape[3]
    )
    U, S, V = torch.linalg.svd(Tmerge_pure)
    U = U[:,:dcut]
    U = U.view(Tmerge_pure_shape[0], Tmerge_pure_shape[1], dcut)
    Tmerge = torch.einsum('abcd,ecfg,aeh,dgi->hbfi', TL, TR, U, U)
    return Tmerge

def f(temp, MaxL, dcut = 20):
    beta = 1 / temp
    T_bare = get_T_bare(beta).cuda(0)
    TL = T_bare
    for j in range(2, MaxL + 1):
        TLx = merge_x_truncate(TL, TL, dcut)
        TL = merge_y_truncate(TLx, TLx, dcut)
        TL = TL / torch.mean(torch.abs(TL))
    TL_Trace = merge_y(TLx, TLx, True, True).cpu()
    w1 = torch.linalg.eigvalsh(TL_Trace)
    E1 = -np.log(w1[-2:])
    Corr_len1 = 1/(E1[-2]-E1[-1])
    TLx = merge_x_truncate(TL, TL, dcut)
    TL = merge_y_truncate(TLx, TLx, dcut)
    TL = TL / torch.mean(torch.abs(TL))
    TL_Trace = merge_y(TLx, TLx, True, True).cpu()
    w2 = torch.linalg.eigvalsh(TL_Trace)
    E2 = -np.log(w2.numpy()[-2:])
    Corr_len2 = 1/(E2[-2]-E2[-1])
    del TL, TLx, TL_Trace
    return Corr_len1-Corr_len2

def Secant_Method(func,x0,x1,Maxiter,MaxL,dcut):
    x = [x0,x1]
    func_out = [func(x0,MaxL,dcut)]
    for i in range(Maxiter):
        if (x[-1] - x[-2]) == 0:
            break
        func_out.append(func(x[-1],MaxL,dcut))
        if (func_out[-1] - func_out[-2]) == 0:
            break
        if np.abs(func_out[-1]) <= 1e-13:
            break
        new_x = x[-1] - func_out[-1] * (x[-1] - x[-2]) / (func_out[-1] - func_out[-2])
        x.append(new_x)
        #print(i,x[-1],func_out[-1])
    return x[-1]

MaxL = 11
dcuts = [32,36]
Tc = 2/np.log(1+np.sqrt(2))
T_stars = []

for dcut in dcuts:
    T_star = [2.36]
    print("dcut =",dcut)
    for i in range(2,MaxL):
        #print(i , 2**(i-1))
        T_star.append(Secant_Method(f,2.265,T_star[-1],10,i,dcut).item())
        print(i , 2**(i-1), T_star[-1], T_star[-1] - Tc)
    T_stars.append(T_star)
T_stars = np.array(T_stars)

# Save results and parameters using pickle
output_data = {
    "MaxL": MaxL,
    "Tc": Tc,
    "dcuts": dcuts,
    "T_stars": T_stars
}

with open(f"Secant_results_2.pkl", "wb") as f:
    pickle.dump(output_data, f)

print("Results saved to Secant_results.pkl")