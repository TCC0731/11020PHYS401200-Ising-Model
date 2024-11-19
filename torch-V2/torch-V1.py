import numpy as np
import torch
import time

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

MaxL = 4
Temp = np.linspace(2.26,4,10000)
W = np.ones((MaxL+1,len(Temp),2))
E = np.ones((MaxL+1,len(Temp),2))
Tc = 2/np.log(1+np.sqrt(2))

st = time.time()

for i,temp in enumerate(Temp):
    T_bare = get_T_bare(1/temp)
    TL = T_bare
    for j in range(2,MaxL+1):
        TL_Trace = merge_y(TL,TL,True,True)
        eigvals, eigvecs = torch.linalg.eigh(TL_Trace)
        W[j,i,:] = eigvals[-2:]
        del TL_Trace
        if j != MaxL:
            TL = merge_y(TL,TL,True)
E = -np.log(W)
Corr_len = 1/(E[:,:,-2]-E[:,:,-1])

print(time.time() - st)