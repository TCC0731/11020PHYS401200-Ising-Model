import numpy as np
import torch
import matplotlib.pyplot as plt
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

MaxL = 11
Temp = np.linspace(2.265,2.285,100)
W = np.ones((MaxL+1,len(Temp),2))
E = np.ones((MaxL+1,len(Temp),2))
Tc = 2/np.log(1+np.sqrt(2))

dcut = 24

st = time.time()

with torch.no_grad():
    for i,temp in enumerate(Temp):
        print(i)
        T_bare = get_T_bare(1/temp)
        TL = T_bare
        for j in range(2,MaxL+1):
            TLx = merge_x_truncate(TL,TL,dcut)
            TL_Trace = merge_y(TLx,TLx,True,True)
            eigvals, eigvecs = torch.linalg.eigh(TL_Trace)
            del TL_Trace
            #W[j,i,:] = eigvals[-2:]
            TL = merge_y_truncate(TLx,TLx,dcut)
            TL = TL/torch.mean(torch.abs(TL))
        del TL
    #E = -np.log(W)
    #Corr_len = 1/(E[:,:,-2]-E[:,:,-1])
    
print(time.time() - st)