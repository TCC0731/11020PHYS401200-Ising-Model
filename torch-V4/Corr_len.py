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
    U, S, V = torch.svd(Tmerge_pure)
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
    U, S, V = torch.svd(Tmerge_pure)
    U = U[:,:dcut]
    U = U.view(Tmerge_pure_shape[0], Tmerge_pure_shape[1], dcut)
    Tmerge = torch.einsum('abcd,ecfg,aeh,dgi->hbfi', TL, TR, U, U)
    return Tmerge

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=int, default=20)
args = parser.parse_args()
print(args)

MaxL = 15
Tc = 2/np.log(1+np.sqrt(2))
Temp = np.linspace(2.265, 2.285, 100)
dbeta = 1e-2
dcut = args.d
num_processes = 8
print(MaxL, dbeta, dcut, num_processes)

def compute_Corr_len_for_temp_cuda(i, temp):

    process_name = int(current_process().name[-1]) - 1

    print(f'Start {i}: {temp:.3f} {process_name} {current_process().name}\n')
    st = time.time()
    beta = 1 / temp
    W = np.zeros((MaxL + 1, 2))

    T_bare = get_T_bare(beta).cuda(process_name)
    TL = T_bare
    for j in range(2, MaxL + 1):
        TLx = merge_x_truncate(TL, TL, dcut)
        TL_Trace = merge_y(TLx, TLx, True, True)
        eigvals = torch.linalg.eigvalsh(TL_Trace).cpu()
        W[j, :] = eigvals[-2:]
        TL = merge_y_truncate(TLx, TLx, dcut)
        TL = TL / torch.mean(torch.abs(TL))

    E = -np.log(W)
    Corr_len = 1 / (E[:, -2] - E[:, -1])
    
    print(f'End {i}: {temp:.3f} Use time: {(time.time() - st) :.2f}')
    return i, Corr_len

Corr_len_matrix = np.zeros((MaxL + 1, len(Temp)))

with Pool(processes=num_processes) as pool:
    results = pool.starmap(compute_Corr_len_for_temp_cuda, [(i, temp) for i, temp in enumerate(Temp)])

# 將結果合併回主矩陣
for i, Corr_len in results:
    Corr_len_matrix[:, i] = Corr_len

# Save results and parameters using pickle
output_data = {
    "MaxL": MaxL,
    "Temp": Temp.tolist(),
    "Tc": Tc,
    "Corr_len": Corr_len_matrix.tolist(),
    "dcut": dcut
}

with open(f"Corr_len_results_{dcut}_{Temp[0]}_{Temp[-1]}.pkl", "wb") as f:
    pickle.dump(output_data, f)

print("Results saved to Corr_len_results.pkl")