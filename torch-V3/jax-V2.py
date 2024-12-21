import jax
import jax.numpy as jnp
import numpy as np
import time
from scipy.sparse.linalg import eigsh

def get_W(beta, J=1, h=0):
    sq_cosh = jnp.sqrt(jnp.cosh(beta * J))
    sq_sinh = jnp.sqrt(jnp.sinh(beta * J))
    W = jnp.array([
        [sq_cosh, sq_sinh],
        [sq_cosh, -sq_sinh]
    ])
    return W

def get_Wh(beta, J, h):
    cosh = jnp.cosh(beta * J)
    sinh = jnp.sinh(beta * J)
    Wh = jnp.array([
        [jnp.sqrt(cosh) * jnp.exp(beta * h / 2), jnp.sqrt(sinh) * jnp.exp(beta * h / 2)],
        [jnp.sqrt(cosh) * jnp.exp(-beta * h / 2), -jnp.sqrt(sinh) * jnp.exp(-beta * h / 2)]
    ])
    return Wh

def get_T_bare(beta, J=1, h=0, get_W=get_W):
    W = get_W(beta, J=J, h=h)
    T_bare = jnp.einsum('ai,aj,ak,al->ijkl', W, W, W, W)
    return T_bare

def merge_y(Tup, Tdn, combine=False, trace=False):
    if trace:
        Tmerge = jnp.einsum('abcd,defa->becf', Tup, Tdn)
        if combine:
            shape = Tmerge.shape
            Tmerge = Tmerge.reshape(shape[0] * shape[1], shape[2] * shape[3])
    else:
        Tmerge = jnp.einsum('abcd,defg->abecfg', Tup, Tdn)
        if combine:
            shape = Tmerge.shape
            Tmerge = Tmerge.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5])
    return Tmerge

def merge_y_truncate(Tup, Tdn, dcut):
    if ((Tup.shape[1] * Tdn.shape[1]) < dcut):
        return merge_y(Tup, Tdn, True)
    Tmerge_pure = jnp.einsum('aAbc,cBef,aCbd,dDef->ABCD', Tup, Tdn, Tup, Tdn)
    Tmerge_pure_shape = Tmerge_pure.shape
    Tmerge_pure = Tmerge_pure.reshape(
        Tmerge_pure.shape[0] * Tmerge_pure.shape[1], Tmerge_pure.shape[2] * Tmerge_pure.shape[3]
    )
    print(Tup.shape[1])
    print(Tmerge_pure.shape)
    U, S, Vt = jnp.linalg.svd(Tmerge_pure, full_matrices=False)
    U = U[:, :dcut].reshape(Tmerge_pure_shape[0], Tmerge_pure_shape[1], dcut)
    Tmerge = jnp.einsum('Aace,ebdD,cdC,abB->ABCD', Tup, Tdn, U, U)
    return Tmerge

def merge_x(TL, TR, combine=False, trace=False):
    if trace:
        Tmerge = jnp.einsum('aAca,ecBe->AB', TL, TR)
    else:
        Tmerge = jnp.einsum('BCaF,AaDE->ABCDEF', TL, TR)
        if combine:
            shape = Tmerge.shape
            Tmerge = Tmerge.reshape(shape[0] * shape[1], shape[2], shape[3], shape[4] * shape[5])
    return Tmerge

def merge_x_truncate(TL, TR, dcut):
    if ((TL.shape[0] * TR.shape[0]) < dcut):
        return merge_x(TL, TR, True)
    Tmerge_pure = jnp.einsum('abcd,ecfg,hbid,jifg->aehj', TL, TR, TL, TR)
    Tmerge_pure_shape = Tmerge_pure.shape
    Tmerge_pure = Tmerge_pure.reshape(
        Tmerge_pure.shape[0] * Tmerge_pure.shape[1], Tmerge_pure.shape[2] * Tmerge_pure.shape[3]
    )
    U, S, Vt = jnp.linalg.svd(Tmerge_pure, full_matrices=False)
    U = U[:, :dcut].reshape(Tmerge_pure_shape[0], Tmerge_pure_shape[1], dcut)
    Tmerge = jnp.einsum('abcd,ecfg,aeh,dgi->hbfi', TL, TR, U, U)
    return Tmerge

MaxL = 11
Temp = np.linspace(2.265, 2.285, 11)
W = np.ones((MaxL+1, len(Temp), 2))
E = np.ones((MaxL+1, len(Temp), 2))
Tc = 2 / np.log(1 + np.sqrt(2))

dcut = 20
st = time.time()

for i, temp in enumerate(Temp):
    print(i)
    T_bare = get_T_bare(1 / temp)
    TL = T_bare
    for j in range(2, MaxL+1):
        TLx = merge_x_truncate(TL, TL, dcut)
        TL_Trace = merge_y(TLx, TLx, True, True)
        TL_Trace = np.array(TL_Trace)
        eigvals, eigvecs = eigsh(TL_Trace, k=2, which='LM')
        W[j,i,:] = eigvals
        TL = merge_y_truncate(TLx, TLx, dcut)
        TL = TL / jnp.mean(jnp.abs(TL))
    E = -np.log(W)
    Corr_len = 1/(E[:,:,-2]-E[:,:,-1])
print(time.time() - st)
