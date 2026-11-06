"""Reconcile pairwise log-ratio estimates into one latent log-price per product."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

def reconcile(n_products, edges, log_ratios, weights=None, anchors=None):
    """Solve weighted least squares for x_i - x_j ~= log(price_i / price_j).

    anchors is an optional iterable of (product_index, log_price, weight).
    The mean log-price is fixed to zero to identify the otherwise free scale.
    """
    edges=np.asarray(edges,dtype=int); y=np.asarray(log_ratios,dtype=float)
    if len(edges)!=len(y) or edges.ndim!=2 or edges.shape[1]!=2: raise ValueError("Invalid edge/target shapes")
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    rows=np.repeat(np.arange(len(y)),2); cols=edges.ravel(); vals=np.tile([1.0,-1.0],len(y))
    A=sparse.coo_matrix((vals,(rows,cols)),shape=(len(y),n_products)).tocsr()
    sw=np.sqrt(np.clip(w,1e-8,None)); A=sparse.diags(sw)@A; b=sw*y
    if anchors:
        ar=[]; ac=[]; av=[]; by=[]
        for k,(idx,value,weight) in enumerate(anchors): ar.append(k);ac.append(int(idx));av.append(float(np.sqrt(weight)));by.append(float(value)*float(np.sqrt(weight)))
        AA=sparse.coo_matrix((av,(ar,ac)),shape=(len(anchors),n_products)).tocsr()
        A=sparse.vstack([A,AA]); b=np.r_[b,by]
    # Mean-zero gauge constraint; high weight approximates exact centering.
    A=sparse.vstack([A,sparse.csr_matrix(np.ones((1,n_products))*1000)])
    b=np.r_[b,0.0]
    return lsqr(A,b,atol=1e-10,btol=1e-10)[0]
