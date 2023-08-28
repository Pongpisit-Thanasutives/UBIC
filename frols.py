"""
Originally from https://github.com/lkilcommons/frols
"""

import numpy as np
import scipy.linalg as linalg

def inner_product_ratio(x,y):
    return np.dot(x.T,y)/np.dot(y.T,y)

def proj(u,v):
    return np.dot(u.T,v)/np.dot(u.T,u)*u

def calc_g(Q,Y):
    n_params = Q.shape[1]
    g = np.full((n_params,1),np.nan)
    for i_param in range(n_params):
        g[i_param]=inner_product_ratio(Y,Q[:,i_param])
    return g

def calc_err(g,Q,Y):
    sigma = np.dot(Y.T,Y)
    n_params = Q.shape[1]
    err = np.full_like(g,np.nan)
    for i_param in range(n_params):
        err[i_param]=g[i_param]**2*np.dot(Q[:,i_param].T,Q[:,i_param])/sigma
    return err

def calc_esr(err):
    return 1-np.sum(err.flatten())

def orthogonalize(P):
    (Q,R) = np.linalg.qr(P,mode='reduced')
    return Q

def remove_B_columns_from_A(A,B):
    n_params = A.shape[1]
    n_cols = B.shape[1]
    A_new = A.copy()
    for i_param in range(n_params):
        for i_col in range(n_cols):
            A_new[:,i_param] -= proj(B[:,i_col],A[:,i_param])
    return A_new

def solve_triangular(A,B):
    """
    Solve Ax = B for x when A is upper triangular, and has unit diagonal
    """
    x = linalg.solve_triangular(A,B,unit_diagonal=True)
    return x

def frols(YY,PP,pcols=None,max_nonzeros=None,term_thresh=None,max_steps=1000,copy_data=True,verbose=False):
    if copy_data: Y,P = YY.copy(),PP.copy()
    l_list = [] # List of indicies (columns of P) which have been selected for model
    g_list = [] # List of coefficients to retained from each step (i.e. for step s g^s[:,l_s])
    Q_list = [] # List of columns of step-specific orthogonal matrix Q columns to retain (i.e. Q^s[:,l_s])
    err_list = [] #List of step-specific error reducing ratios to retain (ERR^s[:,l_s])
    if pcols is None: pcols = {i:'x%d' % (i+1) for i in range(P.shape[1])}
    if max_nonzeros is None: max_nonzeros = len(pcols) 
    if term_thresh is None: term_thresh = 0.0
    #Mask
    selected = np.zeros((P.shape[1],),dtype=bool)
    esr = 1.
    i_step = 0
    selected_inds_list = []
    while esr > term_thresh and i_step<P.shape[1]:
        selected[l_list] = True
        unselected = np.logical_not(selected)
        unselected_inds = np.flatnonzero(unselected)
        selected_inds = np.where(selected)[0].tolist()
        if verbose: print("selected %s" % selected_inds)
        if len(selected_inds) > max_nonzeros: break
        elif len(selected_inds) > 0: selected_inds_list.append(selected_inds)
        if i_step==0:
            Q = P.copy()
            g = calc_g(Q,Y)
            err = calc_err(g,Q,Y)
            l = np.argmax(err)
            l_list.append(l)
            g_list.append(g[l])
            Q_list.append(Q[:,l])
            err_list.append(err[l])
        else:
            if verbose: print("Shape of P = %s" % (str(P[:,unselected].shape)))
            Q = orthogonalize(P[:,selected])
            thisQ = remove_B_columns_from_A(P[:,unselected],Q)
            g = calc_g(thisQ,Y)
            err = calc_err(g,thisQ,Y)
            thisl = np.argmax(err)
            l = unselected_inds[thisl]
            l_list.append(l)
            g_list.append(g[thisl])
            Q_list.append(thisQ[:,thisl])
            err_list.append(err[thisl])
        esr = calc_esr(np.array(err_list))
        if verbose: print("Step %d: Selected term at index %d (%s), ERR=%f, ESR=%f" % (i_step,l_list[-1],pcols[l_list[-1]],err_list[-1],esr))
        i_step+=1
    coeffs = np.zeros((len(selected_inds_list),P.shape[1]))
    for i_ind,ind in enumerate(selected_inds_list):
        coeffs[i_ind][ind] = np.linalg.lstsq(P[:,ind],Y.ravel(),rcond=None)[0]
    coeffs = coeffs.T
    #Finally put together the final coefficients
    model_str = []
    Q_f = np.column_stack(Q_list)
    g_f = np.array(g_list).reshape(-1,1)
    err_f = np.array(err_list).reshape(-1,1)
    n_identified = Q_f.shape[1]
    A = np.eye(n_identified)
    Pl = P[:,l_list]
    #Fill in upper triangluar part of A
    for s in range(n_identified):
        for r in range(s):
            if r==s:
                A[r,s]=1.
            else:
                A[r,s] = np.dot(Q_f[:,r].T,Pl[:,s])/np.dot(Q_f[:,r].T,Q_f[:,r])
    coef = solve_triangular(A,g_f)
    pred_Y = np.dot(Pl,coef)
    modelstr = '+'.join(['%.9f*%s' % (coef[k],pcols[l_list[k]]) for k in range(n_identified)])
    if verbose: print("Using model: %s\n RMSE is %f" % (modelstr,np.sqrt(np.mean((Y-pred_Y)**2))) )
    full_coef = np.zeros((P.shape[1],1))
    full_err = np.zeros((P.shape[1],1))
    full_coef[l_list]=coef
    full_err[l_list]=err_f
    
    return coeffs,list(map(tuple,selected_inds_list)),full_coef,full_err,pred_Y
