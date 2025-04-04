import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    print("gurobipy is not installed.")

from best_subset import backward_refinement
from sklearn.preprocessing import normalize

from tqdm import trange
nplm = np.linalg.lstsq

# Create and deploy the optimization model
# NOTE: This function assumes the design matrix features does not contain a column for the intercept
# With an intercept, L0-norm constrainted MIQP
def miqp(features, response, non_zero, verbose=False):
    """
    Deploy and optimize the MIQP formulation of L0-Regression.
    """
    assert isinstance(non_zero, (int, np.integer))
    regressor = gp.Model()
    samples, dim = features.shape
    assert samples == response.shape[0]
    assert non_zero <= dim

    # Append a column of ones to the feature matrix to account for the y-intercept
    X = np.concatenate([features, np.ones((samples, 1))], axis=1)  
    
    # Decision variables
    norm_0 = regressor.addVar(lb=non_zero, ub=non_zero, name="norm")
    beta = regressor.addMVar((dim + 1,), lb=-GRB.INFINITY, name="beta") # Weights
    intercept = beta[dim] # Last decision variable captures the y-intercept

    regressor.setObjective(beta.T @ X.T @ X @ beta
                           - 2*response.T @ X @ beta
                           + np.dot(response, response), GRB.MINIMIZE)
    
    # Budget constraint based on the L0-norm
    regressor.addGenConstrNorm(norm_0, beta[:-1], which=0, name="budget")
    
    if not verbose:
        regressor.params.OutputFlag = 0
    else:
        regressor.params.OutputFlag = 1
    regressor.params.timelimit = 60
    regressor.params.mipgap = 0.001
    regressor.optimize()

    coeff = np.array([beta[i].X for i in range(dim)])
    return intercept.X, coeff

# Without an intercept, L0-norm constrainted MIQP
def miqp2(features, response, non_zero, alpha=0, verbose=False):
    """
    Deploy and optimize the MIQP formulation of L0-Regression.
    """
    assert isinstance(non_zero, (int, np.integer))
    regressor = gp.Model()
    samples, dim = features.shape
    assert samples == response.shape[0]
    assert non_zero <= dim

    # Decision variables
    X = features
    norm_0 = regressor.addVar(lb=non_zero, ub=non_zero, name="norm")
    beta = regressor.addMVar((dim,), lb=-GRB.INFINITY, name="beta") # Weights

    if alpha > 0:
        # Drop the constant term and add regularization on the weights
        regressor.setObjective(beta.T @ X.T @ X @ beta
                               - 2*response.T @ X @ beta
                               + alpha*(beta.T@beta), 
                               GRB.MINIMIZE)
    else:
        regressor.setObjective(beta.T @ X.T @ X @ beta
                               - 2*response.T @ X @ beta
                               + np.dot(response, response), # Constant term
                               GRB.MINIMIZE)
    
    # Budget constraint based on the L0-norm
    regressor.addGenConstrNorm(norm_0, beta, which=0, name="budget")
    
    if not verbose:
        regressor.params.OutputFlag = 0
    else:
        regressor.params.OutputFlag = 1
    regressor.params.timelimit = 60
    regressor.params.mipgap = 0.001
    regressor.optimize()

    coeff = np.array([beta[i].X for i in range(dim)])
    return coeff

# Without an intercept + MIOSR implementation
def miqp3(features, response, non_zero, alpha=0, verbose=False):
    """
    Deploy and optimize SOS-1 formulated MIO-SINDy
    """
    assert isinstance(non_zero, (int, np.integer))
    regressor = gp.Model()
    samples, dim = features.shape
    assert samples == response.shape[0]
    assert non_zero <= dim

    X = features
    beta = regressor.addMVar((dim,), lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="beta") # Weights as a column vector
    iszero = regressor.addMVar(dim, vtype=gp.GRB.BINARY, name="iszero")

    for i in range(dim):
        regressor.addSOS(gp.GRB.SOS_TYPE1, [beta[i], iszero[i]])
    regressor.addConstr(iszero.sum() >= dim-non_zero, name="sparsity")

    regressor.setObjective(beta.T @ X.T @ X @ beta
                           - 2*response.T @ X @ beta
                           + alpha*(beta.T@beta), 
                           GRB.MINIMIZE)
    
    if not verbose:
        regressor.params.OutputFlag = 0
    else:
        regressor.params.OutputFlag = 1
    regressor.params.timelimit = 100
    # regressor.params.mipgap = 0.001

    regressor.optimize()

    coeff = np.array([beta[i].X for i in range(dim)])
    return coeff

def solvel0(X_pre, y_pre, is_normal=False, intercept=False, miosr=False, refine=False, ic_type='bic', max_complexity=None, verbose=False):
    out = set()
    if max_complexity is None:
        max_complexity = X_pre.shape[1]
    if is_normal:
        X_pre = normalize(X_pre, axis=0)
    for i in trange(1, max_complexity+1):
        if intercept:
            beta = miqp(X_pre, y_pre.flatten(), i)[-1]
        else:
            if not miosr:
                beta = miqp2(X_pre, y_pre.flatten(), i)
            else:
                beta = miqp3(X_pre, y_pre.flatten(), i)
        effective_indices = tuple(np.where(np.abs(beta)>0)[0])
        # Gurantee the solution with every candidates
        # if i == X_pre.shape[1]:
        #     effective_indices = tuple(range(X_pre.shape[1]))
        if len(effective_indices) > 0:
            out.add(effective_indices)

    # Added backward_refinement capability
    if refine:
        print("Call backward_refinement...")
        st = refine_solvel0(out, (X_pre, y_pre), ic_type, verbose)
        # out = sorted([st.track[e][0] for e in st.track if len(st.track[e][0]) <= max_complexity], key=len)
        out = sorted([st.track[e][0] for e in st.track], key=len)
    best_subsets = sorted(out, key=len)
    return best_subsets

def refine_solvel0(out, dataset, ic_type='bic', verbose=False):
    st = backward_refinement(out, dataset, mode='rfe', ic_type=ic_type, verbose=verbose)
    st += backward_refinement(out, dataset, mode='SelectKBest', ic_type=ic_type, verbose=verbose)
    return st
