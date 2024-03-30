import re
import json
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

# For feature selection
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import mean_squared_error
from p_linear_regression import PLinearRegression, MLinearRegression

try:
    from mrmr import mrmr_regression
except ImportError:
    # print("mrmr is not installed in the env you are using. This may cause an error in future if you try to use the (missing) lib.")
    pass

from functools import lru_cache, reduce
from func_timeout import func_timeout, func_set_timeout, FunctionTimedOut
from rapidfuzz import fuzz
from statsmodels.tools.eval_measures import aicc as sm_aicc
from statsmodels.tools.eval_measures import hqic as sm_hqic
    
import pysindy as ps
try:
    from l0bnb import fit_path
    from abess.linear import LinearRegression
except ImportError:
    print("Complete best-subset solvers are not installed.")

from tqdm import trange

def composite_function(*func, left2right=False):
    def compose(f, g, left2right=left2right):
        if left2right: return lambda *args, **kwargs: g(f(*args, **kwargs))
        else: return lambda *args, **kwargs: f(g(*args, **kwargs))
    return reduce(compose, func)

def nonz(wei):
    return tuple(np.nonzero(wei)[0])

def add_columns(X, nums=1, val=1.0):
    return np.pad(X, pad_width=[(0,0), (0,nums)], mode='constant', constant_values=val)

# BIG WARNING: bnb takes at least 5 features to operate unless "TypeError: invalid operation on untyped list"!!!
# lam=1e-3 (slower than 1e-2) is a better default values?
# timout decorator should be applied to the fit_path function
# @timeout_decorator or @func_set_timeout deliver a faster solutio / easy to implement
# @timeout_decorator.timeout(180)
# @func_set_timeout(60)
def bnb(X, y, max_nonzeros, lam=1e-2, threshold=0.0, normalize=True, corrected_coefficients=True, return_only_max_nonzeros=False, padding=False, timeout=60):
    ncols = X.shape[1]
    max_nonzeros = min(max(1, max_nonzeros), ncols)
    # try: sols = func_timeout(timeout, fit_path, args=(X.astype(np.float64), np.ravel(y).astype(np.float64)), kwargs={'lambda_2':lam, 'max_nonzeros':max_nonzeros, 'normalize':normalize, 'intercept':False})
    try: sols = func_timeout(timeout, fit_path, args=(add_columns(X, nums=np.maximum(5-ncols,0), val=1.0).astype(np.float64), np.ravel(y).astype(np.float64)), kwargs={'lambda_2':lam, 'max_nonzeros':max_nonzeros, 'normalize':normalize, 'intercept':False})
    except FunctionTimedOut: return None
    # For debug
    # print(sols)
    bnb_sols = []
    com = 0
    for i in range(len(sols)):
        w = sols[i]['B']
        consider_indices = np.nonzero(w)[0]
        new_com = len(consider_indices)
        # to ensure that the number of supports is increasing.
        if com < new_com <= max_nonzeros: # why not treat max_nonzeros as the max number of cols of bnb_sols?
            com = new_com
            if corrected_coefficients:
                X_con = X[:, consider_indices]
                if threshold > 0.0:
                    w_consider_indices, convergence_flag = recursive_stlsq(X_con, y, threshold=threshold, rcond=None)
                    if convergence_flag: 
                        w[consider_indices] = w_consider_indices.flatten()
                    else: print("Error: recursive_stlsq gives 'False' convergence_flag...")
                else: w[consider_indices] = np.linalg.lstsq(X_con, y, rcond=None)[0].flatten()
                # w[consider_indices] = np.linalg.lstsq(X_con, y, rcond=None)[0].flatten()
            bnb_sols.append(w.reshape(-1, 1))

    if len(bnb_sols) > 0: bnb_sols = np.hstack(bnb_sols)[:ncols, :]
    else: bnb_sols = brute_force(X, y, support_size=max_nonzeros, include=(), top=1, alpha=0.0)
    if return_only_max_nonzeros: return bnb_sols[:, -1:]
    # why not dedicate a col for a number of supports instead of padding?
    elif padding: return add_columns(bnb_sols, nums=max_nonzeros-bnb_sols.shape[1], val=0.0)
    else: return bnb_sols

# a score function
# change to an information criterion with an unique penalty (the last row of X?) for each feature
def mse_score_function(X, y):
    # X and y may be in the pd.DataFrame format
    X = np.array(X); y = np.array(y)
    n_cols = X.shape[1]
    ref = mean_squared_error(X@np.linalg.lstsq(X, y, rcond=None)[0], y)
    scores = []
    for j in range(n_cols):
        X_tmp = X[:, list(range(j))+list(range(j+1, n_cols))]
        w = np.linalg.lstsq(X_tmp, y, rcond=None)[0]
        # scores.append(max(mean_squared_error(X_tmp@w, y)-ref, 0.0))
        scores.append(mean_squared_error(X_tmp@w, y)-ref)
    return np.array(scores, dtype=object)

# a score function
def p_regression(X, y):
    p_values = sm.OLS(y, X).fit().summary2().tables[1]['P>|t|'].values
    scores = 1-p_values
    return scores/scores.sum()

def backward_refinement(feature_hierarchy, dataset, mode="SelectKBest", ic_type="aic", sk_normalize_axis=0, verbose=False):
    XX, yy = dataset
    mode = np.argmax([fuzz.ratio(mode.lower(), m) for m in ['selectkbest', 'rfe', 'mrmr_regression']])
    if verbose: print(['SelectKBest', 'RFE_PLinearRegression', 'mrmr_regression'][mode])
    score_track = ScoreTracker({})
    history = set()
    for init_features in feature_hierarchy:
        init_features = tuple(init_features)
        ols_res = sm.OLS(yy, XX[:, init_features]).fit()
        if ic_type == "aicc":
            AIC = sm_aicc(ols_res.llf, XX.shape[0], len(ols_res.params))
        elif ic_type == "hqic": 
            AIC = sm_hqic(ols_res.llf, XX.shape[0], len(ols_res.params))
        else:
            AIC = getattr(ols_res, ic_type)
        score_track.add(init_features, AIC)
        if len(init_features) > 1:
            if verbose: print(init_features)
            for nfl in range(len(init_features)-1, 0, -1):
                # if score_track.is_in_track(init_features): break
                if init_features not in history: history.add(init_features)
                else: break
                if mode == 0:
                    sf = SelectKBest(mse_score_function, k=nfl).fit(XX[:, init_features], yy.ravel())
                    init_features = [init_features[e] for e in map(lambda e: int(e.replace('x', '')), sf.get_feature_names_out().tolist())]
                elif mode == 1:
                    sf = RFE(MLinearRegression(fit_intercept=False, normalize=True),
                             n_features_to_select=nfl, importance_getter='feature_importances_').fit(XX[:, init_features], yy)
#                    sf = RFE(PLinearRegression(fit_intercept=False, normalize=True),
#                             n_features_to_select=nfl, importance_getter='feature_importances_').fit(XX[:, init_features], yy)
                    init_features = [init_features[e] for e in np.where(sf.support_==True)[0].tolist()]
                elif mode == 2:
                    sf = sorted(mrmr_regression(X=pd.DataFrame(sk_normalize(XX[:, init_features], axis=sk_normalize_axis)),
                                                        y=pd.Series(yy.ravel()),
                                                        K=nfl, relevance=composite_function(lambda x: pd.Series(x.ravel()), mse_score_function),
                                                        redundancy='c', denominator='mean', show_progress=False))
                    init_features = [init_features[e] for e in sf]
                init_features = tuple(init_features)
                ols_res = sm.OLS(yy, XX[:, init_features]).fit()
                if ic_type == "aicc":
                    AIC = sm_aicc(ols_res.llf, XX.shape[0], len(ols_res.params))
                elif ic_type == "hqic": 
                    AIC = sm_hqic(ols_res.llf, XX.shape[0], len(ols_res.params))
                else:
                    AIC = getattr(ols_res, ic_type)
                score_track.add(init_features, AIC)
                if verbose: print(init_features)
            if verbose: print('-'*50)
    return score_track

# moving to what index and what percent gained
def check_percent(bic_scores, complexities):
    slope = (bic_scores[1:]-bic_scores[:-1])/(complexities[1:]-complexities[:-1])
    slope_index = np.argmin(slope)
    percent_improve = 100*np.abs(bic_scores[slope_index+1]-bic_scores[slope_index])/np.abs(bic_scores[slope_index])
    percent_from_1 = 100*np.abs(bic_scores[slope_index+1]-bic_scores[0])/np.abs(bic_scores[0])
    return slope_index+1, percent_improve, percent_from_1

def get_decreasing_vals(bic_scores, complexities):
    ok_indices = []
    mini = np.inf
    for i in range(len(bic_scores)):
        score = bic_scores[i]
        if score < mini:
            ok_indices.append(i)
            mini = score
    return np.array(bic_scores)[ok_indices], np.array(complexities)[ok_indices], ok_indices

# Improvement per 1 support
def find_transition(bic_scores, complexities, percent_oks=(0.099, 0.099), last_improve=0.0):
    if isinstance(percent_oks, float): percent_ok1, percent_ok2 = percent_oks, percent_oks
    else: percent_ok1, percent_ok2 = percent_oks
    assert len(bic_scores) == len(complexities)
    assert percent_ok1 > 0 and percent_ok2 > 0
    if len(complexities) < 2: return complexities[0]
    ### main code ###
    ref_index = 0
    ref_bic = bic_scores[ref_index]
    trans_index = np.argmin((bic_scores[1:]-bic_scores[:-1])/(complexities[1:]-complexities[:-1]))
    com_index = trans_index+1
    improve = (ref_bic-bic_scores[com_index])/(complexities[com_index]-complexities[ref_index])
    percent_improve = improve/np.abs(ref_bic)
    if percent_improve > percent_ok1 and improve > percent_ok2*last_improve:
        print(f"{complexities[com_index]} improves {complexities[ref_index]}")
        if last_improve > 0.0: print("Percent improve:", percent_improve, improve/last_improve)
        else: print("Percent improve:", percent_improve)
        return find_transition(bic_scores[com_index:], complexities[com_index:], percent_oks, last_improve=improve)
    else:
        print(f"{complexities[com_index]} does not improve {complexities[ref_index]}")
        print("Percent improve:", percent_improve)
        return complexities[ref_index]

# Paper: Only the value of improvement is considered
def find_transition_V2(bic_scores, complexities, percent_oks=(0.099, 0.099), last_improve=0.0):
    if isinstance(percent_oks, float): percent_ok1, percent_ok2 = percent_oks, percent_oks
    else: percent_ok1, percent_ok2 = percent_oks
    assert len(bic_scores) == len(complexities)
    assert percent_ok1 > 0 and percent_ok2 > 0
    if len(complexities) < 2: return complexities[0]
    ### main code ###
    ref_index = 0
    ref_bic = bic_scores[ref_index]
    trans_index = np.argmin((bic_scores[1:]-bic_scores[:-1])/(complexities[1:]-complexities[:-1]))
    com_index = trans_index+1
    improve = (ref_bic-bic_scores[com_index])
    percent_improve = improve/np.abs(ref_bic)
    if percent_improve > percent_ok1 and improve > percent_ok2*last_improve:
        print(f"{complexities[com_index]} improves {complexities[ref_index]}")
        if last_improve > 0.0: print("Percent improve:", percent_improve, improve/last_improve)
        else: print("Percent improve:", percent_improve)
        return find_transition(bic_scores[com_index:], complexities[com_index:], percent_oks, last_improve=improve)
    else:
        print(f"{complexities[com_index]} does not improve {complexities[ref_index]}")
        print("Percent improve:", percent_improve)
        return complexities[ref_index]

# paper
def find_transition_V3(bic_scores, complexities, percent_ok=0.099, last_improve=0.0):
    assert len(bic_scores) == len(complexities)
    assert percent_ok > 0
    ### main code ###
    ref_index = 0
    if len(complexities) < 2: return complexities[ref_index]

    trans_index = np.argmin((bic_scores[1:]-bic_scores[:-1])/(complexities[1:]-complexities[:-1]))
    com_index = trans_index+1

    ref_bic = bic_scores[ref_index]
    if last_improve > 0: improve = (ref_bic-bic_scores[com_index])/(complexities[com_index]-complexities[ref_index])
    else: improve = (ref_bic-bic_scores[com_index])
    percent_improve = improve/np.abs(ref_bic)

    if improve > percent_ok*max(np.abs(ref_bic), last_improve):
        print(f"{complexities[com_index]} improves {complexities[ref_index]}")
        if last_improve > 0.0: print("Percent improve:", percent_improve, improve/last_improve)
        else: print("Percent improve:", percent_improve)
        return find_transition_V3(bic_scores[com_index:], complexities[com_index:], percent_ok, last_improve=improve)
    else:
        print(f"{complexities[com_index]} does not improve {complexities[ref_index]}")
        print("Percent improve:", percent_improve)
        return complexities[ref_index]

# paper
def find_transition_V4(bic_scores, complexities, percent_ok=0.099, last_improve=0.0):
    assert len(bic_scores) == len(complexities)
    assert percent_ok > 0
    ### main code ###
    ref_index = 0
    if len(complexities) < 2: return complexities[ref_index]

    trans_index = np.argmin((bic_scores[1:]-bic_scores[:-1])/(complexities[1:]-complexities[:-1]))
    com_index = trans_index+1

    ref_bic = bic_scores[ref_index]
    diff_com = complexities[com_index]-complexities[ref_index]
    improve = ref_bic-bic_scores[com_index]
    if last_improve > 0: improve = improve/diff_com

    # only for print
    percent_improve = improve/np.abs(ref_bic)

    if improve > percent_ok*max(np.abs(ref_bic), last_improve):
        print(f"{complexities[com_index]} improves {complexities[ref_index]}")
        if last_improve > 0.0: print("Percent improve:", percent_improve, improve/last_improve)
        else: print("Percent improve:", percent_improve)
        return find_transition_V4(bic_scores[com_index:], complexities[com_index:], percent_ok, last_improve=improve)
    else:
        print(f"{complexities[com_index]} does not improve {complexities[ref_index]}")
        print("Percent improve:", percent_improve)
        return complexities[ref_index]

# percent_ok=0.095, epsilon=0.005 also a good choice
def find_transition_V5(bic_scores, complexities, percent_ok=0.09, epsilon=0.01, last_improve=0.0, verbose=True):
    assert len(bic_scores) == len(complexities)
    assert percent_ok > 0
    ### main code ###
    if len(complexities) < 2: return complexities[0]

    trans_index = np.argmin((bic_scores[1:]-bic_scores[:-1])/(complexities[1:]-complexities[:-1]))
    com_index = trans_index+1

    ref_index = trans_index
    if last_improve > 0: ref_index = 0
    ref_bic = bic_scores[ref_index]
    improve = (ref_bic-bic_scores[com_index])/(complexities[com_index]-complexities[ref_index])

    # percent_improve -> only for print
    # percent_improve = improve/np.abs(ref_bic) # old but fine...
    
    # Newly add on 2023/01/25
    percent_improve = improve/max(np.abs(ref_bic), last_improve)

    if improve > percent_ok*max(np.abs(ref_bic), last_improve):
        if verbose:
            print(f"{complexities[com_index]} improves {complexities[ref_index]}")
            print("Percent improve:", percent_improve) # percent_improve > 1 ได้
        return find_transition_V5(bic_scores[com_index:], complexities[com_index:], percent_ok+epsilon, epsilon, last_improve=improve, verbose=verbose)
    else:
        if verbose:
            print(f"{complexities[com_index]} does not improve {complexities[ref_index]}")
            print("Percent improve:", percent_improve)
        return complexities[ref_index]

def recursive_stlsq(XX, yy, threshold=0.0, rcond=None):
    convergence_flag = False
    ncols = XX.shape[-1]
    indices = np.array([i for i in range(ncols)])
    for _ in range(ncols):
        w = np.zeros(ncols)
        w[indices] = np.linalg.lstsq(XX[:, indices], yy, rcond=rcond)[0]
        if np.all(np.abs(w[indices])>=threshold):
            convergence_flag = True; break
        indices = np.where(np.abs(w)>=threshold)[0]
    return w, convergence_flag 

# why brute force ไม่เลือกตาม information criterion ที่ใช้ pen also from get_order_degree!!! (NOT JUST USING MSE)
def brute_force(X, y, support_size=1, include=(), top=1, alpha=0.0):
    # assert X.dtype == y.dtype
    y = y.ravel()
    include = set(include)
    n_features = X.shape[1]
    selections = [sel for sel in itertools.combinations(range(n_features), support_size) if include.issubset(set(sel))]
    scores = []
    for sel in selections:
        X_sel = X[:, sel]
        w = np.linalg.lstsq(X_sel, y, rcond=None)[0]
        # mse = ((X_sel@w-y)**2).mean()
        mse = (np.linalg.norm(X_sel@w-y, ord=2))**2+(alpha*np.linalg.norm(w, ord=0)**2)
        # ww = np.zeros(n_features).astype(X.dtype) # By this line, use .ravel() to y before calling the function.
        ww = np.zeros(n_features) # By this line, use .ravel() to y before calling the function.
        ww[list(sel)] = w
        scores.append((mse, ww))
    sorted_indices = np.argsort([mse for mse,w in scores])
    scores = np.array(scores, dtype=object)[sorted_indices]
    scores = [w for mse,w in scores]
    return np.array(scores[:top]).T

def brute_force_all_subsets(X, y, min_support_size=1, max_support_size=None):
    x_shape = X.shape
    y_shape = y.shape
    assert len(x_shape) == 2
    assert len(y_shape) == 2 or len(y_shape) == 1
    if len(y_shape) == 2:
        assert y_shape[-1] == 1
    assert x_shape[0] == y_shape[0]
    if max_support_size is None:
        n_features = X.shape[-1]
    else:
        n_features = max_support_size
    bf_solve = np.hstack([brute_force(X, y, s) for s in trange(min_support_size, n_features+1)]).T
    best_subsets = [nonz(bf) for bf in bf_solve]
    return bf_solve, best_subsets

# a more generalized of brute_force by Scikit_learn package
def sk_brute_force(X, y, support_size=1, include=(), top=1, MODEL=SkLinearRegression, kwargs={"fit_intercept":False}):
    y = y.ravel()
    include = set(include)
    n_features = X.shape[1]
    selections = [sel for sel in itertools.combinations(range(n_features), support_size) if include.issubset(set(sel))]
    scores = []; wws = []
    for sel in selections:
        X_sel = X[:, sel]
        model = MODEL(**kwargs); model.fit(X_sel, y)
        w = model.coef_
        score = model.score(X_sel, y)
        ww = np.zeros(n_features) # By this line, use .ravel() to y before calling the function.
        ww[list(sel)] = w
        scores.append(score)
        wws.append(ww)
    top_indices = np.argsort(scores)[::-1][:top]
    return np.array(wws)[top_indices].T

# Only used with rhs_description in PDE-FIND algo
def get_order_degree(text, verbose=False, accumulate=True):
    patterns = [r"u_(\{[x]+\})", r"u(\^[0-9]+)?"]
    res = re.findall('|'.join(patterns), text)
    if verbose: print(res)
    out = []
    for order, deg in res:
        if len(order) == 0: order = 0
        else: order = Counter(order)['x']
        if len(deg) == 0: deg = 1
        elif deg[0] == '^': deg = int(deg[1:])
        else: deg = 0; print("Found unexpected pattern...")
        out.append(np.array([order, deg]))
    if len(out) > 0: 
        out = np.vstack(out).astype(np.int32)
        if accumulate: return out.sum(axis=0)
        return out
    else: return np.array(out).astype(np.int32)

def prediction_group(Theta_grouped, xi):
    return np.hstack([Theta_grouped[j].dot(xi[j]) for j in range(len(Theta_grouped))]).reshape(-1, 1)

def spatial_temporal_group(Theta, Ut, domain_shape, dependent="temporal"):
    # print("INPUT: domain_shape = len(x), len(t)")
    n, m = domain_shape
    assert n*m == Theta.shape[0], Ut.shape[0]
    if dependent == "temporal":
        Theta_grouped = [(Theta[j*n:(j+1)*n,:]).real for j in range(m)]
        Ut_grouped = [(Ut[j*n:(j+1)*n]).real for j in range(m)]
    elif dependent == "spatial":
        Theta_grouped = [(Theta[n*np.arange(m)+j,:]).real for j in range(n)]
        Ut_grouped = [(Ut[n*np.arange(m)+j]).real for j in range(n)]
    else: return
    return Theta_grouped, Ut_grouped

@lru_cache(maxsize=None)
def gen_blocks(n_steps, size):
    overlap = 0
    for overlap in range(size):
        if (n_steps-size)%(size-overlap) == 0:
            break
    steps = [i for i in range(n_steps)]
    out = [tuple(steps[i:i+size]) for i in range(0, len(steps)-overlap, size-overlap)]
    return out

def sub_grouped_data(grouped_data, block):
    feature_grouped, target_grouped = grouped_data
    n_groups = len(block)
    steps, n_basis_candidates= feature_grouped[0].shape
    F = block_diag(*[feature_grouped[b] for b in block])
    T = np.vstack([target_grouped[b] for b in block])
    F = F[:, np.hstack([[i*n_basis_candidates+j for i in range(n_groups)] for j in range(n_basis_candidates)])]
    return F, T

# Originally used in SGTRidge
def group_normalize(AAs, bbs, normalize=2):
    As, bs = AAs.copy(), bbs.copy()
    m = len(As); n,D = As[0].shape
    # get norm of each column
    candidate_norms = np.zeros(D)
    for i in range(D):
        candidate_norms[i] = np.linalg.norm(np.vstack([A[:,i] for A in As]), normalize)
    norm_bs = [m*Norm(b, normalize) for b in bs]
    # normalize
    for i in range(m):
        As[i] = As[i].dot(np.diag(candidate_norms**-1))
        bs[i] = bs[i]/norm_bs[i]
    return As, bs

def fit_grouped_data(reg_model, grouped_data, considered_indices=None):
    Theta_grouped, Ut_grouped = grouped_data
    n_chunks = len(Ut_grouped)
    if considered_indices is not None:
        return np.hstack([reg_model.fit(Theta_grouped[j][:, considered_indices], Ut_grouped[j].ravel()).coef_.reshape(-1, 1) for j in range(n_chunks)])
    return np.hstack([reg_model.fit(Theta_grouped[j], Ut_grouped[j].ravel()).coef_.reshape(-1, 1) for j in range(n_chunks)])

def linear_fit_grouped_data(grouped_data, considered_indices=None):
    Theta_grouped, Ut_grouped = grouped_data
    n_chunks = len(Ut_grouped)
    if considered_indices is not None:
        return np.hstack([np.linalg.lstsq(Theta_grouped[j][:, considered_indices], Ut_grouped[j], rcond=None)[0] for j in range(n_chunks)])
    return np.hstack([np.linalg.lstsq(Theta_grouped[j], Ut_grouped[j], rcond=None)[0] for j in range(n_chunks)])

def mse_grouped_data(w, grouped_data, considered_indices=None):
    Theta_grouped, Ut_grouped = grouped_data
    n_chunks = len(Ut_grouped)
    if considered_indices is None: considered_indices = [i for i in range(Theta_grouped[0].shape[1])]
    return np.mean([((Theta_grouped[j][:, considered_indices].dot(w[:, j:j+1])-Ut_grouped[j])**2).mean() for j in range(n_chunks)])

class ScoreTracker(object):
    def __init__(self, init_dict={}):
        super(ScoreTracker, self).__init__()
        self.track = init_dict
    def __repr__(self,):
        # return str(json.dumps(self.get_track(), sort_keys=True, indent=4, default=str))
        return str(self.get_track())
    def __str__(self,):
        # return str(json.dumps(self.get_track(), sort_keys=True, indent=4, default=str))
        return str(self.get_track())
    def __add__(self, a_tracker):
        o_track = {}
        a_track = a_tracker.get_track()
        m_track = self.get_track()
        for k in set(a_track.keys()).union(set(m_track.keys())):
            if a_track[k][-1] <= m_track[k][-1]: o_track[k] = a_track[k]
            else: o_track[k] = m_track[k]
        return ScoreTracker(init_dict=o_track)
    def add(self, indices, score, com=None):
        if com == None: com = len(indices)
        if com not in self.track or score < self.track[com][-1]:
            self.track[com] = (tuple(indices), score)
    def delete(self, com):
        return self.track.pop(com, None)
    def get_track(self,):
        return self.track
    def clear_track(self,):
        self.track = {}
    def is_in_track(self, indices):
        if tuple(indices) in self.track: return True
        else: return False
    def update_track(self, a_track):
        for k in set(self.track.keys()).union(set(a_track.keys())):
            if self.track[k][-1] > a_track[k][-1]: 
                self.track[k] = a_track[k]
        return self.get_track()

class ABESS(ps.optimizers.BaseOptimizer):
    def __init__(self, abess_kw=None, group=None, is_normal=False, normalize_columns=False):
        try: super(ABESS, self).__init__(fit_intercept=False, copy_X=True, normalize_columns=normalize_columns)
        except TypeError: super(ABESS, self).__init__(copy_X=True, normalize_columns=normalize_columns)
        self.abess_kw = abess_kw
        self.group = group
        self.is_normal = is_normal
    def _reduce(self, x, y):
        model = LinearRegression(**self.abess_kw)
        model.fit(x, y.ravel(), group=self.group, is_normal=self.is_normal)
        self.coef_ = model.coef_
        self.ind_ = [self.coef_ != 0.0]

class L0BNB(ps.optimizers.BaseOptimizer):
    def __init__(self, max_nonzeros=None, lam=1e-2, threshold=0.0, is_normal=False, normalize_columns=False):
        try: super(L0BNB, self).__init__(fit_intercept=False, copy_X=True, normalize_columns=normalize_columns)
        except TypeError: super(L0BNB, self).__init__(copy_X=True, normalize_columns=normalize_columns)
        self.max_nonzeros = max_nonzeros
        self.lam = lam
        self.threshold = threshold
        self.is_normal = is_normal
    def _reduce(self, x, y):
        # Old and wrong code
#        self.coef_ = bnb(x, y.ravel(), self.max_nonzeros, lam=self.lam, normalize=self.is_normal, 
#                corrected_coefficients=True, return_only_max_nonzeros=True).reshape(1, -1)
#        self.ind_ = [self.coef_ != 0.0]
        sols = []
        for i in range(y.shape[-1]):
            sol = bnb(x, y[:,i:i+1].ravel(), self.max_nonzeros, lam=self.lam, threshold=self.threshold, 
                    normalize=self.is_normal,  corrected_coefficients=True, return_only_max_nonzeros=True).reshape(1, -1)
            sols.append(sol)
        self.coef_ = np.vstack(sols)
        self.ind_ = [c != 0.0 for c in self.coef_]

class BESS(ps.optimizers.BaseOptimizer):
    def __init__(self, bess_kw=None, group=None, is_normal=False, normalize_columns=False):
        try: super(BESS, self).__init__(fit_intercept=False, copy_X=True, normalize_columns=normalize_columns)
        except TypeError: super(BESS, self).__init__(copy_X=True, normalize_columns=normalize_columns)
        self.bess_kw = bess_kw
        self.group = group
        self.is_normal = is_normal
    def _reduce(self, x, y):
        model = PdasLm(**self.bess_kw)
        model.fit(x, y.ravel(), group=self.group, is_normal=self.is_normal)
        self.coef_ = model.beta
        self.ind_ = [self.coef_ != 0.0]

class BruteForceRegressor(ps.optimizers.BaseOptimizer):
    def __init__(self, support_size=None, include=(), top=1, normalize_columns=False):
        try: super(BruteForceRegressor, self).__init__(fit_intercept=False, copy_X=True, normalize_columns=normalize_columns)
        except TypeError: super(BruteForceRegressor, self).__init__(copy_X=True, normalize_columns=normalize_columns)
        self.support_size = support_size
        self.include = include
        self.top = top
    def _reduce(self, x, y):
        coef = brute_force(x, y, support_size=self.support_size, include=self.include, top=self.top)
        self.coef_ = coef.flatten()
        self.ind_ = [self.coef_ != 0.0]

class CandidateLibrary(ps.optimizers.BaseOptimizer):
    def __init__(self, kwargs={'fit_intercept':False, 'copy_X':True, 'normalize_columns':False}):
        super(CandidateLibrary, self).__init__(**kwargs)
        self.preprocessed_data = None
    def _reduce(self, x, y):
        self.preprocessed_data = x, y
        # self.coef_ = np.zeros(x.shape[1])
        # self.ind_ = [self.coef_ != 0.0]

def convert2latex(text):
    s = re.sub('1u', '1 u', text).split()
    for i in range(len(s)): # replace 1 with x
        if '1' in s[i]:
            x = 'x'*(s[i].count('1'))
            s[i] = re.sub('1+', f"{{{x}}}", s[i])
    for i in range(len(s)): # change format of u
        u = s[i].count('u')
        if 'x' in s[i] and u > 2:
            s[i] = re.sub('u+', f"u^{u-1}u", s[i])
        elif 'x' not in s[i] and u > 1:
            s[i] = re.sub('u+', f"u^{u}", s[i])
    return ''.join(s)

def ps_features(target, time, feature_library, kwargs={'fit_intercept':False, 'copy_X':True, 'normalize_columns':False}, optimizer=CandidateLibrary, differentiation_method=None, feature_names=['u'], get_latex=True, center_y=False, verbose=True):
    opt = optimizer(kwargs)
    if differentiation_method is None:
        if hasattr(feature_library, "differentiation_method") and hasattr(feature_library, "differentiation_kwargs"):
            differentiation_method = feature_library.differentiation_method(axis=1, **feature_library.differentiation_kwargs)
        else:
            if verbose:
                print("differentiation method or differentiation_kwargs is not implemented in feature_library.")
    cl = ps.SINDy(feature_library=feature_library, optimizer=opt, differentiation_method=differentiation_method, feature_names=feature_names)
    if len(target.shape) < 3: target = np.expand_dims(target, -1)
    cl.fit(target, t=time)
    X_pre, y_pre = cl.optimizer.preprocessed_data
    if center_y:
        y_pre = y_pre - y_pre.mean(axis=0)
    # Bugs for convert2latex (unfixed)
    if get_latex: return X_pre, y_pre, list(map(convert2latex, feature_library.get_feature_names()))
    else: return X_pre, y_pre, feature_library.get_feature_names()
        
