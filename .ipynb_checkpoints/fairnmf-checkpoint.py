import numpy as np
import cvxpy as cp

rng = np.random.default_rng()

MACHINE_EPSILON = np.finfo(np.float32).eps

def FairNMF_step_MU(Xs, Ws, H, min_errs, norms, coefs, c_rule):
    errs = (np.array([np.linalg.norm(X - W @ H, ord='fro') for X, W in zip(Xs,Ws)]) - min_errs) / norms 
    max_i = np.argmax(errs)

    if c_rule == 'decaying_lr':
        coefs[max_i] += 1
    elif c_rule == 'largest':
        coefs[:] = 0
        coefs[max_i] = 1
    elif c_rule == 'fixed_lr':
        coefs = coefs*0.9
        coefs[max_i] += 0.1
    else:
        raise ValueError("Invalid value for c_rule.")

    W_step = np.concatenate([c * W / n for (c, W, n) in zip(coefs, Ws, norms)])
    X_step = np.concatenate([c * X / n for (c, X, n) in zip(coefs, Xs, norms)])
    
    denom = W_step.T @ W_step @ H
    denom[denom < MACHINE_EPSILON] = MACHINE_EPSILON
    H *= (W_step.T @ X_step) / denom
    
    HHt = H @ H.T
    for j in range(len(Xs)):        
        denom = Ws[j] @ HHt
        denom[denom < MACHINE_EPSILON] = MACHINE_EPSILON
        Ws[j] *= (Xs[j] @ H.T) / denom
    
    return Ws, H, coefs, (np.array([np.linalg.norm(X - W @ H, ord='fro') for X, W in zip(Xs,Ws)]) - min_errs) / norms
    
def FairNMF_MU(Xs, num_topics, max_iter = 1000, rel_tol=1e-3, c_rule='decaying_lr'):
    '''
    Performs a FairNMF decomposition.  Xs is a sequence of NumPy arrays each with shape (_, d) where the
    first dimension may differ.  Each array X in Xs gets decomposed into W @ H where W has shape (_, num_topics)
    and H has shape (num_topics, d).  The H array is shared between all X.
    
    Inputs
    Xs - A sequence of 2d NumPy arrays with common second dimension size, representing the data to be
    decomposed.  Each array is assumed to be a distinct group of interest.
    num_topics - An int representing the number of topics to find in the data.  Equivalently, an int representing
    the dimension to embed the data into.
    max_iter - An int representing the maximum number of multiplicative update steps to perform. Default 1000.
    rel_tol - Floating point representing the decrease in relative error for each group at which convergence
    is declared to be reached.  Default 1e-3.
    c_rule - Rule to update group coefficients.  One of 'decaying_lr', 'fixed_lr', or 'largest'.  The recommended
    choice is 'decaying_lr', with the other options included to demonstrate unsatisfactory alternatives considered
    in the paper. Default 'decaying_lr'.
    
    Outputs
    Ws - A list of 2d NumPy arrays, one for each array in X, representing the representation of each
    group in the topic space.  Each array has shape (_, num_topics) with the first dimension of agreeing
    with the first dimension of the corresponding array in Xs.
    H - A NumPy array of shape (num_topics, d), representing the features associated with each topic.
    errs - A NumPy array of shape (len(Xs), T) where T is the number of iterations performed.  errs[i,j] is the
    reconstruction error of the i-th group at the j-th iteration.
    min_errs - A NumPy array of shape (len(Xs),) containing the minimum reconstruction error of each group when
    decomposed separately.
    '''
    # error per group per iteration
    if not(max_iter is None):
        errs = np.zeros((len(Xs), max_iter))
    min_errs = np.zeros((len(Xs),))
    norms = np.array([np.linalg.norm(X, ord='fro') for X in Xs])
    
    #NMF approximate of optimal
    for i in range(len(Xs)):
        opt_errs = []
        for approx_itr in range(5):
#            model = NMF(n_components=num_topics, max_iter = 10000, solver='mu', init="random")
#            W = model.fit_transform(Xs[i])
#            H = model.components_
            W, H, _ = NMF(Xs[i], num_topics, max_iter=max_iter, rel_tol=rel_tol)
            opt_errs.append(np.linalg.norm(Xs[i] - W@H, 'fro'))
        min_errs[i] = np.nanmean(opt_errs)

    # Initializing matrices
    Ws = [rng.random((X.shape[0],) + (num_topics,)) for X in Xs]
    H = rng.random((num_topics,) + (Xs[0].shape[1],))

    # Solve
    coefs = np.ones((len(Xs),)) / len(Xs)
    
    if max_iter is None:
        prev_err = None
        while True:
            Ws, H, coefs, iter_err = FairNMF_step_MU(Xs, Ws, H, min_errs, norms, coefs, c_rule)
            if not(prev_err is None) and not (rel_tol is None) and np.max(np.abs(iter_err-prev_err)/(np.abs(iter_err))) < rel_tol:
                #convergence reached
                break
            prev_err = iter_err

        return Ws, H, iter_err.reshape((-1,1)), min_errs
        
    else:
        for i in range(max_iter):
            Ws, H, coefs, iter_err = FairNMF_step_MU(Xs, Ws, H, min_errs, norms, coefs, c_rule)
            errs[:,i] = iter_err
            if i > 0 and not (rel_tol is None) and np.max(np.abs(errs[:,i-1]-errs[:,i])/(np.abs(errs[:,i]))) < rel_tol:
                #convergence reached
                break

        return Ws, H, errs[:,:i+1], min_errs

# alternating minimization with cvxpy
def FairNMF_step_AM(Xs, Ws, W_vars, W_params, W_probs, H, H_var, H_param, H_prob):
    
    # Initialize all vars and params
    H_var.value = H
    for W_v, W_p, W in zip(W_vars, W_params, Ws):
        W_v.value = W
        W_p.value = W

    # Find H fixing the Ws
    try:
        H_prob.solve(warm_start=True, solver=cp.ECOS, ignore_dpp=True)
    except:
        # ECOS failed
        H_prob.solve(warm_start=True, solver=cp.SCS, ignore_dpp=True)
    
    # Get the new H value and update the H involved in the other problems of finding Ws fixing H
    H_new = H_var.value
    H_param.value = H_new

    # Update the Ws with the optimized value given the fixed H    
    errs = []
    Ws_new = []

    for W_v, W_pb in zip(W_vars, W_probs):
        try:
            errs.append(W_pb.solve(warm_start=True, solver=cp.OSQP, ignore_dpp=True))
            Ws_new.append(W_v.value)
        except:
            # OSQP failed
            errs.append(W_pb.solve(warm_start=True, solver=cp.SCS, ignore_dpp=True))
            Ws_new.append(W_v.value)
    
    return Ws_new, H_new, errs
    
def FairNMF_AM(Xs, num_topics, max_iter = 10, rel_tol=1e-8):
    '''
    Performs a FairNMF decomposition.  Xs is a sequence of NumPy arrays each with shape (_, d) where the
    first dimension may differ.  Each array X in Xs gets decomposed into W @ H where W has shape (_, num_topics)
    and H has shape (num_topics, d).  The H array is shared between all X.
    
    Inputs
    Xs - A sequence of 2d NumPy arrays with common second dimension size, representing the data to be
    decomposed.  Each array is assumed to be a distinct group of interest.
    num_topics - An int representing the number of topics to find in the data.  Equivalently, an int representing
    the dimension to embed the data into.
    max_iter - An int representing the maximum number of convex optimizations problems to solve per variable.
    Default 10.
    rel_tol - Floating point representing the decrease in relative error for each group at which convergence
    is declared to be reached.  Default 1e-8.
    
    Outputs
    Ws - A list of 2d NumPy arrays, one for each array in X, representing the representation of each
    group in the topic space.  Each array has shape (_, num_topics) with the first dimension of agreeing
    with the first dimension of the corresponding array in Xs.
    H - A NumPy array of shape (num_topics, d), representing the features associated with each topic.
    errs - A NumPy array of shape (len(Xs), T) where T is the number of iterations performed.  errs[i,j] is the
    reconstruction error of the i-th group at the j-th iteration.
    min_errs - A NumPy array of shape (len(Xs),) containing the minimum reconstruction error of each group when
    decomposed separately.
    '''
    # error per group per iteration
    if not(max_iter is None):
        errs = np.zeros((len(Xs), max_iter))
    min_errs = np.zeros((len(Xs),))
    norms = np.array([np.linalg.norm(X, ord='fro') for X in Xs])
        
    #NMF approximate of optimal
    for i in range(len(Xs)):
        opt_errs = []
        for approx_itr in range(5):
#            model = NMF(n_components=num_topics, max_iter = 10000, solver='mu', init="random")
#            W = model.fit_transform(Xs[i])
#            H = model.components_
            W, H, _ = NMF(Xs[i], num_topics, max_iter=max_iter, rel_tol=rel_tol)
            opt_errs.append(np.linalg.norm(Xs[i] - W@H, 'fro'))
        min_errs[i] = np.nanmean(opt_errs)

    # Initializing matrices
    Ws = [rng.random((X.shape[0],) + (num_topics,)) for X in Xs]
    H = rng.random((num_topics,) + (Xs[0].shape[1],))

    # Variables for CVXPY problems
    H_var = cp.Variable(H.shape)
    s_var = cp.Variable((1,))
    W_vars = [cp.Variable(W.shape) for W in Ws]
    
    # Parameters for CVXPY problems
    H_param = cp.Parameter(H.shape)
    W_params = [cp.Parameter(W.shape) for W in Ws]
    
    # Optimization problem for H
    objective = cp.Minimize(s_var)
    constraints = [H_var >= 0] + [(cp.norm(X - W_p @ H_var, 'fro') - min_e)/norm <= s_var
             for X, W_p, min_e, norm in zip(Xs, W_params, min_errs, norms)]
    
    H_prob = cp.Problem(objective, constraints)

    # Optimization problems for Ws
    W_probs = []
    for i in range(len(Xs)):
        objective = cp.Minimize(cp.sum_squares(Xs[i] - W_vars[i] @ H_param))
        constraints = [W_vars[i] >= 0]
        W_probs.append(cp.Problem(objective, constraints))

    # Solve optimization problems
    if max_iter is None:
        prev_err = None
        while True:
            Ws, H, iter_err = FairNMF_step_AM(Xs, Ws, W_vars, W_params, W_probs, H, H_var, H_param, H_prob)
            iter_err = (np.sqrt(np.array(iter_err)) - min_errs) / norms
            if not(prev_err is None) and not (rel_tol is None) and np.max(np.abs(iter_err-prev_err)/(np.abs(iter_err))) < rel_tol:
                #convergence reached
                break
            prev_err = iter_err

        return Ws, H, iter_err.reshape((-1,1)), min_errs
    else:
        for i in range(max_iter):
            Ws, H, iter_err = FairNMF_step_AM(Xs, Ws, W_vars, W_params, W_probs, H, H_var, H_param, H_prob)
            errs[:,i] = (np.sqrt(np.array(iter_err)) - min_errs) / norms
            if i > 0 and not (rel_tol is None) and np.max(np.abs(errs[:,i-1]-errs[:,i])/(np.abs(errs[:,i]))) < rel_tol:
                #convergence reached
                break

        return Ws, H, errs[:,:i+1], min_errs

#Standard NMF

def NMF_step(X, W, H):
    
    denom = W.T @ W @ H
    denom[denom < MACHINE_EPSILON] = MACHINE_EPSILON
    H *= (W.T @ X) / denom
    
    denom = W @ H @ H.T
    denom[denom < MACHINE_EPSILON] = MACHINE_EPSILON
    W *= (X @ H.T) / denom
    
    return W, H, np.linalg.norm(X - W @ H, ord='fro')
    
def NMF(X, num_topics, max_iter = 1000, rel_tol=1e-3):
    '''
    Performs a NMF decomposition.  X is a NumPy arrays each with shape (n, d). The arrray X gets decomposed 
    into W @ H where W has shape (n, num_topics) and H has shape (num_topics, d).
    
    Inputs
    X - A 2d NumPy representing the data to be decomposed.
    num_topics - An int representing the number of topics to find in the data.  Equivalently, an int representing
    the dimension to embed the data into.
    max_iter - An int representing the maximum number of multiplicative update steos to perform. Default 1000.
    rel_tol - Floating point representing the decrease in relative error for each group at which convergence
    is declared to be reached.  Default 1e-3.
    
    Outputs
    W - A 2d NumPy array, representing each row in the topic space.  The array W has shape (n, num_topics).
    H - A NumPy array of shape (num_topics, d), representing the features associated with each topic.
    errs - A NumPy array of shape (T,) where T is the number of iterations performed.  errs[i] is the
    reconstruction error at the i-th iteration.
    '''
    
    if max_iter is None:
        max_iter = 10000
        
    # error per group per iteration
    errs = np.zeros((max_iter,))

    # Initializing matrices
    W = rng.random((X.shape[0], num_topics))
    H = rng.random((num_topics, X.shape[1]))

    # Solve
    for i in range(max_iter):
        W, H, iter_err = NMF_step(X, W, H)
        errs[i] = iter_err
        if not (rel_tol is None) and i > 0 and np.abs(errs[i-1]-errs[i])/(np.abs(errs[i])) < rel_tol:
            #convergence reached
            break
        
    return W, H, errs[:i+1]