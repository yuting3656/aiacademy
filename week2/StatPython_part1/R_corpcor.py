#!/usr/bin/env python
# coding: utf-8

"""
# Converte R corpcor package to Python version with Pandas, Numpy, statistics.
# Cor, Cov, Var shrinkage method.
# Created by Guo Yuan Li (Jimmy), 2019-03-11.

# References:
# Accurate Ranking of Differentially Expressed Genes by a Distribution-Free Shrinkage Approach 2007, p.7-14
# empirircal sample variance with bessel correction.
# weighted variance with bessel correction, Frequency weights in Wiki.
# https://stats.stackexchange.com/questions/47325/bias-correction-in-weighted-variance
# A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and Implications for Functional Genomics 2005, p.4, 11-13, 26-27
# https://github.com/cran/corpcor/blob/5df4d249ead2dbfe973c0d9edc5310a429e40337/R/shrink.intensity.R
# https://github.com/cran/corpcor/blob/5df4d249ead2dbfe973c0d9edc5310a429e40337/R/fast.svd.R
# https://github.com/cran/corpcor/blob/5df4d249ead2dbfe973c0d9edc5310a429e40337/R/mpower.R
# https://github.com/cran/corpcor/blob/master/R/pvt.powscor.R
# https://github.com/cran/corpcor/blob/master/R/shrink.estimates.R
"""

import pandas as pd
import numpy as np
import scipy as sp


# In[1]:


def estimate_lambda_var(x):
    # Reference: Accurate Ranking of Differentially Expressed Genes by a Distribution-Free Shrinkage Approach 2007, p.7-14
    # Shrinkage t statistic
    # R: estimate.lambda.var and estimate.lambda
    
    empirical_mean = []
    #empirical_var = []
    
    ## n: sample size(rows) , p: number of variables(columns) -
    import numpy as np
    x = np.array(x)
    n = x.shape[0]
    p = x.shape[1]
    
    if n < 3:
        print("Sample size too small !!") 
    
    ## Calculate empirical sample mean for each variable(column) -
    for i in range(x.shape[1]):
        empirical_mean.append(x[:, i].mean())
    
    ## Calculate empirical sample variance for each variable(column) -
    ## empirical_var = v_k
    #for j in range(x.shape[1]):
    #    empirical_var.append(x[:, j].var(ddof = 1))
    
    ## Sample version of W_ik - 
    ## W_ik = ( X_ik - X_ik_mean ) ** 2
    w_ik = x.copy()
    nn = w_ik.shape[0]
    pp = w_ik.shape[1]
    for t in range(w_ik.shape[1]):
        w_ik[:, t] = ( w_ik[:, t] - empirical_mean[t] ) ** 2
    
    ## Sample version of W_ik_mean and Vk -
    ## W_ik_mean = sum of W_ik / nn
    ## Vk = empirical variance of column
    w_k_mean = []
    v_k = []
    for t in range(w_ik.shape[1]):
        w_k_mean.append(w_ik[:, t].mean())
        v_k.append( nn * w_ik[:, t].mean() / (nn-1) )
    
    ## Sample version of Vk_median - 
    from statistics import median
    v_k_median = median(v_k)
    
    ## Sample version of Var(Vk) - 
    v_k_Var = []
    for t in range(w_ik.shape[1]):
        v_k_Var.append( ( (w_ik[:, t] - w_k_mean[t]) ** 2 ).sum() * nn / (nn - 1) ** 3 )
    
    ## lambda_var: Optimal estimated pooling parameter = shrinkage intensity
    ## 0 <= lambda_var <= 1
    lambda_var = min(1., np.array(v_k_Var).sum() / ( (np.array(v_k) - v_k_median) ** 2 ).sum())
        
    return lambda_var, v_k_median, v_k, v_k_Var


# In[2]:


def wt_moments(x, w = None):
    column_mean = []
    column_var = []
    
    ## n: sample size(rows) , p: number of variables(columns) -
    import numpy as np
    x = np.array(x)
    n = x.shape[0]
    p = x.shape[1]
    
    ## Determine the weight for each sample - 
    if not w:
        w = np.repeat(1/n, n, axis=0)
    else:
        if len(np.array(w)) != n:
            print("Length Error: Length of w != Sample size n.")
        else:
            if np.array(w).sum() != 1:
                print("Weighted sum Error: Length of w = Sample size n, But Sum of weights != 1.")
            else:
                w = np.array(w)
    
    ## Compute empirical weighted sample mean for each variable(column) -
    for j in range(x.shape[1]):
        weighted_sum = 0
        for i in range(x.shape[0]):
            weighted_sum += x[i, j] * w[i]
        column_mean.append(weighted_sum)
    
    ## Compute empirical weighted sample variance for each variable(column) -
    ## Check equally weighted or not first - 
    ## True: empirircal sample variance with bessel correction.
    ## False: weighted variance with bessel correction, Frequency weights in Wiki.
    ## Reference: https://stats.stackexchange.com/questions/47325/bias-correction-in-weighted-variance
    if np.all(np.isclose(w, w[0])):
        for j in range(x.shape[1]):
            column_var.append(x[:, j].var(ddof = 1))
    else:
        for j in range(x.shape[1]):
            weighted_sum_square_diff = 0
            sum_w2 = 0
            for i in range(x.shape[0]):
                weighted_sum_square_diff += w[i] * ( (x[i, j] - column_mean[i]) ** 2 )
                sum_w2 += w[i]**2
            column_var.append(weighted_sum_square_diff / (1 - sum_w2) )
    
    return np.array(column_mean), np.array(column_var)


# In[3]:


def pvt_svar(x, lambda_var = None, w = None):
    import numpy as np
    x = np.array(x)
    
    ## Determine the Lambda_var for correlation shrinkage intensity - 
    if not lambda_var:
        lambda_var = estimate_lambda_var(x)[0]
    else:
        if lambda_var < 0:
            lambda_var = 0
        if lambda_var > 1:
            lambda_var = 1
            
    ## Vk = empirical variances of column -
    v_k = wt_moments(x, w)[1]
    
    ## Compute Sample version of Vk_median -
    v_k_median = estimate_lambda_var(x)[1]
    
    ## Shrinkage t statistic estimation of Variance vector - 
    ## Vk* = Vs
    v_s = lambda_var * v_k_median + (1 - lambda_var) * v_k
    
    #print('Lambda_Var estimated or used : ', lambda_var, '\n')
    print('Estimating optimal shrinkage intensity Lambda_Var (variance vector) :', lambda_var, '\n')
    
    return v_s
    

# In[4]:


def wt_scale(x, w = None):
    ## n: sample size(rows) , p: number of variables(columns) -
    import numpy as np
    x = np.array(x)
    
    ## Compute column mean and column variance - 
    column_mean, column_var = wt_moments(x)
    
    for j in range(x.shape[1]):
        x[:, j] = ( x[:, j] - column_mean[j] ) / (column_var[j] ** 0.5 )
        
    return x


# In[5]:


def R_sweep(x, axis, stats, func):
    ## Only works for 2D array
    
    import numpy as np
    import pandas as pd
    X = x.copy()
    X = pd.DataFrame(X)
    
    if axis == 0:
        ## Check len(stats) == Sameple size
        if len(stats) != X.shape[0]:
            print("Length of stats array != Sample size !!!")
            return
    if axis == 1:
        ## Check len(stats) == Number of Featues
        if len(stats) != X.shape[1]:
            print("Length of stats array != Number of Features !!!")
            return 
        
    stats = pd.Series(stats)
    
    ## Only works for +, -, *
    if func == '*':
        X = X.apply(lambda row: row * stats, axis = axis)
    if func == '+':
        X = X.apply(lambda row: row + stats, axis = axis)
    if func == '-':
        X = X.apply(lambda row: row - stats, axis = axis)
        
    return X.values


# In[6]:


def estimate_lambda(x, w = None):
    # Reference: A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and Implications for Functional Genomics 2005
    # p.4, 11-13, 26-27
    # https://github.com/cran/corpcor/blob/5df4d249ead2dbfe973c0d9edc5310a429e40337/R/shrink.intensity.R
    # Shrinkage estimation of the covariance matrix
    # Estimation of the variance and covariance of the components of the S and R matrix
    # R: estimate.lambda.var and estimate.lambda
    
    ## n: sample size(rows) , p: number of variables(columns) -
    import numpy as np
    X = x.copy()
    X = np.array(X)
    n = X.shape[0]
    p = X.shape[1]
    
    ## Determine the weight for each sample - 
    if not w:
        w = np.repeat(1/n, n, axis=0)
    else:
        if len(np.array(w)) != n:
            print("Length Error: Length of w != Sample size n.")
        else:
            if np.array(w).sum() != 1:
                print("Weighted sum Error: Length of w = Sample size n, But Sum of weights != 1.")
            else:
                w = np.array(w)
    
    if p == 1:
        lambd = 1
    if n < 3:
        print("Sample size too small !!!")
        
    XS = wt_scale(X)
    
    ## Bias correction factors
    w2 = (w * w).sum()   # for w = 1/n this equals 1/n   where n = XS.shape[0]
    h1w2 = w2 / (1 - w2)     # for w = 1/n this equals 1/(n-1)
    
    ## Direct slow algorithm
    sw = np.sqrt(w)
    xsw = R_sweep(XS, axis = 0, stats = sw, func = "*")
    xs2w = R_sweep(np.power(XS, 2), axis = 0, stats = sw, func = "*")
    
    xsw_crossprod = np.dot(xsw.transpose(), xsw)
    E2R = np.power(xsw_crossprod, 2)
    ER2 = np.dot(xs2w.transpose(), xs2w)
        
    ## Offdiagonal sums
    sE2R = E2R.sum() - np.diag(E2R).sum()
    sER2 = ER2.sum() - np.diag(ER2).sum()
    
    denominator = sE2R
    numerator = sER2 - sE2R
    
    if denominator == 0:
        lambd = 1
    else:
        lambd = max(0, min(1, (numerator/denominator) * h1w2))
        
    return lambd


# In[7]:


def positive_svd(x, tol = None):
    # svd that retains only positive singular values 
    # Reference: https://github.com/cran/corpcor/blob/5df4d249ead2dbfe973c0d9edc5310a429e40337/R/fast.svd.R
    
    import numpy as np
    X = x.copy()
    X = np.array(X)
    
    ## Output will correspond to R svd() - 
    ## X = UDV'
    S_U, S_D, S_V = np.linalg.svd(X, full_matrices=False)
    S_V_transpose = S_V.T
    
    if not tol:
        # 2.220446e-16 = R's .Machine$double.eps = R's min difference 
        tol = max(X.shape[0], X.shape[1]) * max(S_D) * 2.220446e-16
        
    Positive = S_D > tol
    D = S_D[Positive]
    U = S_U[:, Positive]
    V = S_V_transpose[:, Positive]
    
    return D, U, V


# In[8]:


def nsmall_svd(x, tol = None):
    # fast computation of svd(x) if n << p
    # (n are the rows, p are columns)
    # Reference: https://github.com/cran/corpcor/blob/5df4d249ead2dbfe973c0d9edc5310a429e40337/R/fast.svd.R
    
    import numpy as np
    X = x.copy()
    X = np.array(X)
    
    # B, nxn matrix
    B = np.dot(X, X.T)
    
    ## Output will correspond to R svd() - 
    ## X = UDV'
    S_U, S_D, __ = np.linalg.svd(B, full_matrices=False)
    
    ## Determine rank of B  (= rank of X) - 
    if not tol:
        # 2.220446e-16 = R's .Machine$double.eps = R's min difference 
        tol = B.shape[0] * max(S_D) * 2.220446e-16
        
    Positive = S_D > tol
    
    ## positive singular values of m - 
    D = np.power(S_D[Positive], 0.5)
    
    ## corresponding orthogonal basis vectors - 
    U = S_U[:, Positive]
    
    X_U_crossprod = np.dot(X.transpose(), U)
    I_D = np.eye(len(D))
    np.fill_diagonal(I_D, val = D)
    V = np.dot(X_U_crossprod, I_D)
    
    return D, U, V


# In[9]:


def psmall_svd(x, tol = None):
    # fast computation of svd(x) if n >> p  
    # (n are the rows, p are columns)
    # Reference: https://github.com/cran/corpcor/blob/5df4d249ead2dbfe973c0d9edc5310a429e40337/R/fast.svd.R
    
    import numpy as np
    X = x.copy()
    X = np.array(X)
    
    # B, pxp matrix
    B = np.dot(X.transpose(), X)
    
    ## Output will correspond to R svd() - 
    ## X = UDV'
    __, S_D, S_V = np.linalg.svd(B, full_matrices=False)
    S_V_transpose = S_V.T
    
    ## Determine rank of B  (= rank of X) - 
    if not tol:
        # 2.220446e-16 = R's .Machine$double.eps = R's min difference 
        tol = B.shape[0] * max(S_D) * 2.220446e-16
        
    Positive = S_D > tol
    
    ## positive singular values of m - 
    D = np.power(S_D[Positive], 0.5)
    
    ## corresponding orthogonal basis vectors - 
    V = S_V[:, Positive]
    
    X_V_dot = np.dot(X, V)
    I_D = np.eye(len(D))
    np.fill_diagonal(I_D, val = D)
    U = np.dot(X_V_dot, I_D)
    
    return D, U, V


# In[10]:


def fast_svd(x, tol = None):
    # fast computation of svd(x)
    # (n are the rows, p are columns)
    # note that also only positive singular values are returned
    # Reference: https://github.com/cran/corpcor/blob/5df4d249ead2dbfe973c0d9edc5310a429e40337/R/fast.svd.R
    # Output will correspond to R svd()
    # Output: D, U, V
    
    import numpy as np
    X = x.copy()
    X = np.array(X)
    
    n = X.shape[0]
    p = X.shape[1]
    
    # use standard SVD if matrix almost square
    EDGE_RATIO = 2
    
    if n > EDGE_RATIO * p:
        return psmall_svd(x, tol)
    elif EDGE_RATIO * n < p:
        return nsmall_svd(x, tol)
    
    # if p and n are approximately the same
    else:
        return positive_svd(x, tol)


# In[11]:


def R_mpower(x, alpha, pseudo = False, tol = None):
    # Reference: https://github.com/cran/corpcor/blob/5df4d249ead2dbfe973c0d9edc5310a429e40337/R/mpower.R
    # compute X^alpha where X is a symmetric matrix

    import numpy as np
    X = x.copy()
    X = np.array(X)
    
    ## Check whether X is symmetric matrix or not - 
    ## 2.220446e-16 = R's .Machine$double.eps = R's min difference 
    X_symmetric_check = X - X.T
    if np.sum(X_symmetric_check > 100 * 2.220446e-16) > 0:
        print("Input matrix is not symmetric!")
    
    ## eigen decompositon, descending order - 
    eigenValues, eigenVectors = np.linalg.eigh(X)
    idx = np.argsort(eigenValues)
    ordered_idx = np.sort(idx)[::-1]
    eigenValues = eigenValues[ordered_idx]
    eigenVectors = eigenVectors[:, ordered_idx]
    
    # set small eigenvalues to exactly zero - 
    if not tol:
        # 2.220446e-16 = R's .Machine$double.eps = R's min difference 
        tol = max(X.shape[0], X.shape[1]) * max(np.absolute(eigenValues)) * 2.220446e-16
        
    eigenValues[np.absolute(eigenValues) <= tol] = 0
    
    # use only the nonzero eigenvalues
    if pseudo:
        index = (eigenValues != 0)
    
    # use all eigenvalues
    else:
        index = np.array([i for i in range(len(eigenValues))])
        
    e2 = np.power(eigenValues[index], alpha)
    I_e2 = np.eye(len(e2))
    np.fill_diagonal(I_e2, val = e2)
    
    # tcrossprod(x, y) : x %*% t(y)
    ma = np.dot( eigenVectors[:, index], np.dot(I_e2, eigenVectors[:, index].transpose()) )
    
    return ma


# In[12]:


def pvt_powscor(x, alpha, lambd = None, w = None):
    # Reference: https://github.com/cran/corpcor/blob/master/R/pvt.powscor.R
    # R Non-public function for computing R_shrink^alpha
    
    import numpy as np
    ## n: sample size(rows) , p: number of variables(columns) -
    X = x.copy()
    X = np.array(X)
    
    XS = wt_scale(X)
    n = XS.shape[0]
    p = XS.shape[1]

    ## Determine the weight for each sample - 
    if not w:
        w = np.repeat(1/n, n, axis=0)
    else:
        if len(np.array(w)) != n:
            print("Length Error: Length of w != Sample size n.")
        else:
            if np.array(w).sum() != 1:
                print("Weighted sum Error: Length of w = Sample size n, But Sum of weights != 1.")
            else:
                w = np.array(w)
    
    ## Determine the correlation shrinkage intensity Lambda - 
    if not lambd:
        lambd = estimate_lambda(x)
    else:
        if lambd < 0:
            lambd = 0
        if lambd > 1:
            lambd = 1
        
    ## Bias correction factors
    w2 = (w * w).sum()    # for w = 1/n this equals 1/n   where n = XS.shape[0]
    h1 = 1 / (1 - w2)     # for w = 1/n this equals the usual h1 = n/(n-1)
    
    sw = np.sqrt(w)
    
    # result in both cases is the identity matrix
    if (lambd == 1 or alpha == 0):
        powr = np.eye(p)   # return identity matrix
        
    # # don't do SVD in this simple case
    elif alpha == 1:
        # unbiased empirical estimator
        # for w = 1/n  the following  would simplify to:  r = 1/(n-1)*crossprod(XS)
        # r0 = h1 * t(XS) %*% diag(w) %*% XS
        # r0 = h1 * t(XS) %*% sweep(XS, 1, w, "*")
        # r0 = h1 * crossprod( sweep(XS, 1, sqrt(w), "*") )
        xsw = R_sweep(XS, axis = 0, stats = sw, func = "*")
        xsw_crossprod = np.dot(xsw.transpose(), xsw)
        r0 = h1 * xsw_crossprod
        
        # shrink off-diagonal elements
        powr = (1 - lambd) * r0
        np.fill_diagonal(powr, val = 1)
        
    else:
        # number of zero-variance variables
        # zeros: logical index
        XS_column_std = np.power(wt_moments(XS)[1], 0.5)    # XS's column std
        zeros = (XS_column_std == 0.)
        
        svdxs = fast_svd(XS)
        m = len(svdxs[0])    # rank of XS
        
        # t(U) %*% diag(w) %*% U
        UTWU = np.dot( svdxs[1].transpose(), R_sweep(svdxs[1], axis = 0, stats = w, func = "*") )
        
        # D %*% UTWU %*% D
        C = R_sweep( R_sweep(UTWU, axis = 0, stats = svdxs[0], func = "*"), axis = 1, stats = svdxs[0], func = '*' )
        C = (1 - lambd) * h1 * C
        
        # symmetrize for numerical reasons (mpower() checks symmetry)
        # note: C is of size m x m, and diagonal if w = 1/n
        C = ( C + C.transpose() ) / 2
        
        # use eigenvalue decomposition computing the matrix power
        if lambd == 0.:
            if m < p - np.sum(zeros):
                # R: powr =  svdxs$v %*% tcrossprod( mpower(C, alpha), svdxs$v)
                powr = np.dot( svdxs[2], np.dot(R_mpower(C, alpha), svdxs[2].transpose()) ) 
                
        # use a special identity for computing the matrix power
        else:
            # R: F = diag(m) - mpower(C/lambda + diag(m), alpha)
            F = np.eye(m) - R_mpower( (C/lambd + np.eye(m)), alpha)
            
            # R: powr = (diag(p) - svdxs$v %*% tcrossprod(F, svdxs$v))*(lambda)^alpha 
            F_V_tcrossprod = np.dot(F, svdxs[2].transpose())            
            powr = ( np.eye(p) - np.dot(svdxs[2], F_V_tcrossprod) ) * (lambd ** alpha)

        
        # set all diagonal entries corresponding to zero-variance variables to 1 - 
        # R: diag(powr)[zeros] = 1
        np.fill_diagonal(powr[zeros], val = 1)
        
    print('Alpha for Eigenvalues**Alpha estimated or used: ', alpha, '\n')
    print('Estimating optimal shrinkage intensity Lambda (correlation matrix) :', lambd, '\n')
    #print('Lambda estimated or used: ', lambd, '\n')
        
    return powr


# In[13]:


def powcor_shrink(x, alpha, lambd = None, w = None):
    # Reference: https://github.com/cran/corpcor/blob/master/R/shrink.estimates.R
    # # power of the shrinkage correlation matrix
    
    import numpy as np
    ## n: sample size(rows) , p: number of variables(columns) -
    X = x.copy()
    X = np.array(X)
    
    XS = wt_scale(X)
    n = XS.shape[0]
    p = XS.shape[1]
    
    ## Determine the correlation shrinkage intensity Lambda - 
    if not lambd:
        lambd = estimate_lambda(x)
    else:
        if lambd < 0:
            lambd = 0
        if lambd > 1:
            lambd = 1
    
    ## matrix power of shrinkage correlation -
    powr = pvt_powscor(X, alpha, lambd, w)
    
    return powr


# In[14]:


def cor_shrink(x, alpha = 1, lambd = None, w = None):
    # Reference: https://github.com/cran/corpcor/blob/master/R/shrink.estimates.R
    # Correlation
    
    import numpy as np

    ## n: sample size(rows) , p: number of variables(columns) -
    X = x.copy()
    X = np.array(X)
    
    XS = wt_scale(X)
    n = XS.shape[0]
    p = XS.shape[1]
    
    Corr = powcor_shrink(x, alpha, lambd, w)
    
    return Corr


# In[15]:


def invcor_shrink(x, alpha = -1, lambd = None, w = None):
    # Reference: https://github.com/cran/corpcor/blob/master/R/shrink.estimates.R
    # Inverse Correlation
    
    import numpy as np

    ## n: sample size(rows) , p: number of variables(columns) -
    X = x.copy()
    X = np.array(X)
    
    XS = wt_scale(X)
    n = XS.shape[0]
    p = XS.shape[1]
    
    Corr = powcor_shrink(x, alpha, lambd, w)
    
    return Corr


# In[16]:


def var_shrink(x, lambda_var = None, w = None):
    # Reference: https://github.com/cran/corpcor/blob/master/R/shrink.estimates.R
    # Variances
    
    import numpy as np

    ## n: sample size(rows) , p: number of variables(columns) -
    X = x.copy()
    X = np.array(X)
    
    XS = wt_scale(X)
    n = XS.shape[0]
    p = XS.shape[1]
    
    # Shrinkage Variance
    SV = pvt_svar(x, lambda_var, w)
    
    return SV


# In[17]:


def cov_shrink(x, alpha = 1, lambd = None, lambda_var = None, w = None):
    # Reference: https://github.com/cran/corpcor/blob/master/R/shrink.estimates.R
    # Covariances, alpha = 1
    
    import numpy as np

    ## n: sample size(rows) , p: number of variables(columns) -
    X = x.copy()
    X = np.array(X)
    
    XS = wt_scale(X)
    n = XS.shape[0]
    p = XS.shape[1]    
    
    # Shrinkage scale factors
    # SC.shape = (p, )
    SC = np.power( pvt_svar(x, lambda_var, w), 0.5)
    
    # Shrinkage correlation
    # C.shape = (p, p)
    C = pvt_powscor(X, alpha, lambd, w)
    
    # Shrinkage covariance 
    # R: is.null(dim(C)) , which dim(C) = (p, ) for a sequence
    if np.ndim(C) < 2:
        # R: c.shape = ( C * SC * SC ).shape = (p, p)
        # Because in R, SC_SC = SC * SC = np.power(SC, 2) which shape still remain (p, )
        # and C * SC in R will produce a matrix.shape = (p, p). Each elements in matrix M_ij = C_ij * SC_i (multiply on each row)
        SC_SC = np.power(SC, 2)
        
        # c = R_sweep(C, axis = 0, stats = SC_SC, func = '*')
        c = C * SC_SC.reshape(len(SC_SC), -1)
        
    else:
        # R: c = sweep(sweep(c, 1, sc, "*"), 2, sc, "*")
        # R: sweep(c, 1, sc, "*") , which elements in matrix M_ij = C_ij * SC_i (multiply on each row)
        c = R_sweep( R_sweep(C, axis = 0, stats = SC, func = '*'), axis = 1, stats = SC, func = '*' )
        
    return c


# In[18]:


def invcov_shrink(x, alpha = -1, lambd = None, lambda_var = None, w = None):
    # Reference: https://github.com/cran/corpcor/blob/master/R/shrink.estimates.R
    # Precision matrix (Inverse Covariance), alpha = -1
    
    import numpy as np

    ## n: sample size(rows) , p: number of variables(columns) -
    X = x.copy()
    X = np.array(X)
    
    XS = wt_scale(X)
    n = XS.shape[0]
    p = XS.shape[1]    
    
    # Shrinkage scale factors
    # SC.shape = (p, )
    SC = np.power( pvt_svar(x, lambda_var, w), 0.5)
    
    # Inverse Shrinkage correlation
    # INVC.shape = (p, p)
    INVC = pvt_powscor(X, alpha, lambd, w)
    
    # Inverse Shrinkage covariance 
    # R: is.null(dim(INVC)) , which dim(INVC) = (p, ) for a sequence
    if np.ndim(INVC) < 2:
        # R: invc.shape = ( C / SC / SC ).shape = (p, p) , which is C / (SC * SC)
        # Because in R, SC_SC = SC * SC = np.power(SC, 2) which shape still remain (p, )
        # and C * SC in R will produce a matrix.shape = (p, p). Each elements in matrix M_ij = C_ij * SC_i (multiply on each row)
        SC_SC = np.power(SC, 2)
        
        # invc = R_sweep(INVC, axis = 0, stats = SC_SC, func = '/')
        invc = INVC / SC_SC
        
    else:
        # R: invc = sweep(sweep(invc, 1, 1/sc, "*"), 2, 1/sc, "*")
        # R: sweep(c, 1, 1/sc, "*") , which elements in matrix M_ij = C_ij * (1/SC_i) (multiply on each row)
        invc = R_sweep( R_sweep(INVC, axis = 0, stats = 1/SC, func = '*'), axis = 1, stats = 1/SC, func = '*' )
        
    return invc


"""
# Created by Guo Yuan Li (Jimmy), 2019-03-11.
    
## Check all functions
## Testting data: xdata.csv
## The following are usage:

df = pd.read_csv("xdata.csv", index_col=False)
df = df.loc[:, ~df.columns.str.match('Unnamed')]
x = df.values
x
x.shape   # (6, 10)

estimate_lambda_var(x)[0]

estimate_lambda_var(x)[1]

estimate_lambda_var(x)[2]

estimate_lambda_var(x)[3]

wt_moments(x)[0]

wt_moments(x)[1]

np.power(wt_moments(x)[1], 0.5)

pvt_svar(x)

wt_scale(x)

R_sweep(x, axis = 1, stats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], func = '*')

R_sweep(x, axis = 0, stats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], func = '*')

R_sweep(x, axis = 1, stats = [0, 1, 2, 3, 4, 5], func = '*')

R_sweep(x, axis = 0, stats = [0, 1, 2, 3, 4, 5], func = '*')

estimate_lambda(x)

positive_svd(x, tol = None)

nsmall_svd(x, tol = None)

psmall_svd(x, tol = None)

fast_svd(x, tol = None)

R_mpower(np.dot(x.T, x), alpha = 0.5)

pvt_powscor(x, lambd = 0.5, alpha = 1)

pvt_powscor(x, alpha = 1)

pvt_powscor(x, alpha = -1)

powcor_shrink(x, alpha = 1, lambd = 0.5)

powcor_shrink(x, alpha = 1)

powcor_shrink(x, alpha = -1)

cor_shrink(x)

invcor_shrink(x)

var_shrink(x)

cov_shrink(x)

invcov_shrink(x)
"""



