import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import scipy

import nltk
import re
import time
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
from matplotlib.ticker import MaxNLocator

from tqdm import tqdm

import fairnmf

def split_matrix(M, indices):
    return [M[idx,:] for idx in indices]

def filter_format(d):
    return {k: v for k, v in d.items() if k not in ['marker']}

def load_dataset(data_str):
    if data_str == "20n":
        # import the data
        dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
        stop_words_list = nltk.corpus.stopwords.words('english')

        # subsampling the data
        data_sub, _, data_sub_label, _ = train_test_split(dataset.data, dataset.target,
                                                            stratify=dataset.target, 
                                                            train_size=1500, random_state = 1)

        #Remove special characters
        for i in range(len(data_sub)):
            data_sub[i] = data_sub[i].replace("\n"," ")
            data_sub[i] = data_sub[i].replace("\t"," ")
            data_sub[i] = data_sub[i].lower()
            data_sub[i] = re.sub(r'[0-9]', '', data_sub[i])
            data_sub[i] = re.sub(r'[^\w\s]', '', data_sub[i])

        # Separate the data into groups with six major topics as described on http://qwone.com/~jason/20Newsgroups/
        # Only for counting and indexing the data
        Comp       = [data_sub[i] for i in range(len(data_sub)) if (data_sub_label[i] in (1,2,3,4,5))]
        Sale       = [data_sub[i] for i in range(len(data_sub)) if (data_sub_label[i] in (6,))]
        Recreation = [data_sub[i] for i in range(len(data_sub)) if (data_sub_label[i] in (7,8,9,10))]
        Poli       = [data_sub[i] for i in range(len(data_sub)) if (data_sub_label[i] in (16,17,18))]
        Religion   = [data_sub[i] for i in range(len(data_sub)) if (data_sub_label[i] in (0,15,19))]
        Sci        = [data_sub[i] for i in range(len(data_sub)) if (data_sub_label[i] in (11,12,13,14))]
        
        # Indices for the data
        m1 = len(Comp); m2 = len(Sale); m3 = len(Recreation); m4 = len(Poli); m5 = len(Religion); m6 = len(Sci)
        indices = [range(0, m1),
                   range(m1, m1+m2),
                   range(m1+m2, m1+m2+m3),
                   range(m1+m2+m3, m1+m2+m3+m4),
                   range(m1+m2+m3+m4, m1+m2+m3+m4+m5),
                   range(m1+m2+m3+m4+m5, m1+m2+m3+m4+m5+m6)]
        
        # Names for the data
        names = ["Computer", "Sale", "Recreation", "Politics", "Religion", "Science"]
        
        # Formatting of lines for plot
        formats = [{'marker': 'x', 'linestyle': '-',  'color': '#000000'},
                   {'marker': '*', 'linestyle': '--', 'color': '#0072B2'},
                   {'marker': '+', 'linestyle': ':',  'color': '#E69F00'},
                   {'marker': 'o', 'linestyle': '-',  'color': '#CC79A7'},
                   {'marker': 'v', 'linestyle': '--', 'color': '#009E73'},
                   {'marker': '^', 'linestyle': ':',  'color': '#F0E442'}]
        
        # TFIDF vectorize the data with pre-defined English stop words
        vectorizer = TfidfVectorizer(max_features=200, stop_words=stop_words_list)
        X_all = vectorizer.fit_transform(Comp+Sale+Recreation+Poli+Religion+Sci).toarray()
        
        X_all = X_all / np.linalg.norm(X_all, axis=0).reshape((1,-1))

        # Form the data matrix that will be used in fNMF
        Xs = split_matrix(X_all, indices)

        return {'Xs': Xs,
                'names': names,
                'formats': formats,
                'indices': indices,
                'ranks': range(1,21)}

    elif data_str == "hd":
        df = pd.read_csv("processed.cleveland.data", 
                 names=["age", "sex", "cp", "trestbps", "chol",
                         "fbs", "restecg", "thalach", "exang", "oldpeak",
                         "slope", "ca", "thal", "target"])
        df.replace(to_replace=["?"], value=np.nan, inplace=True)
        df = df.dropna()
        df = df.drop(columns=['target'])
        df = df.astype(float)
        
        X1 = df[df['sex'] == 1.0].drop(columns=['sex'])
        X2 = df[df['sex'] == 0.0].drop(columns=['sex'])
        
        X_all = np.concatenate((X1,X2), axis=0)
        
        X_all = X_all / np.linalg.norm(X_all, axis=0).reshape((1,-1))
        
        indices = [range(0,len(X1)), range(len(X1),len(X1)+len(X2))]
        names = ["female", "male"]
        formats = [{'marker': 'x', 'linestyle': '-',  'color': '#000000'},
                   {'marker': '*', 'linestyle': '--', 'color': '#0072B2'}]
        Xs = split_matrix(X_all, indices)

        return {'Xs': Xs,
                'names': names,
                'formats': formats,
                'indices': indices,
                'ranks': range(1, 11)}

    elif data_str == "ortho1":
        # Groups of various sizes and ranks
        m = 500
        d1 = 3; d2 = 6;
        n = 12
        
        rng = np.random.default_rng(seed=5109832)

        H1 = np.concatenate([np.eye(3), np.eye(3), np.zeros((3,6))],axis=1)
        H2 = np.eye(12)[-d2:,:]

        W1 = np.zeros((m,d1))
        W2 = np.zeros((m,d2))

        W1[np.arange(m),rng.integers(d1,size=(m,))] = 1
        W2[np.arange(m),rng.integers(d2,size=(m,))] = 1
        
        X1 = W1 @ H1 #low rank
        X2 = W2 @ H2 #high rank

        X_all = np.concatenate([X1, X2],axis=0)
        
        X_all = X_all / np.linalg.norm(X_all, axis=0).reshape((1,-1))
        
        # Indices for the data
        indices = [range(0, m),
                   range(m,2*m)]
        
        # Names for the data
        names = ["Low rank",
                 "High rank"]
        
        # Formatting of lines for plot
        formats = [{'marker': 'x', 'linestyle': '-',  'color': '#000000'},
                   {'marker': '*', 'linestyle': '--', 'color': '#0072B2'}]

        # Form the data matrix that will be used in fNMF
        Xs = split_matrix(X_all, indices)

        return {'Xs': Xs,
                'names': names,
                'formats': formats,
                'indices': indices,
                'ranks': range(1, 10)}

    elif data_str == "ortho2":
        # Groups of various sizes and ranks
        m = 500
        n = 12
        d = 3
        
        rng = np.random.default_rng(seed=413)

        H1 = np.concatenate([np.eye(3), np.eye(3), np.zeros((3,6))],axis=1)
        H3 = np.concatenate([np.zeros((3,6)), np.eye(3), np.eye(3)],axis=1)

        W1 = np.zeros((m,d))
        W2 = np.zeros((m,d))
        W3 = np.zeros((m,d))

        W1[np.arange(m),rng.integers(d,size=(m,))] = 1
        W2[np.arange(m),rng.integers(d,size=(m,))] = 1
        W3[np.arange(m),rng.integers(d,size=(m,))] = 1
        
        X1 = W1 @ H1 #clean aligned
        X2 = W2 @ H1 #noisy aligned
        X3 = W3 @ H3 #clean unaligned
        
        X2 += rng.normal(size=(m,n)) / 10
        X2[X2<0] = 0
        
        X_all = np.concatenate([X1, X2, X3],axis=0)
        
        X_all = X_all / np.linalg.norm(X_all, axis=0).reshape((1,-1))
        
        # Indices for the data
        indices = [range(0, m),
                   range(m,2*m),
                   range(2*m,3*m)]
        
        # Names for the data
        names = ["Group 1",
                 "Group 2",
                 "Group 3"]
        
        # Formatting of lines for plot
        formats = [{'marker': 'x', 'linestyle': '-',  'color': '#000000'},
                   {'marker': '*', 'linestyle': '--', 'color': '#0072B2'},
                   {'marker': '+', 'linestyle': ':',  'color': '#E69F00'}]

        # Form the data matrix that will be used in fNMF
        Xs = split_matrix(X_all, indices)

        return {'Xs': Xs,
                'names': names,
                'formats': formats,
                'indices': indices,
                'ranks': range(1, 7)}

    else:
        raise ValueError("Unknown dataset")
    
def exp_standard_NMF(data, num_trials=10, max_iter=None, rel_tol=1e-4):    
    Xs = data['Xs']
    ranks = data['ranks']
    indices = data['indices']

    X_all = np.concatenate(Xs)
    
    s_losses = np.zeros((len(Xs), len(ranks), num_trials))
    s_errors = np.zeros((len(Xs), len(ranks), num_trials))
    s_grperr = np.zeros((len(Xs), len(ranks), num_trials))

    for itr, (r_ind, r) in tqdm(itertools.product(range(num_trials), enumerate(ranks)),total=num_trials*len(ranks)):
        
        norms = np.array([np.linalg.norm(X, ord='fro') for X in Xs])
        min_errs = norms.copy()
        
        # NMF approximate of the optimal error
        for i in range(len(Xs)):
            opt_errs = []
            for approx_itr in range(5):
                W, H, _ = fairnmf.NMF(Xs[i], r, max_iter=max_iter, rel_tol=rel_tol)
                opt_errs.append(np.linalg.norm(Xs[i] - W@H, ord='fro'))
            min_errs[i] = np.nanmean(opt_errs)
            s_grperr[i,r_ind,itr] = np.linalg.norm(Xs[i] - W@H, ord='fro') / norms[i]
        
        # Find the vanilla NMF
        W, H, _ = fairnmf.NMF(X_all, r, max_iter=max_iter, rel_tol=rel_tol)
        
        # Compute and save the losses and reconstruction errors
        Xhats = split_matrix(W@H, indices)
        
        s_losses[:,r_ind,itr] = [(LA.norm(X - Xh, 'fro') - min_e)/n for X, Xh, min_e, n in zip(Xs, Xhats, min_errs, norms)]
        s_errors[:,r_ind,itr] = [LA.norm(X - Xh, 'fro') / n for X, Xh, n in zip(Xs, Xhats, norms)]

    return s_losses, 100*s_errors, 100*s_grperr

def exp_Fairer_NMF(data, alg, num_trials=10, max_iter=None, rel_tol=1e-4):
    if alg == 'mu':
        fnmf_alg = fairnmf.FairNMF_MU
    elif alg == "am":
        fnmf_alg = fairnmf.FairNMF_AM
    else:
        raise ValueError("Unknown algorithm")

    Xs = data['Xs']
    ranks = data['ranks']

    f_losses   = np.zeros((len(Xs), len(ranks), num_trials))
    f_errors   = np.zeros((len(Xs), len(ranks), num_trials))
    f_times    = np.zeros((len(ranks), num_trials))
    
    for itr, (r_ind, r) in tqdm(itertools.product(range(num_trials), enumerate(ranks)),total=num_trials*len(ranks)):
            
        t = time.time()
        Ws, H, errs, min_errs = fnmf_alg(Xs, r, max_iter=max_iter, rel_tol=rel_tol)
        f_times[r_ind, itr] = time.time()-t
        
        norms = np.array([np.linalg.norm(X, ord='fro') for X in Xs])            

        # Compute and save the losses and reconstruction errors
        Xhats = [W@H for W in Ws]
        
        f_losses[:,r_ind,itr] = [(LA.norm(X - Xh, 'fro') - min_e)/n for X, Xh, min_e, n in zip(Xs, Xhats, min_errs, norms)]
        f_errors[:,r_ind,itr] = [LA.norm(X - Xh, 'fro') / n for X, Xh, n in zip(Xs, Xhats, norms)]
    
    return f_losses, 100*f_errors, f_times

def exp_MU_convergence(data, rank, c_rule, num_trials=100, max_iter=1000, rel_tol=None):
    fnmf_alg = fairnmf.FairNMF_MU

    Xs = data['Xs']
    r = data['ranks'][-1]
    
    norms = np.array([np.linalg.norm(X, ord='fro') for X in Xs]).reshape((-1,1))

    f_losses   = np.zeros((len(Xs), max_iter, num_trials))
    f_errors   = np.zeros((len(Xs), max_iter, num_trials))

    for trial in tqdm(range(num_trials)):

        _, _, losses, min_errs = fnmf_alg(Xs, r, max_iter=max_iter, rel_tol=rel_tol, c_rule=c_rule)

        errs = (losses + min_errs.reshape((-1,1))) * norms

        f_losses[:,:,trial] = losses
        f_errors[:,:,trial] = errs

    return f_losses, f_errors

def make_plots(data, plot_vals, ylabel, file_names = None, xaxis='ranks', use_markers=True):
    plt.rcParams.update({'font.size': 15})

    Xs = data['Xs']
    formats = data['formats']
    names = data['names']

    if xaxis == 'ranks':
        x = data['ranks']
        xlabel = 'Rank'
    elif xaxis == 'iter':
        x = np.arange(plot_vals[0].shape[1])
        xlabel = 'Iteration'

    if file_names is None:
        file_names = [None] * len(plot_vals)

    if not use_markers:
        formats = [filter_format(f) for f in formats]
        
    ylim_all = None

    for pv in plot_vals:
        pv_avg = np.mean(pv, axis=2)
        pv_std = np.std(pv, axis=2)

        for i, f, n in zip(range(len(Xs)), formats, names):
            plt.plot(x, pv_avg[i,:], **f, label=n, linewidth='1.75')
            plt.fill_between(x,
                             pv_avg[i,:]-pv_std[i,:],
                             pv_avg[i,:]+pv_std[i,:],
                             **filter_format(f), alpha=0.2)
    
        ylim = plt.gca().get_ylim()
        if ylim_all == None:
            ylim_all = ylim
        else:
            ylim_all = [min(ylim_all[0],ylim[0]),max(ylim_all[1],ylim[1])]
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize=13)
        
        plt.clf()

    for pv, fn in zip(plot_vals, file_names):
        
        pv_avg = np.mean(pv, axis=2)
        pv_std = np.std(pv, axis=2)
        for i, f, n in zip(range(len(Xs)), formats, names):
            plt.plot(x, pv_avg[i,:], **f, label=n, linewidth='1.75')
            plt.fill_between(x,
                             pv_avg[i,:]-pv_std[i,:],
                             pv_avg[i,:]+pv_std[i,:],
                             **filter_format(f), alpha=0.2)
    
        plt.gca().set_ylim(ylim_all)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize=13)

        if not(fn is None):
            plt.savefig(fn, bbox_inches='tight', dpi=600)
        
        plt.show()