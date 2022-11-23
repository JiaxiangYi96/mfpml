
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pdb

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF 
from mfpml.models.mf_surrogates import Kriging, HierarchicalKriging, ScaledKRG

def test_mfModels():
    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    y = np.squeeze(X * np.sin(X)) 

    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=3, replace=False)
    Xhf_train, Yhf_train = X[training_indices], y[training_indices] 

    training_indices = rng.choice(np.arange(y.size), size=9, replace=False)
    Xlf_train, Ylf_train = X[training_indices], y[training_indices]-4 

    # mfModel = HierarchicalKriging(kernel_mode='krg', n_dim=X.shape[1])
    mfModel = ScaledKRG(kernel_mode='krg', n_dim=X.shape[1])
    mfModel.train_whole(Xhf_train, Yhf_train, Xlf_train, Ylf_train) 

    Pre_LF, std_LF = mfModel.predict_lf(X, return_std=True)
    Pre_HF, std_HF = mfModel.predict(X, return_std=True)

    plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.plot(X, y-4, label=r"$f(x) = x \sin(x)-4$", linestyle="dotted")
    plt.scatter(Xhf_train, Yhf_train, label="HF Observations")
    plt.scatter(Xlf_train, Ylf_train, label="LF Observations")
    plt.plot(X, Pre_LF, label="LF prediction")
    plt.plot(X, Pre_HF, label="HF prediction")
    plt.fill_between(
        X.ravel(),
        (Pre_HF - 1.96 * std_HF).squeeze(),
        (Pre_HF + 1.96 * std_HF).squeeze(),
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("Gaussian process regression on noise-free dataset") 
    plt.show() 

def test_KRG():
    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    y = np.squeeze(X * np.sin(X)) 

    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=3, replace=False)
    X_train, Y_train = X[training_indices], y[training_indices] 

    model1 = Kriging(kernel_mode='krg', n_dim=X.shape[1]) 
    model1.train(X_train, Y_train) 
    
    mean_prediction, std_prediction = model1.predict(X, return_std=True) 

    flg = plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.scatter(X_train, Y_train, label="Observations")
    plt.plot(X, mean_prediction, label="Prediction")
    plt.fill_between(
        X.ravel(),
        (mean_prediction - 1.96 * std_prediction).squeeze(),
        (mean_prediction + 1.96 * std_prediction).squeeze(),
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("Gaussian process regression on noise-free dataset") 
    plt.show() 

if __name__ == "__main__":
    # test_KRG()
    test_mfModels()