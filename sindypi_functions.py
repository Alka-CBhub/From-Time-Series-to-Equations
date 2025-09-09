## sindypi_functions.py (Total = 4)
#----------------------------------

# Import Necessary Library 
import numpy as np
import pandas as pd
import pysindy as ps
import sympy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt
from sympy import symbols, Eq, solve, Matrix
import plotly
import plotly.express as px
from IPython.display import display
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers.sindy_pi import SINDyPI
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm
# from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

###############################################################
# Import helpers
from implicit_to_explicit import*
from plot_utils import set_plot_style, set_spines_black

# Apply global style 
set_plot_style(dpi=300)



#----------------------------------------------------------------------------
## Utility 1: Tuning the hyperparameter- Threshold
#---------------------------------------------------------------------------

def optimize_threshold(
    library, # Feature matrix
    x_train, # time series data matrix
    t,  # time points
    threshold_values, # grid of thresholds
    feature_names, # name of the state variables
    tol: float = 1e-8, # hyperparameter for cvxpy
    max_iter: int = 20000, # hyperparameter for cvxpy
    show_progress: bool = True,
):
    """
    Serial grid-search over thresholds, returning:
      – mses:       list of MSEs.
      – rel_errs:   list of relative errors.
      – nonzeros:   list of number of active terms.
      – aics:       list of AIC values.
      – bics:       list of BIC values.
    If show_progress=True, wraps in a tqdm bar.
    """

    n = x_train.shape[0] # Number of sample points (time points in this case)
    eps = 1e-12  # to prevent log(0) while calculating ICs
    # Initialize storage
    mses, rel_errs, nonzeros, aics, bics = [], [], [], [], []

    iterator = tqdm(threshold_values, desc="Threshold grid search") if show_progress else threshold_values

    for lam in iterator:
        # Fit SINDy-PI
        opt = SINDyPI(threshold=lam, tol=tol, thresholder="l1", max_iter=max_iter)
        model = ps.SINDy(
            optimizer=opt,
            feature_library=library,
            differentiation_method=FiniteDifference(drop_endpoints=True),
            feature_names=feature_names,
        ).fit(x_train, t=t)

         # Metrics
        Y = library.transform(x_train) # LHS = Theta(X, dot(X_k))
        Yhat = model.predict(x_train) # LHS predicted

        ## Remark: We are analyzing the Global Performance:
        # MSE
        mse = np.mean((Y - Yhat) ** 2)
        mses.append(mse)

        # Relative error
        normY = np.linalg.norm(Y)
        rel = np.linalg.norm(Y - Yhat)/(normY + eps)
        rel_errs.append(rel)

        # Sparsity
        k = np.sum(np.abs(model.coefficients()) > 1e-6)
        nonzeros.append(k)

        # RSS 
        rss = max(eps, np.sum((Y - Yhat) ** 2))
        aics.append(2 * k + n * np.log(rss / n))
        bics.append(np.log(n) * k + n * np.log(rss / n))

    # Best threshold (based on minimum MSE)
    best_idx = np.argmin(mses) # Change meses to other as per the requirement.
    best_threshold = threshold_values[best_idx]
    return mses, rel_errs, nonzeros, aics, bics, best_threshold


   
       
#----------------------------------------------------------------------------
## Utility 2: Plot the results of the previous step
#---------------------------------------------------------------------------
def plot_threshold_diagnostics(
    thresholds, mses, rel_errors, aics, bics, nonzero_terms, dpi=None
):
    """
    Generates and displays four separate diagnostic plots:
    1. MSE vs Threshold
    2. Relative Error vs Threshold
    3. AIC & BIC vs Threshold
    4. MSE vs Sparsity ## Optional, as This has not been used in analysis

    Parameters:
    -----------
    thresholds : array‐like
        The lambda values used for each candidate model.
    mses : array‐like
        Mean squared errors corresponding to each threshold.
    rel_errors : array‐like
        Relative errors corresponding to each threshold.
    aics : array‐like
        AIC values for each threshold.
    bics : array‐like
        BIC values for each threshold.
    nonzero_terms : array‐like
        Number of active terms (sparsity) for each threshold.
    dpi : int or None, optional
        Resolution (dots per inch) for each figure. If None, uses the global rcParams setting.
    """

    best_mse_idx = np.nanargmin(mses)
    best_rel_idx = np.nanargmin(rel_errors)
    best_aic_idx = np.nanargmin(aics)
    best_bic_idx = np.nanargmin(bics)

    # --- (A) MSE vs Threshold ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    ax.semilogx(thresholds, mses, 'o-', markeredgecolor='k', label="MSE")
    ax.scatter(
        thresholds[best_mse_idx],
        mses[best_mse_idx],
        color='red',
        s=100,
        label="Best MSE"
    )
    ax.set_xlabel(r"Threshold ($\lambda$)")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True)
    set_spines_black(ax)
    plt.tight_layout()
    plt.show()

    # --- (B) Relative Error vs Threshold ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    ax.semilogx(thresholds, rel_errors, 's-', markeredgecolor='k', label="Relative Error")
    ax.scatter(
        thresholds[best_rel_idx],
        rel_errors[best_rel_idx],
        color='red',
        s=100,
        label="Best Relative Error"
    )
    ax.set_xlabel(r"Threshold ($\lambda$)")
    ax.set_ylabel("Relative Error")
    ax.legend()
    ax.grid(True)
    set_spines_black(ax)
    plt.tight_layout()
    plt.show()

    # --- (C) AIC & BIC vs Threshold ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    ax.semilogx(thresholds, aics, 'd-', markeredgecolor='k', label='AIC')
    ax.semilogx(thresholds, bics, '^-', markeredgecolor='k', label='BIC')
    ax.scatter(
        thresholds[best_aic_idx],
        aics[best_aic_idx],
        color='red',
        s=100,
        marker='o',
        label="Best AIC"
    )
    ax.scatter(
        thresholds[best_bic_idx],
        bics[best_bic_idx],
        color='green',
        s=100,
        marker='x',
        label="Best BIC"
    )
    ax.set_xlabel(r"Threshold ($\lambda$)")
    ax.set_ylabel("Information Criterion")
    ax.legend()
    ax.grid(True)
    set_spines_black(ax)
    plt.tight_layout()
    plt.show()

    # --- (D) MSE vs Sparsity ---
    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    sc = ax.scatter(
        nonzero_terms,
        mses,
        c=thresholds,
        cmap='viridis',
        edgecolor='k',
        s=60
    )
    ax.set_xlabel(r"Number of Active Terms ($|\xi_{ij}| > 10^{-6}$)")
    ax.set_ylabel("MSE")
    cb = plt.colorbar(sc)
    cb.set_label(r"Threshold ($\lambda$)")
    ax.grid(True)
    set_spines_black(ax)
    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------------
## Utility 3: Populate the results of the previous step in a table 
#----------------------------------------------------------------------------
def table_best_thresholds(thresholds, mses, rel_errors, aics, bics):
    """
    Table for the best threshold based on different criteria:
    MSE, Relative Error, AIC, BIC.
    """
    best_mse_idx = np.argmin(mses)
    best_rel_idx = np.argmin(rel_errors)
    best_aic_idx = np.argmin(aics)
    best_bic_idx = np.argmin(bics)

    print("\n=================================================")
    print("Best Thresholds for each metric:")
    print("==================================================")
    print(f"Best Threshold (based on MSE)         : {thresholds[best_mse_idx]:.4g}")
    print(f"Best Threshold (based on Relative Error) : {thresholds[best_rel_idx]:.4g}")
    print(f"Best Threshold (based on AIC)         : {thresholds[best_aic_idx]:.4g}")
    print(f"Best Threshold (based on BIC)         : {thresholds[best_bic_idx]:.4g}")
    print("==================================================\n")



#-----------------------------------------------------------------------------------------------------
## Utility 4: Fit the model for the best threshold and then, Populate the results for each equation. 
#-----------------------------------------------------------------------------------------------------
def evaluate_model_per_equation(
    library,          # Feature matrix
    x_train,          # Training data 
    t,                # Time points
    best_threshold,   # Selected threshold
    feature_names,    # Name of the state variables
    tol=1e-8,         # Hyperparameter for cvxpy
    max_iter=20000    # Hyperparameter for cvxpy
):
    """
    Fit a SINDy-PI model at a single threshold, then compute and print
    per‐model sorted tables:
      * By R square
      * By RelError
      * By AIC
      * By BIC
      * By Active Terms (k_i)
    """
    # 1) Fit SINDy-PI
    optimizer = ps.SINDyPI(
        threshold=best_threshold,
        tol=tol,
        thresholder="l1",
        max_iter=max_iter
    )
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=library,
        differentiation_method=FiniteDifference(drop_endpoints=True),
        feature_names=feature_names
    ).fit(x_train, t=t)

    # 2) Build theta and predict
    features = model.get_feature_names()
    features_r = get_reformatted_feature_names(model.get_feature_names())
    coefficients = model.coefficients()
    equations = model.equations()
    Y = library.transform(x_train) 
    Yhat = model.predict(x_train)
    n = x_train.shape[0]
    eps = 1e-12

    # 3) Compute metrics
    per_R2  = []
    per_rel = []
    per_AIC = []
    per_BIC = []
    per_k   = []

    for eq_idx in range(Y.shape[1]):
        y = Y[:, eq_idx]
        yhat = Yhat[:, eq_idx]
        res = y - yhat

        r2 = r2_score(y, yhat)
        rel = np.linalg.norm(res) / (np.linalg.norm(y) + 1e-12)
        k_i = np.count_nonzero(np.abs(coefficients[eq_idx, :]) > 1e-6)
        # k_i = np.count_nonzero(coefficients[eq_idx, :] )
        rss = max(eps, np.sum(res**2))
        aic_i = 2*k_i + n*np.log(rss/n)
        bic_i = np.log(n)*k_i + n*np.log(rss/n)

        per_R2.append(r2)
        per_rel.append(rel)
        per_AIC.append(aic_i)
        per_BIC.append(bic_i)
        per_k.append(k_i)

    # 4) Build results table
    results = pd.DataFrame({
        'Feature': features_r,
        # 'Equation': equations,
        'R Square': per_R2,
        'Relative Error': per_rel,
        'AIC': per_AIC,
        'BIC': per_BIC,
        'Active Terms (Implicit Equation)': per_k
    })
    results.index.name = 'Eq'
    results.reset_index(inplace=True)

    # 5) Create sorted tables
    results_r2  = results.sort_values('R Square', ascending=False)
    results_rel = results.sort_values('Relative Error', ascending=True)
    results_aic = results.sort_values('AIC', ascending=True)
    results_bic = results.sort_values('BIC', ascending=True)
    results_k   = results.sort_values('Active Terms (Implicit Equation)', ascending=True)

    # 6) Define table formatting
    formatter = {
        'R Square': "{:.4f}".format,
        'Relative Error': "{:.4e}".format,
        'AIC': "{:.4f}".format,
        'BIC': "{:.4f}".format,
        'Active Terms (Implicit Equation)': "{:d}".format
    }

    # 7) Highlight best
    def highlight_best(s, best="max"):
        if best == "max":
            best_val = s.max()
        elif best == "min":
            best_val = s.min()
        else:
            raise ValueError("best must be 'max' or 'min'")
        return ['background-color: lightgreen' if val == best_val else '' for val in s]

    # 8) Display tables
    def display_with_highlighted_row(title, df, sort_col, ascending, formatter, best_type):
        print(f"\n{title} (lambda={best_threshold}):\n" + "="*70)
        styled = (
            df.sort_values(sort_col, ascending=ascending)
              .style
              .apply(highlight_best, best=best_type, subset=[sort_col], axis=0)
              .format(formatter)
        )
        display(styled)

    # display_with_highlighted_row("Models sorted by highest R Square", results, 'R Square', False, formatter, best_type="max")
    display_with_highlighted_row("Models sorted by lowest Relative Error", results, 'Relative Error', True, formatter, best_type="min")
    # display_with_highlighted_row("Models sorted by lowest AIC", results, 'AIC', True, formatter, best_type="min")
    # display_with_highlighted_row("Models sorted by lowest BIC", results, 'BIC', True, formatter, best_type="min")
    # display_with_highlighted_row("Models sorted by lowest Active Terms in Implicit Equations", results, 'Active Terms (Implicit Equation)', True, formatter, best_type="min")

    # 9) Interactive plot
    results_r2 = results.sort_values('Relative Error', ascending=True).copy()
    results_r2["Equation"] = equations
    fig = px.scatter(
        results_r2, 
        x='Eq', y='Relative Error', 
        color='Relative Error',
        color_continuous_scale='Viridis',
        hover_data={
            'Feature': True,
            'R Square': ':.4e',
            'AIC': ':.4f',
            'BIC': ':.4f',
            'Active Terms (Implicit Equation)': True,
            'Equation': True
        },
        title=f"Relative Error for Each Implicit Equation (Threshold={best_threshold})",
        labels={'Eq': 'Model Index'}
    )
    fig.add_hline(
        y=results_r2['Relative Error'].mean(), 
        line_dash='dot',
        annotation_text=f"Average Relative Error = {results_r2['Relative Error'].mean():.3f}",
        annotation_position='bottom right'
    )
    fig.show()

    return features, coefficients, results

#---------------------------------------------END-----------------------------------------------------