## steady_states_utils.py (Total = 2)
#-----------------------------------

# Import Necessary Library 
import numpy as np
from scipy.optimize import fsolve


#----------------------------------------------------------------------------
## Utility 1: Rounds values close to zero to exactly zero
#----------------------------------------------------------------------------
def round_very_small(val, zero_tol=1e-10):
    """Rounds values close to zero to exactly zero (supports scalars and arrays)."""
    if np.isscalar(val):
        return 0 if abs(val) < zero_tol else val
    else:
        return np.where(np.abs(val) < zero_tol, 0.0, val)


#----------------------------------------------------------------------------
## Utility 2: Find steady states in a given range
#----------------------------------------------------------------------------
def find_steady_states(system_func, num_vars, num_samples=500, tol=1e-8,
                       domain=None, zero_tol=1e-10, verbose=False):
    """
    Identify unique steady states of an n-dimensional dynamical system
    by solving f(X)=0 from multiple random initial guesses.

    Parameters
    ----------
    system_func:callable
        Function returning derivatives [dx1/dt, ..., dxn/dt]
        for input state [x1, ..., xn].

    num_vars:int
        Number of state variables in the system.

    num_samples:int, optional
        Number of random initial guesses to try. Default is 500.

    tol:float, optional
        Tolerance for determining uniqueness of solutions.
        Default is 1e-8.

    domain:list of tuple(float, float), optional
        Variable bounds as [(min1, max1), ..., (minN, maxN)].
        Default is (0, 1) for all.

    zero_tol:float, optional
        Threshold below which values are rounded to zero.
        Helps remove floating-point noise.

    verbose:bool, optional
        If True, prints info on accepted steady states and rejections.

    Returns
    -------
    roots:list of np.ndarray
        List of unique steady-state vectors satisfying f(X)=0 (Approx).
    """
    if domain is None:
        domain = [(0, 1)] * num_vars

    rng = np.random.default_rng()
    guesses = rng.uniform(
        low=[d[0] for d in domain],
        high=[d[1] for d in domain],
        size=(num_samples, num_vars)
    )
    
    roots = []

    for i, guess in enumerate(guesses):
        sol, info, ier, _ = fsolve(system_func, guess, full_output=True)

        if ier != 1:
            if verbose:
                print(f"Guess {i+1}: fsolve failed to converge.")
            continue

        # Round very small values to zero
        round_very_small(sol, zero_tol=zero_tol)

        # Check if solution lies within domain
        if not all(low <= val <= high for val, (low, high) in zip(sol, domain)):
            if verbose:
                print(f"Guess {i+1}: solution out of bounds {sol}.")
            continue

        # Check for uniqueness
        if any(np.allclose(sol, root, atol=tol) for root in roots):
            continue

        roots.append(sol)

        if verbose:
            print(f"Guess {i+1}: accepted steady state {np.round(sol, 4)}.")

    # Optional: Sort for reproducibility
    roots.sort(key=lambda x: tuple(np.round(x, 12)))
    return roots

#---------------------------------------------END-----------------------------------------------------