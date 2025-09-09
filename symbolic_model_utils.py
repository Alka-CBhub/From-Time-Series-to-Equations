## symbolic_model_utils.py (Total = 5)
#-------------------------------------

# Import Necessary Library 
import sympy as sp
import pandas as pd
from sympy.printing.latex import latex
from typing import Optional, List, Dict, Union, Tuple

#----------------------------------------------------------------------------
## Utility 1: Rational Simplification
#----------------------------------------------------------------------------
def rational_simplify(expression, expand_denominator=False):
    """
    Simplify a rational SymPy expression into its numerator and denominator.

    This function:
    1. Combines all terms into a single rational expression,
    2. Cancels common factors between the numerator and denominator,
    3. Separates the simplified expression into its numerator and denominator parts.

    Parameters:
    ----------
    expression : sympy.Expr
        The input symbolic expression to simplify.
    expand_denominator : bool, optional (default=False)
        If True, the denominator will be expanded for readability.

    Returns:
    -------
    numerator : sympy.Expr
        The simplified numerator.
    denominator : sympy.Expr
        The simplified denominator.
        (Both Numerator and Denominator seperate)
    """
    # Combine terms and cancel common factors
    combined = sp.cancel(sp.together(expression))  
    # Separate the combined expression into numerator and denominator
    numerator, denominator = combined.as_numer_denom() 
    # Expand the denominator for readability (Optional)
    if expand_denominator:
        denominator = sp.expand(denominator)
    return numerator, denominator


## For expanded version of a rational expression with a single expression return
def expand_rational_expr(expr):
    num, den = sp.fraction(expr)
    return sp.expand(num)/sp.expand(den)

#----------------------------------------------------------------------------
## Utility 2: Drop Small Terms
#----------------------------------------------------------------------------
def drop_small_terms(expr:sp.Expr, tol:float=1e-6)->sp.Expr:
    """
    Remove terms with small numeric coefficients from a rational SymPy expression.

    This utility is used for postprocessing symbolic expressions
    as very small coefficients may arise in a rational SymPy expression due to 
    numerical noise or overfitting. It retains only the dominant terms 
    (those above the given tolerance) in both numerator and denominator.

    Parameters
    ----------
    expr : sympy.Expr
        A rational SymPy expression.
    tol : float, optional (default=1e-6)
        Threshold below which numeric coefficients are considered negligible
        and the corresponding terms are dropped.

    Returns
    -------
    cleaned_expr : sympy.Expr
        A simplified rational expression with small terms removed.

    Remarks
    -----
    - If all terms are below the threshold, the largest term (by absolute value)
      is retained to avoid returning a zero numerator or denominator.
    - Final expression is returned in expanded form for consistency.
    """
    # Separate numerator and denominator
    num, den = expr.as_numer_denom()

    # Expand first
    num = sp.expand(num)
    den = sp.expand(den)

    def filter_terms(terms):
        """
        Filter out terms with small numeric coefficients.
        """
        new_terms = []
        for t in terms:
            coeff, _ = t.as_coeff_Mul()
            # Keep term only if coefficient is numeric, nonzero, and above threshold
            if coeff.is_number and not coeff.is_zero and abs(coeff.evalf()) > tol:
                new_terms.append(t)
        return new_terms
    # Apply filtering to numerator and denominator terms
    num_terms = filter_terms(num.as_ordered_terms())
    den_terms = filter_terms(den.as_ordered_terms())

    # Fallback: Retain largest term if all terms were removed
    if not num_terms and num.as_ordered_terms():
        num_terms = [max(num.as_ordered_terms(), key=lambda t: abs(t.as_coeff_Mul()[0].evalf()))]
    if not den_terms and den.as_ordered_terms():
        den_terms = [max(den.as_ordered_terms(), key=lambda t: abs(t.as_coeff_Mul()[0].evalf()))]
        
    # Reconstruct cleaned numerator and denominator
    num_clean = sp.Add(*num_terms)
    den_clean = sp.Add(*den_terms)
    
    # Avoid division by zero
    if den_clean.is_zero:
        return sp.zoo  # Represents undefined

    return expand_rational_expr(num_clean/den_clean)



#----------------------------------------------------------------------------
## Utility 3: Rescaling
#----------------------------------------------------------------------------
def rescale_expression(expr: sp.Expr,
                       *,
                       target: Optional[sp.Expr] = None,
                       gens: Optional[Union[List[sp.Symbol], tuple]] = None,
                       verbose: bool = False) -> sp.Expr:
    """
    Rescale a rational SymPy expression by dividing numerator and denominator
    by a chosen coefficient from the denominator.

    This makes the expression cleaner and easier to interpret.

    Parameters
    ----------
    expr : sympy.Expr
        The rational expression to rescale.
    target : sympy.Expr, optional
        Specific monomial from the denominator to use for rescaling.
        If not found or not provided, falls back to the leading term.
    gens : list of sympy.Symbol, optional
        Generator variables to define the polynomial structure of the denominator.
        If None, inferred from free symbols.
    verbose : bool, optional
        If True, prints diagnostic messages during rescaling.

    Returns
    -------
    rescaled_expr : sympy.Expr
        The rescaled rational expression with simplified coefficients.
    """
    expr = sp.together(expr)
    N, D = expr.as_numer_denom()
    # If denominator is constant, nothing to do
    if D.is_number:
        if verbose:
            print("[Skip] Denominator is constant.")
        return expr
    
    # Infer or use given variable generators
    gens = tuple(gens) if gens else tuple(sorted(D.free_symbols, key=lambda s: s.name))
    scaling_coeff = None
    used_monomial = None

    try:
        # Represent denominator as a polynomial
        poly = sp.Poly(D, *gens)

        # Option 1: Try user-specified target monomial
        if target is not None:
            coeff = poly.coeff_monomial(sp.expand(target))
            if (
                coeff.is_number and
                not coeff.has(sp.nan) and
                not coeff.is_zero and
                abs(coeff.evalf()) > 1e-12
            ):
                scaling_coeff = coeff
                used_monomial = target
                if verbose:
                    print(f"[Target Found] Using monomial {target} with coefficient {scaling_coeff}")
            else:
                if verbose:
                    print(f"[Target Invalid] Target {target} missing or zero. Falling back to leading term.")

        # Option 2: Fallback — use leading monomial by total degree (for the case when user specified monomial is absent)
        if scaling_coeff is None:
            monoms = poly.monoms()
            if not monoms:
                if verbose:
                    print("[Fail] No monomials in denominator.")
                return expr

            total_degrees = [sum(m) for m in monoms]
            max_deg = max(total_degrees)
            lead = sorted([m for m, d in zip(monoms, total_degrees) if d == max_deg])[0]
            leading_term = sp.prod(v**e for v, e in zip(poly.gens, lead))
            coeff = poly.coeff_monomial(leading_term)
            if (
                coeff.is_number and
                not coeff.has(sp.nan) and
                not coeff.is_zero and
                abs(coeff.evalf()) > 1e-12
            ):
                scaling_coeff = coeff
                used_monomial = leading_term
                if verbose:
                    print(f"[Leading Term] Using monomial {leading_term} with coefficient {scaling_coeff}")
            else:
                if verbose:
                    print("[Abort] Leading term coefficient is invalid.")
                return expr

    except Exception as e:
        if verbose:
            print(f"[Error] Poly failed: {e}")
        return expr

    # Final rescaling step 
    if verbose:
        print(f"[Success] Rescaling by coefficient {scaling_coeff}")
    rescaled_expr = (N/scaling_coeff)/(D/scaling_coeff)
    return expand_rational_expr(rescaled_expr)



#----------------------------------------------------------------------------
## Utility 4: Print Rational Model
#----------------------------------------------------------------------------
def print_rational_eq(i:int, var:sp.Symbol, expr:sp.Expr)->None:
    """
    Print a rational symbolic equation of the form:
        var = numerator/denominator
    It shows:
    - Equation index
    - Expanded numerator and denominator
    - Number of terms in numerator and denominator

    Parameters
    ----------
    i:int
        Index or ID of the model.
    var:sympy.Symbol
        The derivative in this case.
    expr:sympy.Expr
        The symbolic expression to print 
        (typically rational: numerator/denominator).

    Returns
    -------
    None
        Prints output to the console.
    """
    num, den = sp.fraction(expr)
    num, den = sp.expand(num), sp.expand(den)
    block = [
        "=" * 80,
        f"Model {i}:",
        " " * (len(str(var)) + 4) + str(num),
        f"{var} = " + "-" * max(len(str(num)), len(str(den))),
        " " * (len(str(var)) + 4) + str(den),
        f"[Terms: numerator={len(num.as_ordered_terms())}, denominator={len(den.as_ordered_terms())}]",
        "=" * 80
    ]
    print("\n".join(block))

#----------------------------------------------------------------------------
## Utility 5: Solve, Simplify, and Export Models
#----------------------------------------------------------------------------
def print_and_store_models(eq_list: List[sp.Eq],
                           xdot_symbol: sp.Symbol,
                           *,
                           target: Optional[sp.Expr] = None,
                           gens: Optional[Union[List[sp.Symbol], tuple]] = None,
                           sig_digits: int = 4,
                           tol: float = 1e-6,
                           latex_path: Optional[str] = None,
                           csv_path: Optional[str] = None
                          )->Tuple[Dict[int, sp.Expr], Dict[int, sp.Expr], Dict[int, sp.Expr], Dict[int, sp.Expr]]:
    """
    Process, simplify, and export symbolic models.
    This pipeline does the following:
    For each equation:
    - Solves for the specified derivative symbol (xdot_symbol),
    - Rescales the rational expression,
    - Drops terms with small coefficients,
    - Rounds numerical values for interpretability,
    - Prints a readable rational form,
    - Optionally exports LaTeX equations and CSV summary,
    - Stores each intermediate stage.

    Parameters
    ----------
    eq_list : List[sp.Eq]
        List of SymPy equations, typically of the form Eq(xdot, rhs_expr).
    xdot_symbol : sp.Symbol
        Symbol representing the derivative (LHS variable to solve for).
    target : sympy.Expr, optional
        Preferred monomial in the denominator to use for rescaling.
    gens : list of sympy.Symbol, optional
        List of generator variables for the polynomial denominator.
    sig_digits : int, optional (default=4)
        Number of significant digits for final rounding.
    tol : float, optional (default=1e-6)
        Threshold below which coefficients are considered negligible.
    latex_path : str, optional
        If provided, writes LaTeX equations to this file.
    csv_path : str, optional
        If provided, writes a CSV summary of each model.

    Returns
    -------
    Tuple of 4 dictionaries (by model index):
    - raw_models: raw rational expressions (expanded)
    - rescaled_models: after rescaling the denominator
    - cleaned_models: after dropping small terms
    - final_models: final expressions (rounded and simplified)
    """
    # Storage for all stages of models
    raw_models = {}
    rescaled_models = {}
    cleaned_models = {}
    final_models = {}

    # Counters
    failed_models = 0
    successful_models = 0
    
    rows_for_csv = []
    
    # Optional LaTeX file handle
    latex_file = open(latex_path, "w") if latex_path else None

    # Iterate through each symbolic equation
    for i, eq in enumerate(eq_list):
        # Solve the equation for the xdot symbol
        sols = sp.solve(eq, xdot_symbol)
        if not sols:
            print(f"Model {i}: No solution.")
            failed_models += 1
            continue

        raw = sols[0] # Use first solution 
        raw_expanded = sp.together(sp.expand(raw)) # Expand and unify as rational
        raw_models[i] = raw_expanded # Store raw expanded form

        # Skip if numerator is zero (This occurs when CVXPY solver fails)
        num, _ = sp.fraction(raw_expanded)
        if getattr(num, "is_zero", False):
            print(f"Model {i}: Solver failed (zero numerator).")
            failed_models += 1
            continue

        # Rescale denominator using leading or target term
        rescaled = rescale_expression(raw_expanded, target=target, gens=gens)
        rescaled_models[i] = rescaled   # store rescaled
    

        # Drop terms with small coefficients
        cleaned = drop_small_terms(rescaled, tol=tol)
        num_clean, den_clean = sp.fraction(cleaned)
        # Skip if all numerator terms are removed
        if getattr(num_clean, "is_zero", False):
            print(f"Model {i}: All numerator terms dropped (zero).")
            failed_models += 1
            continue

       # Round remaining terms to desired precision
        final_expr = expand_rational_expr(cleaned.evalf(sig_digits))

        # Store all surviving stages
        cleaned_models[i]  = cleaned
        final_models[i]    = final_expr
        successful_models += 1

        # Print final model in rational format
        print_rational_eq(i, xdot_symbol, final_expr)

        # Export to LaTeX (if file path is provided)
        if latex_file:
            latex_expr = sp.latex(sp.Eq(xdot_symbol, final_expr))
            latex_file.write(f"% Model {i}\n\\[\n{latex_expr}\n\\]\n\n")

        # Prepare CSV row 
        has_negative = any(
            term.as_coeff_Mul()[0] < 0
            for term in den_clean.as_ordered_terms()
            if term.as_coeff_Mul()[0].is_number
        )

        rows_for_csv.append({
            "Model": i,
            "Equation": f"{xdot_symbol} = {final_expr}",
            "Numerator Terms": len(num_clean.as_ordered_terms()),
            "Denominator Terms": len(den_clean.as_ordered_terms()),
            "Any Negative in Denominator": has_negative
        })
        
    # Finalize LaTeX file if used
    if latex_file:
        latex_file.close()
    # Finalize CSV file if used
    if csv_path:
        df = pd.DataFrame(rows_for_csv)
        df.to_csv(csv_path, index=False)
    # Summary report
    print("\n======================")
    print(f" Total models processed: {len(eq_list)}")
    print(f" Models obtained      : {successful_models}")
    print(f" Models failed        : {failed_models}")
    print("======================\n")
    # Return all collected models
    return raw_models, rescaled_models, cleaned_models, final_models

#---------------------------------------------END-----------------------------------------------------