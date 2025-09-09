## implicit_to_explicit.py (Total = 6)
#-----------------------------------
"""
Functios to convert the "Implicit Equations" produced by SINDy-PI into explicit ODEs for derivatives:

-----------
Workflow:
-----------
1.  feature_names     <--  model.get_feature_names()
2.  coeffs            <--   model.coefficients()
3.  left_coeff        <--  identity matrix  (makes each row start with its own feature)
4.  eq_list, symbols  <--   generate_symbolic_equations_both_sides(...)
"""

# Import Necessary Library 
import re
import numpy as np
import sympy as sp


#----------------------------------------------------------------------------
## Utility 1: Token-level parsing helpers
#----------------------------------------------------------------------------
def parse_feature(feature: str):
    """
    Parses a raw feature string (e.g., 'x0x1x2_dot') into a tuple of tokens,
    preserving derivative suffixes like '_dot' or '_t'.

    This function helps reconstruct symbolic terms from raw strings (in SINDy-PI's
    feature library), where multiplication and derivatives are encoded as strings.

    Parameters
    ----------
    feature : str
        A feature string, e.g., 'x0x1x2_dot' or 'x1x2_t'.

    Returns
    -------
    tuple of str
        Parsed tokens, e.g., ('x0', 'x1', 'x2_dot')
    """
    # Find all variables that end with a derivative suffix (_dot or _t)
    matches = list(re.finditer(r"([a-zA-Z]\d*)(_dot|_t)", feature))

    # If no suffix matches, treat entire string as a plain product like 'x0x1'
    if not matches:  # Plain monomial like 'x0x1'
        return tuple(re.findall(r"[a-zA-Z]\d*", feature))

    # Initialize parsed tokens and last position pointer
    tokens, last = [], 0
    
     # Iterate over each match (e.g., x2_dot)
    for m in matches:
        # Get the prefix string before this matched derivative term
        prefix = feature[last:m.start()]
        if prefix:
            # Extract all variable tokens in the prefix (e.g., 'x0x1' ---> ['x0', 'x1'])
            tokens.extend(re.findall(r"[a-zA-Z]\d*", prefix))

        # Keep the matched variable and its suffix (e.g., 'x2_dot')
        base, deriv = m.groups()                   
        tokens.append(base+deriv)   
        # Move the pointer ahead
        last = m.end()

    # If there's any leftover string after the last derivative token, process it
    if last < len(feature):
        tokens.extend(re.findall(r"[a-zA-Z]\d*", feature[last:]))

    # Return all parsed tokens as a tuple
    return tuple(tokens)

# -------------------------------------------------------------------------
# Above works for variables like a to z, A to Z, each alphabet with numeric 
#.....indices as state variables can take any variable name.

# If the string contains no “_dot”/“_t”, it simply extracts every alphanumeric
#.....  variable block.

#----------------------------------------------------------------------------
## Utility 2: Build a vocabulary of unique tokens and a map  feature -> token list
#----------------------------------------------------------------------------
def extract_distinct_features(feature_names):
    """
    Extracts a sorted list of all unique variables and derivative tokens
    used in the given raw feature strings, and builds a token map for each.

    Parameters
    ----------
    feature_names:list[str]
        Raw feature strings from model.get_feature_names(), e.g. ['x0', 'x0x1_dot'].

    Returns
    -------
    unique_tokens:list[str]
        Sorted list of all unique variable/derivative tokens seen in the feature names.

    feature_map:dict[str, tuple[str]]
        A mapping from each feature name to a tuple of its parsed tokens.
    """
    
    unique_tokens, feature_map = set(), {}
    # Iterate over each feature string
    for feat in feature_names:
        # Tokenize and store the map
        toks = parse_feature(feat)
        feature_map[feat] = toks
        # Update
        unique_tokens.update(toks)
        
    # Return sorted list of tokens and their map
    return sorted(unique_tokens), feature_map


#-----------------------------------------------------
## Utility 3:Convert a token list -> SymPy product
#------------------------------------------------------
def consolidate_product(tokens, token_symbols):
    """
    Converts a list of tokens like ('x0','x0','x1_dot') into a SymPy product:
    Parameters
    ----------
    tokens : tuple[str]
        A tuple of parsed variable names.
    
    token_symbols:dict[str, sp.Symbol]
        A dictionary mapping string tokens to SymPy symbolic variables.

    Returns
    -------
    sp.Expr
        A symbolic expression combining tokens into a single product.
    """
    # Count how many times each token appears
    token_counts = {}
    for token in tokens:
        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

    # Construct symbolic product 
    terms = []
    for token, count in token_counts.items():
        symbol = token_symbols[token]
        terms.append(symbol ** count)

    # Return the product expression 
    return sp.Mul(*terms, evaluate=False)

#--------------------------------------------------------------------------------
## Utility 4: Produce SymPy Eq objects for every row of an implicit SINDy model
#--------------------------------------------------------------------------------
def generate_symbolic_equations_both_sides(feature_names,
                                           left_coeff,
                                           right_coeff):
    """
    Converts raw coefficients and feature names from a SINDy-PI model into symbolic
    SymPy equations in the form: Eq(lhs - rhs, 0)

    Parameters
    ----------
    feature_names:list[str]
        Feature strings such as ['1', 'x0', 'x1', 'x0x1_dot'], etc.
        These come from model.get_feature_names().
    
    left_coeff:array-like (n x n)
        Left-hand-side coefficient matrix (often identity matrix),
        selects which feature appears on the LHS of each equation.
    
    right_coeff:array-like (n x n)
        Right-hand-side coefficient matrix from sparse regression.
        Defines which terms appear on the RHS of each equation.
        These come from model.coefficients().

    Returns
    -------
    eq_list:list[sympy.Eq]
        List of symbolic equations like Eq(lhs - rhs, 0), one per row.
    token_symbols:dict[str -> sympy.Symbol]
        Dictionary mapping each token string (e.g., 'x0_dot') to its symbolic object.
        Useful for later symbolic simplification or substitution.
    """
    # Parse feature names into tokens
    distinct, fmap = extract_distinct_features(feature_names)
    
    # Create a SymPy symbol for every unique token
    token_symbols  = {tok: sp.Symbol(tok) for tok in distinct}
    n = len(feature_names)

     # Convert lHS and RHS coefficient matrices to symbolic matrices
    L = sp.Matrix(left_coeff)
    R = sp.Matrix(right_coeff)

    # Build symbolic equations
    eq_list = []
    for i in range(n):
        # Construct the LHS & RHS of equation i 
        lhs = sum(L[i, j]*consolidate_product(fmap[feature_names[j]], token_symbols)
                  for j in range(n))
        rhs = sum(R[i, j] * consolidate_product(fmap[feature_names[j]], token_symbols)
                  for j in range(n))
        
        # Create equation: LHS - RHS = 0
        eq_list.append(sp.Eq(lhs - rhs, 0))             

    return eq_list, token_symbols

#---------------------------------------------------
## Utility 5: Produce an identity of proper size
#---------------------------------------------------
def generate_identity_matrix(features):
    """
    Convenience: np.eye(len(features)) so that
    each implicit equation starts with its "own" feature on the LHS.
    """
    return np.eye(len(features), dtype=int)
    
# --------------------------------------------
## Utility 6: Produce reformatted features
# --------------------------------------------
def get_reformatted_feature_names(feature_names):
    """
    Reformats feature strings from SINDy-PI into clean symbolic expressions.

    Parameters
    ----------
    feature_names : list[str]
        Raw feature strings such as ['x0', 'x0x1_dot', 'x0x1x1'] from SINDy-PI.

    Returns
    -------
    reformatted : list[str]
        List of human-readable symbolic expressions.
    Useful for labeling or displaying cleaned-up versions of
    terms discovered by the model.
    """
    distinct_tokens, feature_map = extract_distinct_features(feature_names)
    token_symbols = {token: sp.Symbol(token) for token in distinct_tokens}

    reformatted = []
    for feat in feature_names:
        tokens = feature_map[feat]
        sym_expr = consolidate_product(tokens, token_symbols)
        reformatted.append(str(sym_expr))
    return reformatted

#---------------------------------------------END-----------------------------------------------------