## txt_data_frame.py (Total = 2)
#------------------------------
# Purpose: For Converting the text file to suitable data frame for sensitivity analysis (obtatined from COPASI)

# Import Necessary Library 
import re
import pandas as pd

#----------------------------------------------------------------------------
## Utility 1: Convert parameter names into LaTeX-style labels
#----------------------------------------------------------------------------

def read_block(lines, block_title):
    """
    Parses a sensitivity matrix block from COPASI-exported text files.
    Supports both single- and multi-variable output blocks.

    Parameters
    ----------
    lines : list of str
        Text file content split into lines.
    block_title : str
        The block name to locate (e.g., "Scaled sensitivities array").

    Returns
    -------
    pd.DataFrame
        Rows: output variables (or single dummy row)
        Columns: parameter names
        Values: sensitivities
    """

    # Regular expression to match parameter names like (R0).k1 or (R1).alpha
    param_pattern = re.compile(r"\(R\d+\)\.(\w+)")

    # Find the line where the block title appears
    header_index = lines.index(block_title)

    # Search for the first line after the header that contains parameter names
    for i in range(header_index + 1, len(lines)):
        if param_pattern.search(lines[i]):
            param_line_index = i
            break
    else:
        raise ValueError(f"Could not find parameter line under {block_title}")

    # Check if this block contains multiple output variables
    start = param_line_index + 1
    end = next((j for j in range(start, len(lines)) if not lines[j].strip()), len(lines))
    multi_variable = any("[" in line for line in lines[start:end])

    if multi_variable:
        # Case 1: Multiple output variables
        # Get parameter names from the parameter line
        param_names = param_pattern.findall(lines[param_line_index])

        data_dict = {}
        for line in lines[start:end]:
            # Match a line like: [VarName]   s1   s2  ...
            match = re.match(r"\[([^\]]+)\]\s+(.*)", line)
            if not match:
                continue
            var_name, raw_values = match.groups()
            values = [float(val) for val in raw_values.split()]
            data_dict[var_name] = dict(zip(param_names, values))

        return pd.DataFrame(data_dict).T[param_names]

    else:
        # Case 2: Single output variable
        param_values = {}
        for line in lines[param_line_index:]:
            if not line.strip():
                break
            parts = line.strip().split()
            if len(parts) == 2:
                param_match = param_pattern.match(parts[0])
                if param_match:
                    param_name = param_match.group(1)
                    param_values[param_name] = float(parts[1])

        return pd.DataFrame([param_values], index=["Var1"])


#---------------------------------------------END-----------------------------------------------------