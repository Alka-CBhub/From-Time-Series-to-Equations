## network_utils.py (Total = 1)
#------------------------------
"""
Visualizing Jacobian interaction networks via Graphviz.
Requirements:
  - numpy
  - graphviz (Python bindings + Graphviz binaries on PATH)
"""
# Import Necessary Library
import numpy as np
import graphviz


#----------------------------------------------------------------------------
## Utility 1: Render and save the interaction network of a Jacobian matrix
#----------------------------------------------------------------------------
def draw_network(jacobian_matrix,
                 nodes,
                 output_path='network',
                 label_fontsize=14,
                 graph_size='8,8',
                 dpi=100,
                 engine='dot'):
    """
    Render and save the interaction network of a Jacobian matrix.

    Parameters
    ----------
    jacobian_matrix : numpy.ndarray, shape (n,n)
        Square Jacobian where entry [i,j] is (f_i)_x_j.
    nodes : list of str
        Length-n list of node names in the same order as the matrix axes.
    output_path : str
        Base filename (without extension) to save the image to.
    label_fontsize : int
        Font size for edge labels.
    graph_size : str
        Graphviz size attribute.
    dpi : int
        Resolution of the output image.
    engine : str
        Graphviz layout engine.

    Returns
    -------
    graphviz.Digraph
        The Digraph object (image is also saved).
    """

    # Basic checks 
    if jacobian_matrix.shape[0] != jacobian_matrix.shape[1]:
        raise ValueError("Jacobian matrix must be square.")
    if jacobian_matrix.shape[0] != len(nodes):
        raise ValueError("Length of 'nodes' must match Jacobian dimension.")

    # Initialise the graph
    dot = graphviz.Digraph(format='png')
    dot.engine = engine
    dot.attr(dpi=str(dpi))
    dot.attr(rankdir='LR', size=graph_size, ratio='compress')

    # Node and edge default styles
    dot.attr('node',
             shape='circle',
             style='filled,setlinewidth(2)',
             color='black',
             fillcolor='lightgoldenrodyellow',
             fontname='Helvetica',
             fontsize=str(label_fontsize))
    dot.attr('edge',
             fontname='Helvetica',
             fontsize=str(label_fontsize),
             penwidth='2')

    # Declare every node first 
    for name in nodes:
        dot.node(name)
    if len(nodes) == 2:
        dot.edge(nodes[0], nodes[1], style='invis')


    # Add edges based on the Jacobian
    n = jacobian_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            val = jacobian_matrix[i, j]
            if val == 0:
                continue  

            color, arrow = ('blue', 'normal') if val > 0 else ('red', 'tee')

            if i == j:
                # self-interaction
                dot.edge(nodes[i], nodes[i],
                         color=color,
                         arrowhead=arrow,
                         label=" ")
            else:
                # j --> i  (column j affects row i)
                dot.edge(nodes[j], nodes[i],
                         color=color,
                         arrowhead=arrow,
                         label=" ")

    # Render the Output
    dot.render(output_path, view=False)
    return dot

#---------------------------------------------END-----------------------------------------------------