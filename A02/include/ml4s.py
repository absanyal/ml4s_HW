# ml4s.py
# useful scripts and utilities for our course

import numpy as np
import matplotlib.pyplot as plt
from viznet import connecta2a, node_sequence, NodeBrush, EdgeBrush, DynamicShow,theme

# --------------------------------------------------------------------------
def draw_feed_forward(ax, num_node_list, node_labels=None, weights=None,biases=None):
    '''
    draw a feed forward neural network.

    Args:
        num_node_list (list<int>): number of nodes in each layer.
    '''
    num_hidden_layer = len(num_node_list) - 2
    token_list = ['\sigma^z'] + \
        ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
    kind_list = ['nn.input'] + ['nn.hidden'] * num_hidden_layer + ['nn.output']
    radius_list = [0.3] + [0.2] * num_hidden_layer + [0.3]
    y_list = 1.5 * np.arange(len(num_node_list))
    
    theme.NODE_THEME_DICT['nn.input'] = ["#E65933","circle","none"]
    theme.NODE_THEME_DICT['nn.hidden'] = ["#B9E1E2","circle","none"]
    theme.NODE_THEME_DICT['nn.output'] = ["#579584","circle","none"]

    seq_list = []
    for n, kind, radius, y in zip(num_node_list, kind_list, radius_list, y_list):
        b = NodeBrush(kind, ax)
        seq_list.append(node_sequence(b, n, center=(0, y)))
    
    # add labels
    if node_labels:
        for i,st in enumerate(seq_list):
            for j,node in enumerate(st):
                lab = node_labels[i][j]
                if isinstance(lab, float):
                    lab = f'{lab:.2f}'
                node.text(f'{lab}',fontsize=8)

    # add biases
    if biases:
        for i,st in enumerate(seq_list[1:]):
            for j,node in enumerate(st):
                x,y = node.pin(direction='right')
                lab = biases[i][j]
                if isinstance(lab, float):
                    lab = f'{lab:.2f}'
                ax.text(x+0.05,y,lab,fontsize=6)
       
    eb = EdgeBrush('-->', ax,color='#58595b')
    layer = 0
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        c = connecta2a(st, et, eb)
        if weights:
            w = weights[layer]
            if isinstance(w,np.ndarray):
                w = w.flatten()

            for k,cc in enumerate(c):
                factor = 1
                if k%2:
                    factor = -1
                    
                lab = w[k]
                if isinstance(lab, float):
                    lab = f'{lab:.2f}' 

                cc.text(lab,fontsize=6,text_offset=0.075*factor, position='top')
            
        layer += 1

def draw_network(num_node_list,node_labels=None,weights=None,biases=None):
    fig = plt.figure()
    ax = fig.gca()
    draw_feed_forward(ax, num_node_list=num_node_list, node_labels=node_labels,weights=weights, biases=biases)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.show()

# --------------------------------------------------------------------------
from IPython.core.display import HTML
def _set_css_style(css_file_path):
   """
   Read the custom CSS file and load it into Jupyter.
   Pass the file path to the CSS file.
   """

   styles = open(css_file_path, "r").read()
   s = '<style>%s</style>' % styles     
   return HTML(s)

# --------------------------------------------------------------------------
def get_linear_colors(cmap,num_colors,reverse=False):
    '''Return num_colors colors in hex from the colormap cmap.'''
    
    from matplotlib import cm
    from matplotlib import colors as mplcolors

    cmap = cm.get_cmap(cmap)

    colors_ = []
    for n in np.linspace(0,1.0,num_colors):
        colors_.append(mplcolors.to_hex(cmap(n)))

    if reverse:
        colors_ = colors_[::-1]
    return colors_

# --------------------------------------------------------------------------
def random_psd_matrix(size,seed=None):
    '''Return a random positive semi-definite matrix with unit norm.'''
    
    np.random.seed(seed)
    
    A = np.random.randn(*size)
    A = A.T @ A
    A = A.T @ A
    A = A / np.linalg.norm(A)
    return A