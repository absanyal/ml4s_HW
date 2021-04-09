# ml4s.py
# useful scripts and utilities for our course

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from viznet import connecta2a, node_sequence, NodeBrush, EdgeBrush, DynamicShow,theme

# --------------------------------------------------------------------------
def draw_feed_forward(ax, num_node_list, node_labels=None, weights=None,biases=None, zero_index=False, 
                     weight_thickness=False):
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
    
    shift = not zero_index

    # generate some default node labels
    if node_labels is None:
        node_labels = []
        for ℓ,nℓ in enumerate(num_node_list):
            if ℓ == 0:
                node_labels.append([f'$x_{j+shift}$' for j in range(nℓ)])
            else:
                node_labels.append([r'$a^{' + f'{ℓ}' + r'}_{' + f'{j+shift}' + r'}$' for j in range(nℓ)])
                
    # generate default bias labels
    if weights is None:
        weights = []
        for ℓ,nℓ in enumerate(num_node_list):
            if ℓ > 0:
                nℓm1 = num_node_list[ℓ-1]
                w_lab = np.zeros([nℓm1,nℓ],dtype='<U32')
                for k in range(nℓm1):
                    for j in range(nℓ):
                        w_lab[k,j] = r'$w^{' + f'{ℓ}' + r'}_{' + f'{k+shift}{j+shift}' + r'}$'
                weights.append(w_lab)
                    
    # generate some default weight labels
    if biases is None:
        biases = []
        for ℓ,nℓ in enumerate(num_node_list):
            if ℓ > 0:
                biases.append([r'$b^{' + f'{ℓ}' + r'}_{' + f'{j+shift}' + r'}$' for j in range(nℓ)])

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
                if isinstance(lab, np.floating) or isinstance(lab,float):
                    lab = f'{lab:.2f}'
                ax.text(x+0.05,y,lab,fontsize=6)
       
    eb = EdgeBrush('-->', ax,color='#58595b')
    ℓ = 0
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        
        if not weight_thickness:
            c = connecta2a(st, et, eb)
            if weights:

                w = weights[ℓ]

                if isinstance(w,np.ndarray):
                    w = w.flatten()

                for k,cc in enumerate(c):

                    factor = 1

                    # get the input and output neuron indices
                    idx = np.unravel_index(k,weights[ℓ].shape)
                    if idx[0]%2:
                        factor = -1

                    lab = w[k]
                    if isinstance(lab, np.floating) or isinstance(lab,float):
                        lab = f'{lab:.2f}' 

                    wtext = cc.text(lab,fontsize=6,text_offset=0.08*factor, position='top')
                    wtext.set_path_effects([path_effects.withSimplePatchShadow(offset=(0.5, -0.5),shadow_rgbFace='white', alpha=1)])
        
        else:
            # this is to plot individual edges with a thickness dependent on their weight
            # useful for convolutional networks where many weights are "zero"
            for i,cst in enumerate(st):
                for j,cet in enumerate(et):
                    if weights:
                        w = weights[ℓ]
                        
                        if np.abs(w[i,j]) > 1E-2:
                            eb = EdgeBrush('-->', ax,color='#58595b', lw=np.abs(w[i,j]))
                            e12 = eb >> (cst, cet)
                        
                            factor = 1
                            if i%2:
                                factor = -1
                            wtext = e12.text(f'{w[i,j]:.2f}',fontsize=6,text_offset=0.08*factor, position='top')
                            wtext.set_path_effects([path_effects.withSimplePatchShadow(offset=(0.5, -0.5),shadow_rgbFace='white', alpha=1)])

        ℓ += 1
        
def draw_network(num_node_list,node_labels=None,weights=None,biases=None,zero_index=False, weight_thickness=False):
    fig = plt.figure()
    ax = fig.gca()
    draw_feed_forward(ax, num_node_list=num_node_list, node_labels=node_labels,weights=weights, biases=biases, zero_index=zero_index, weight_thickness=weight_thickness)
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


# --------------------------------------------------------------------------
def feed_forward(aₒ,w,b,ffprime):
    '''Propagate an input vector x = aₒ through 
       a network with weights (w) and biases (b).
       Return: activations (a) and derivatives f'(z).'''
    
    a,df = [aₒ],[]
    for wℓ,bℓ in zip(w,b):
        zℓ = np.dot(a[-1],wℓ) + bℓ
        _a,_df = ffprime(zℓ)
        a.append(_a)
        df.append(_df)
        
    return a,df

# --------------------------------------------------------------------------
def backpropagation(y,a,w,b,df): 
    '''Inputs: results of a forward pass
       Targets     y: dim(y)  = batch_size ⨯ nL
       Activations a: dim(a)  = L ⨯ batch_size ⨯ nℓ
       Weights     w: dim(w)  = L-1 ⨯ nℓ₋₁ ⨯ nℓ
       Biases      b: dim(b)  = L-1 ⨯ nℓ
       f'(z)      df: dim(df) = L-1 ⨯ batch_size ⨯ nℓ
       
       Outputs: returns mini-batch averaged gradients of the cost function w.r.t. w and b
       dC_dw: dim(dC_dw) = dim(w)
       dC_db: dim(dC_db) = dim(b)
    '''
    
    num_layers = len(w)
    L = num_layers-1        
    batch_size = len(y)
    
    # initialize empty lists to store the derivatives of the cost functions
    dC_dw = [None]*num_layers
    dC_db = [None]*num_layers
    Δ = [None]*num_layers
    
    # perform the backpropagation
    for ℓ in reversed(range(num_layers)):
        
        # treat the last layer differently
        if ℓ == L:
            Δ[ℓ] = (a[ℓ] - y)*df[ℓ]
        else: 
            Δ[ℓ] = (Δ[ℓ+1] @ w[ℓ+1].T) * df[ℓ]
            
        dC_dw[ℓ] = (a[ℓ-1].T @ Δ[ℓ]) / batch_size
        dC_db[ℓ] = np.average(Δ[ℓ],axis=0)
        
    return dC_dw,dC_db

# --------------------------------------------------------------------------
def gradient_step(η,w,b,dC_dw,dC_db):
    '''Update the weights and biases as per gradient descent.'''
    
    for ℓ in range(len(w)):
        w[ℓ] -= η*dC_dw[ℓ]
        b[ℓ] -= η*dC_db[ℓ]
    return w,b

# --------------------------------------------------------------------------
def train_network(x,y,w,b,η,ffprime):
    '''Train a deep neural network via feed forward and back propagation.
       Inputs:
       Input         x: dim(x) = batch_size ⨯ n₁
       Target        y: dim(y) = batch_size ⨯ nL
       Weights       w: dim(w)  = L-1 ⨯ nℓ₋₁ ⨯ nℓ
       Biases        b: dim(b)  = L-1 ⨯ nℓ
       Learning rate η
       
       Outputs: the least squared cost between the network output and the targets.
       '''
    
    a,df = feed_forward(x,w,b,ffprime)
    
    # we pass a cycled a by 1 layer for ease of indexing
    dC_dw,dC_db = backpropagation(y,a[1:]+[a[0]],w,b,df)
    
    w,b = gradient_step(η,w,b,dC_dw,dC_db)
    
    return 0.5*np.average((y-a[-1])**2)

# --------------------------------------------------------------------------
def make_batch(n,batch_size,extent,func):
    '''Create a mini-batch from our inputs and outputs.
    Inputs:
    n0        : number of neurons in each layer
    batch_size: the desired number of samples in the mini-batch
    extent    : [min(xₒ),max(xₒ), min(x₁),max(x₁),…,min(x_{n[0]-1}),max(x_{n[0]-1})]
    func:     : the desired target function.
    
    Outputs: returns the desired mini-batch of inputs and targets.
    '''
    
    x = np.zeros([batch_size,n[0]])
    for i in range(n[0]):
        x[:,i] = np.random.uniform(low=extent[2*i],high=extent[2*i+1],size=[batch_size])

    y = func(*[x[:,j] for j in range(n[0])]).reshape(-1,n[-1])
    
    return x,y 