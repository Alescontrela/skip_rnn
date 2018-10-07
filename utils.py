import numpy as np
f_binarize = np.round # f_binarize


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def initializeWeights(hidden_size, vocab_size, save = True, fileName = 'params.pkl'):
    if save:
        import _pickle as pickle
        params = pickle.load(open(fileName,'rb'))
        Wxh, Whh, Why, Wp = params['Wxh'], params['Whh'], params['Why'], params['Wp']
        bh, by, bp = params['bh'], params['by'], params['bp']
    else:
        Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
        Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
        Wp = np.random.randn(1, hidden_size)*0.01
        bh = np.zeros((hidden_size, 1)) # hidden bias
        by = np.zeros((vocab_size, 1)) # output bias
        bp = np.zeros((1,1))

    return Wxh, Whh, Why, Wp, bh, by, bp

def sample(h, seed_ix, n, params, vocab_size):
    # gather information on _planned_ skip steps. The sample() method
    # does not actually skip updates. This is simply used for visualization.
    us = {}
    us[0] = 1
    updateInfo = ""

    Wxh, Whh, Why, Wp, bh, by, bp = params
    x = np.zeros((vocab_size,1))
    x[seed_ix] = 1
    ixes = []

    for i in range(n):
        u = f_binarize(us[i]) # binarize update gate value
        if u == 1:
            updateInfo += "^"
        else:
            updateInfo += " "

        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y)/np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p = p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)

        # update the update gate
        delta_u = np.dot(Wp, h) + bp
        dus = sigmoid(delta_u)
        us[i+1] = u * dus + (1-u) * (us[i] + min(dus, 1 - us[i]))

    return ixes, updateInfo
