import numpy as np

from utils import sigmoid

f_binarize = np.round # f_binarize

def lossFun(inputs, targets, hprev, params, vocab_size):
    Wxh, Whh, Why, Wp, bh, by, bp = params
    xs, hs, ys, ps, us, dus = {}, {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    us[0] = 1 # set first update gate to true. (perform weight update on first iteration)
    loss = 0
    skipSteps = 0

    # forward pass
    for t in range(len(inputs)):
        u = f_binarize(us[t]) # binarize update gate value
        if u:
            xs[t] = np.zeros((vocab_size,1)) # create one-hot
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # calculate hidden state
            ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities
            ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t])) # probabilities for next chars. softmax
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
        else:
            hs[t] = hs[t-1] # copy over previous state

        # update the update gate
        delta_u = np.dot(Wp, hs[t]) + bp
        dus[t] = sigmoid(delta_u)
        us[t+1] = u * dus[t] + (1-u) * (us[t] + min(dus[t], 1 - us[t]))

    # backward pass: compute gradients going backward
    dhs = {} # used for update gate gradient computation
    dWxh, dWhh, dWhy, dWp = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why), np.zeros_like(Wp)
    dbh, dby, dbp = np.zeros_like(bh), np.zeros_like(by), np.zeros_like(bp)
    dhnext = np.zeros_like(hs[0])
    dx = np.zeros_like(xs[0]) # for gradient wrt input (to be passed into previous layer's cell)

    for t in reversed(range(len(inputs))):
        u = f_binarize(us[t])

        if u:
            # print("iteration -- [update step]", t)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(Why.T, dy) + dhnext # backprop into h
            dhs[t] = dh # used for update gate gradient computation
            dhraw = (1-hs[t]*hs[t])*dh # backprop through tanh non-linearity
            dbh += dhraw
            dWxh  += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(Whh.T, dhraw)
            dx = np.dot(Wxh.T, dhraw)

            # update gate backpropagation
            if t+1 not in dhs.keys(): continue
            du = np.multiply(dhs[t+1], hs[t+1]) * dus[t]*(1-us[t]) # backprop into du
            dWp += np.multiply(du, hs[t]).T
            dbp += np.mean(du)
        else:
            skipSteps += 1
            # print("iteration -- [non-update step]", t)
            if t+1 not in dhs.keys(): continue
            dhs[t] = dhs[t+1]
            continue




    for dparam in [dWxh, dWhh, dWhy, dWp, dbh, dby, dbp, dx]:
        np.clip(dparam, -1.5, 1.5, out = dparam) # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dWp, dbh, dby, dbp, hs[len(inputs)-1], skipSteps
