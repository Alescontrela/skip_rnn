import pandas as pd
import numpy as np

from utils import initializeWeights, sample
from backprop import lossFun

# Parameters
verbose = True
create_new_data = False

saveWeights = True
saveSteps = 5000 # save weights every 5000 steps
loadWeights = False
weightsPath = 'params.pkl'


# Create a fresh dataset or open a preloaded one. (1min. vs ~5sec. amortized)
if verbose: print("LOADING DATASET...", end="")
if create_new_data:
    f = open("foodreviews.txt", "a")

    df_reviews = pd.read_csv('../Reviews.csv')

    display(df_reviews.columns)

    for index, row in df_reviews.iterrows():
        text = row["Text"]
        f.write(text)
        f.write(" ")

    f.flush()

    f.close()
else:
    data = open("foodreviews.txt", "r").read()
if verbose: print("DONE.\n")

chars = list(set(data)) # Find all unique characters
data_size, vocab_size = len(data), len(chars)

# create mappings from character to int and int to character.
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}

if verbose:
    print("data has %d characters, %d unique" % (data_size, vocab_size))
    print("-----------------------------------------")
    print("SAMPLE REVIEW:")
    print(data[8998:9489], end = "\n-----------------------------------------\n")

# hyperparamters
hidden_size = 100
seq_length = 50
learning_rate = 1e-1

Wxh, Whh, Why, Wp, bh, by, bp = initializeWeights(hidden_size, vocab_size, save = loadWeights, fileName = weightsPath)
params = [Wxh, Whh, Why, Wp, bh, by, bp]

n, p = 0, 0
# memory variables for Adagrad
mWxh, mWhh, mWhy, mWp = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why), np.zeros_like(Wp)
mbh, mby, mbp = np.zeros_like(bh), np.zeros_like(by), np.zeros_like(bp)

smooth_loss = -np.log(1.0/vocab_size)*seq_length

if verbose:
    print("TRAINING NETWORK")

while True:
    if p+seq_length + 1 >= len(data) or n==0:
        hprev = np.zeros((hidden_size, 1))
        p = 0
    totalSteps = 0
    totalSkips = 0

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]] # set up inputs to rnn
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]] # labels for each rnn output

    loss, dWxh, dWhh, dWhy, dWp, dbh, dby, dbp, hprev, skipSteps = lossFun(inputs, targets, hprev, params, vocab_size)
    smooth_loss = smooth_loss * 0.999 + 0.001 * loss

    totalSteps += len(inputs)
    totalSkips += skipSteps

    if n%1000 == 0:
        percentStepsSkipped = (totalSkips/totalSteps) * 100
        print("\tIteration: %d | Loss: %f | Percentage Steps Skipped: %.2f%%" % (n, smooth_loss, percentStepsSkipped))
        sample_ix, updateInfo = sample(hprev, inputs[0], 50, params, vocab_size)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print("\t|------Sampled Output:      {}".format(txt))
        print("\t|------Update Locations:    {}".format(updateInfo))

        # book-keeping
        totalSteps = 0
        totalSkips = 0

    if saveWeights:
        if n%saveSteps == 0:
            import _pickle as pickle
            pickle.dump(params, open(weightsPath, 'wb'))


    for param, dparam, mem in zip([Wxh, Whh, Why, Wp, bh, by, bp], [dWxh, dWhh, dWhy, dWp, dbh, dby, dbp], [mWxh, mWhh, mWhy, mWp, mbh, mby, mbp]):
        mem += dparam * dparam
        param += -learning_rate * dparam/np.sqrt(mem + 1e-8)

    p += seq_length
    n+=1
