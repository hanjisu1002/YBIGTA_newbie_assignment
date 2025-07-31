from typing import Literal


device = "cpu"
d_model = 256

# Word2Vec
window_size = 7
method: Literal["cbow", "skipgram"] = "skipgram"
lr_word2vec = 1e-03
num_epochs_word2vec = 5

# GRU
hidden_size = 256
num_classes = 4

#lr = 8e-4 macro: 0.297166 | micro: 0.548077 epoch 150
#lr = 7e-4 macro: 0.280870 | micro: 0.567308
#lr = 1e-4 batch 32 macro: 0.310000 | micro: 0.615385
#lr = 1e-3 batch 16 macro: 0.363401 | micro: 0.563769
lr = 1e-3
num_epochs = 150
batch_size = 32