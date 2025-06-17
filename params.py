# Model parameters
N_LAYERS = 4
N_HEADS = 8
N_EMBD = 512

# Expert parameters
N_EXPERT = 8
N_INTER_SIZE = 4096
N_TOP_K = 2
SCORE_FN = 'softmax'
NUM_DENSE_LAYER = 1

# Training parameters
MAX_LEN = 128
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
LR = 1e-4
EPOCHS = 10
