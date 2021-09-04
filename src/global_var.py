import torch
BATCH_SIZE = 128
IMAGE_SIZE = 128
DECAY = 0.98
TRAIN_DIR = 'avia-train/avia-train/'
TEST_DIR = 'avia-test/avia-test/'
TEST_EXT_DIR = 'avia-test-extended/avia-test-extended/'
DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')