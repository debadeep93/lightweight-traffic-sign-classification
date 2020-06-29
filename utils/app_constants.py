
LR = 0.001  # Learning Rate for Optimizer
WD = 0.0001  # Weight decay value for Optimizer
CIFAR_CLASSES = 10
GTSRB_CLASSES = 43
DISPLAY_BATCH = 50  # Display every count
GTSRB_DATASET = 1
CIFAR_10_DATASET = 2

SAVE_N = 10 
SAVE_PATH = "./saved_models/teacher_saved.pth"

'''
HYPERPARAMETERS
'''
ALPHA = 0.9  # Alpha hyperparam for KD Loss
T = 20  # Temperature hyperparam for KD Loss
EPOCHS = 300  # Total Epoch Count
K = 128;  # Growth rate of dense networks
