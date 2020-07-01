
LR = 0.001  # Learning Rate for Optimizer
WD = 0.0001  # Weight decay value for Optimizer
GTSRB_CLASSES = 43
DISPLAY_BATCH = 100  # Display every count

SAVE_PATH = "./saved_models/teacher_saved.pth"

'''
HYPERPARAMETERS
'''
ALPHA = 0.9  # Alpha hyperparam for KD Loss
T = 20  # Temperature hyperparam for KD Loss
TEACHER_EPOCHS = 501  # Total Epoch Count
STUDENT_EPOCHS = 200
K = 64;  # Growth rate of dense networks
