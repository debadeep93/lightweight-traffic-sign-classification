import torch
import torchvision
from torch import nn
from torch.autograd import Variable as var
import torchvision.transforms as transforms
from utils.gtsrb_loader import GTSRB

from utils.app_constants import ALPHA, T, K, GTSRB_DATASET, CIFAR_10_DATASET


# KD Loss implementation
def computeKDLoss(teacher_output, student_output,labels):
    '''
    Knowledge Distillation Loss Computation using
    Cross-Entropy between Student and Teacher labels
    :param teacher_output: output prediction of teacher
    :param student_output: output prediction of student
    :param labels: ground truth labels
    :return: loss value
    '''
    softmax = nn.Softmax2d()
    cec = nn.CrossEntropyLoss()

    loss = (1 - ALPHA) * cec(softmax(student_output, labels)) + 2 * T * T * ALPHA * cec(softmax(student_output / T), softmax(teacher_output / T))
    return loss;


def validate(model,data):
    '''
    # Model validation against test data
    To get validation accuracy = (correct/total)*100.

    :param model: model to evaluate
    :param data: data to evaluate on
    :return:
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    #model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i,(images,labels) in enumerate(data):
            images = var(images.to(device))
            x = model.forward(images)
            value,pred = torch.max(x,1)
            pred = pred.data.cpu()
            total += x.size(0)
            correct += torch.sum(pred == labels)
        return  correct*100./total

def load_datasets(dataset_type):
    '''
    Method to load the GTSRB Dataset from data folder
    or specified location. For pre-processing, refer 
    to dataset_loader file
    '''

    global train_set, test_set
    if dataset_type == GTSRB_DATASET:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                 (0.2724, 0.2608, 0.2669))
        ])

        train_data = GTSRB(
            root_dir='./data/', train=True,  transform=transform)
        test_data = GTSRB(
            root_dir='./data/', train=False,  transform=transform)

        train_set = torch.utils.data.DataLoader(
            train_data, batch_size=K, shuffle=True, num_workers=0)
        test_set = torch.utils.data.DataLoader(
            test_data, batch_size=K, shuffle=False, num_workers=0)

    elif dataset_type == CIFAR_10_DATASET:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                                  download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                                 download=True, transform=transform)

        train_set = torch.utils.data.DataLoader(train_data, batch_size=K,
                                                shuffle=True, num_workers=0)
        test_set = torch.utils.data.DataLoader(test_data, batch_size=K,
                                               shuffle=False, num_workers=0)

    return train_set, test_set


def saveModel(epoch, model,optimizer,loss,path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)

def loadModel(model,optimizer,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model,optimizer, epoch, loss;