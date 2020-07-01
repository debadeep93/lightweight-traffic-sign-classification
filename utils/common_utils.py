import torch
import torchvision
from torch import nn
from torch.autograd import Variable as var
import torchvision.transforms as transforms
from utils.gtsrb_loader import GTSRB
import torch.nn.functional as F

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

    cec = nn.CrossEntropyLoss().cuda()
    loss = nn.KLDivLoss()(F.log_softmax(student_output/T, dim=1),
                             F.softmax(teacher_output/T, dim=1)) * (2. * alpha * T * T) + \
              cec(F.softmax(student_output), labels) * (1. - alpha)
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
    total = 0
    correct = 0
    with torch.no_grad():
        for i,(images,labels) in enumerate(data):
            images = var(images.cuda())
            x = model.forward(images)
            value,pred = torch.max(x,1)
            pred = pred.data.cpu()
            total += x.size(0)
            correct += torch.sum(pred == labels)
        return  correct*100./total


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
