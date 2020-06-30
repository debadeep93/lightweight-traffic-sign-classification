# Lightweight Traffic Sign Classification

## Introduction

Traffic sign classification is the task of identifying and recognizing the different types of traffic signs that are found. Traffic sign classification is an integral part of autonomous vehicles, more popularly known as self-driving cars as it would enable the car to understand the traffic rules and adhere to traffic laws without human intervention. This is also a real-time classification task and any delay in correctly identifying a traffic sign would be life-threatning, be it for the passengers in the autonomous vehicles, or people in the immediate vicinity of the vehicle including pedestrians and cyclists. Additionally, since the classification systems are deployed in a resource-constrained environment, the network has to be lightweight, robust and highly accurate.  

In [1], Zhang et. al. proposed two unique lightweight networks that can obtain higher recognition precision while preserving less trainable parameters in the models. The first network is a teacher model which is trained on the GTSRB dataset, and the second network is a simple five-layer convolutional neural network which learns from the teacher model through knowledge distillation. For our project, we aim to *reproduce* the model proposed in the paper from scratch. There is no existing code implementation of this paper as released by the authors, or as an independent implementation. We use the model architectures and the hyperparameters given in the paper to achieve results as close as possible to the ones described in the paper. We also attempt to describe any discrepancies that may arise between the paper's results and the results obtained by us.

## Related Work

There are existing deep neural networks which can identify objects, and specifically traffic signs, with significant accuracy. Traditional traffic sign classification networks which use ELM or SVM for feature classes use handcrafted features which results in significant loss of information. CNNs like MSCNN, MCDNN, CNN-ELM take advatage of traditional learning classifiers and achieve greater recognition rate as compared to the traditional methods. CNN-HLSGD trains a CNN with hinge loss and achieves a recognition rate on the GTSRB dataset better than that of most methods.

Though these networks perform very well on the traffic sign classification task, they are not suitable to be deployed in a resource-constrained environment with reduced computing power and storage space. A possible solution to this challenge is to compress these heavy CNNs using five different methods: low-rank, pruning, quantization, knowledge distillation, and compact network design. In [1], the authors design two networks which take advantage of the network compression techniques resulting in lightweight networks with fewer trainable parameters.

## Dataset

## Teacher-Student Network Architecture

## Knowledge Distillation

## Algorithm

## Results

## Challenges Faced

### Training the teacher model
The major challenge we faced was in the training of the teacher model. Google Colab spec, local machines specs.

## Inference

## References
[1] Zhang, J., Wang, W., Lu, C. et al. Lightweight deep network for traffic sign classification. Ann. Telecommun. (2019). https://doi.org/10.1007/s12243-019-00731-9