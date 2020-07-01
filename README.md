# Lightweight Traffic Sign Classification

## Introduction

Traffic sign classification is the task of identifying and recognizing the different types of traffic signs that are found. Traffic sign classification is an integral part of autonomous vehicles, more popularly known as self-driving cars as it would enable the car to understand the traffic rules and adhere to traffic laws without human intervention. This is also a real-time classification task and any delay in correctly identifying a traffic sign would be life-threatning, be it for the passengers in the autonomous vehicles, or people in the immediate vicinity of the vehicle including pedestrians and cyclists. Additionally, since the classification systems are deployed in a resource-constrained environment, the network has to be lightweight, robust and highly accurate.  

In [1], Zhang et. al. proposed two unique lightweight networks that can obtain higher recognition precision while preserving less trainable parameters in the models. The first network is a teacher model which is trained on the GTSRB dataset, and the second network is a simple five-layer convolutional neural network which learns from the teacher model through knowledge distillation. For our project, we aim to _reproduce_ the model proposed in the paper from scratch. There is no existing code implementation of this paper as released by the authors, or as an independent implementation. We use the model architectures and the hyperparameters given in the paper to achieve results as close as possible to the ones described in the paper. We also attempt to describe any discrepancies that may arise between the paper's results and the results obtained by us.

## Related Work

There are existing deep neural networks which can identify objects, and specifically traffic signs, with significant accuracy. Traditional traffic sign classification networks which use ELM or SVM for feature classes use handcrafted features which results in significant loss of information. CNNs like MSCNN, MCDNN, CNN-ELM take advatage of traditional learning classifiers and achieve greater recognition rate as compared to the traditional methods. CNN-HLSGD trains a CNN with hinge loss and achieves a recognition rate on the GTSRB dataset better than that of most methods.

Though these networks perform very well on the traffic sign classification task, they are not suitable to be deployed in a resource-constrained environment with reduced computing power and storage space. A possible solution to this challenge is to compress these heavy CNNs using five different methods: low-rank, pruning, quantization, knowledge distillation, and compact network design. In [1], the authors design two networks which take advantage of the network compression techniques resulting in lightweight networks with fewer trainable parameters.

## Dataset

The paper uses the German Traffic Sign Recognition Benchmark dataset which is a multi-class, single-image dataset consisting of more than 50,000 images with unique physical traffic sign instances. Since the images are of actual real-life traffic signs, the images also reflect real-life image identification challenges like varying perspectives, shades, lighting conditions and colour degradation. A sample of the dataset is shown in figure 1. The dataset is split into training and test sets with 39,209 images in the training set and 12,630 images.

![Figure 1: A sample of the GTSRB dataset[3]](assets/Overview-of-the-GTSRB-Dataset.png?raw=true)

We have also trained our teacher model on the CIFAR-10[4] dataset which consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Figure 2 shows the classes in the dataset, along with 10 images from each class.

![Figure 2: Classes from the CIFAR-10 dataset](assets/cifar-10.png?raw=true)

## Network Architecture

The authors of this paper have designed two novel lightweight networks for the traffic sign classification task. To summarize, the network primarily consists of a large neural network called the teacher model which is trained on the GTSRB dataset and transfers its knowledge to a smaller network called the student model through knowledge distillation. An overview of the architecture is given in figure 3. We elaborate on each of the components in the sections below.

![Figure 3: A high-level overview of the teacher-student architecture](assets/teacher_student.png?raw=true "Image credits https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764")

### Teacher Network

The teacher network is usually a complex and deep network which is fully trained offline as it requires high computational performance. 

The architecture of the teacher network is shown in figure 4. It consists of two 1x1 convolutional filters that are used to reduce the number of channels of the input feature maps. Using 1x1 instead of 3x3 kernels reduces the number of parameters by one-ninth, while also increasing the non-linearity of the network without changing the size of the feature maps, thereby deepening the network. This is depicted in figure 5, which shows the network's stage module. Inside each cell, the input is spliced and there is a 1x1 kernel and a 3x3 kernel that perform convolutional operations in parallel. The output is then concatenated and passed on to the next layer. The cell operations can be visualized in figure 6. From the architecture of the teacher network, we can observe that there are six cells that are used to connect the different layers and taking advantage of the feature maps of each layer while also accumulating the characteristics of each channel. This relieves us of the vanishing gradient problem. As part of the convolutional operations, batch normalization and ReLU functions are performed in each layer to further avoid the vanishing gradient and gradient explosion problems while also incresing the degree of non-linearity in the network. At the end of stage 3, a 2x2 maxpooling operation is done followed by a linear layer which fully connects all neurons from the penultimate layer to the classfication layer.

The teacher network is trained to converge on the same training set as the student network. The parameters of the teacher network are not updated during the training of the student model, as they only guide the updating of the student network parameters. The performance of the student network is dependent on the performance of the teacher network, so if the teacher network has a low accuracy on the training set, it is difficult to guide the student model to update parameters, and the student model may get stuck in a local optima.

![Figure 4: Architecture of the teacher network](assets/Teacher_architecture.jpeg)

![Figure 5: A stage module](assets/stage_module.jpeg)

![Figure 6: A cell block](assets/cell_block.png)

### Student Network

The student network has a simple end-to-end architecture consisting of five convolutional layers and a fully connected layer. Each type of convolutional filter in a convolutional layer are used to extract specific features of an image. Similar to the teacher network, a batch normalization layer and a ReLU layer are added in each convolutional layer to increase the non-linearity in the network. Max-pooling layers are used after the second and fourth convolutional layers to obtain textural features and an average pooling layer is used after the final convolutional layer to preserve the information of the input feature maps. The pooling layers also prevent overfitting and reduce the dimensionality of the feature maps. Finally, there is a fully connected layer or the classifier layer that transfers the feature vectors into a target class probability. The optimizer used to update weights and bias during the training is the Adam optimizer. The student network learns low-level information like texture from the low-level convolutional features, whole learning semantic information from the high-level convolutional layers. Figure 7 gives an overview of the student network.

![Figure 7: An overview of the student network](assets/student_network.png)

### Knowledge Distillation

Knowledge obtained from the teacher network is transferred to the student network through the process of knowledge distillation. An overview of the knowledge distillation process is given in figure 8. The softened output of the teacher network is used to train the student network on the target datasets. Consider the training sets to be represented as $$D = {X = {x_1, x_2,..., x_n}, {Y = {y_1, y_2,..., y_n}}$$, where $$x$$ and $$y$$ represent an input and a target output respectively. The output of the teacher model and the student model can be represented as $$t = teacher(x)$$ and $$s = student(x)$$ respectively. The student model minimizes the knowledge distillation loss function defined below:
<div align="center">$$ L_{KD} = (1-\alpha)L_{CE}(y, \sigma(s)) + 2T^2\alpha L_{CE}(\sigma(\frac{t}{T}), \sigma(\frac{s}{T})$$</div>
where $$\alpha$$ is a hyperparameter that controls the ratio of the two terms and $$Ta$$ is a temperaturue parameter, $$\sigma()$$ is the softmax function  and $$L_{CE}$$ is a standard cross-entropy loss function.

![Figure 8: Knowledge Distillation Process](assets/knowledge_distillation.png)

### Advantages of the Teacher-Student Architecture

One of the advantages of the teacher student architecture is that the student network only learns the most significant information from the teacher model. The student learns a one-hot class label directly from the traffic sign learning. Stating the example given in the paper, consider an object classification network trained to recognize a car, a cat or a dog, and the likely probabilities of an object being one of these three is [0.05, 0.9, 0.6]. This is a softened output, and since cars are more different that cats or dogs, the difference between the second and the third probablities is smaller than the difference between the first and the second. Since this information only makes a minimal contribution to the weights update, the temperature parameter T is increased to help transfer the knowledge to the student model. 

Another advantage is that since the student network is guided by the teacher network, its convergence space becomes significantly smaller as compared to a model learning directly from a dataset. A visualization is shown in figure 9. The search space is much larger for a traditional network trained directly on a training dataset. We assume that convergence similar to the teacher network can be achieved by the student network too, and so the convergence space of the teacher and student network should overlap. The student network can also have convergence different than that of the teacher network, but given that the student network is guided by the teacher network, the convergence space of the student network will signigicantly overlap with the convergence space of the teacher network.

![Figure 9: Teacher-student convergence space](assets/teacher-student_convergence_space.png)

### Hyperparameters

We have used the same hyperparameters as the authors. For knowledge distillation, we set $$T$$ to 20 and $$\alpha$$ to 0.9 for optimal results.

## Experiment



## Results

## Challenges Faced

### Training the teacher model
The major challenge we faced was in the training of the teacher model. Google Colab spec, local machines specs.

### Limitations
Handling class skew

## Inference

## Future Work

Network Pruning

Data augmentation

Real time from video

## References
[1] Zhang, J., Wang, W., Lu, C. et al. Lightweight deep network for traffic sign classification. Ann. Telecommun. (2019). https://doi.org/10.1007/s12243-019-00731-9

[2] J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, Neural Networks, Available online 20 February 2012, ISSN 0893-6080, 10.1016/j.neunet.2012.02.016. (http://www.sciencedirect.com/science/article/pii/S0893608012000457) Keywords: Traffic sign recognition; Machine learning; Convolutional neural networks; Benchmarking

[3] An Efficient Traffic Sign Recognition Approach Using a Novel Deep Neural Network Selection Architecture: Proceedings of IEMIS 2018, Volume 3 - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/Overview-of-the-GTSRB-Dataset_fig1_327389916 [accessed 30 Jun, 2020]

[4] Krizhevsky A (2009) Learning multiple layers of features from tiny images. Technical Report TR–2009, University of Toronto. http://
www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf. Accessed 30 Jun 2020