# Convolutional Network Implementation for Semantic Segmentation using Pytorch

In this third assignment, we were given the PASCAL VOC-2007 dataset, and were tasked to perform pixel-level segmentation. This dataset is a very popular one in the computer vision spectrum, and was created as part of the annual Visual Object Classes Challenge, and served as the dataset of that competition from 2005 to 2012. The original dataset contains more than 9000 images, that are annotated with object bounding boxes for 20 different categories such as car, cat, dog, boat and many more. We haven’t been given the whole dataset but only roughly 600 images split evenly in thirds between training, validation and testing. Before giving a rundown of the assignment given and how we approached the task at hand, let us first talk about the evaluation metrics. We had 2 evaluation metrics, pixel accuracy and intersection over union (IoU). The assignment went as follows: we first started by implementing a baseline model that had briefly the following architecture: 5 convolution layers, and 5 deconvolution layers each followed by a batch normalization. This gave us a loss of 1.363, a pixel accuracy of 72.08% and an IoU of 0.055, which were close to expectations. Then we started improving it by doing multiple tricks. The first was using a cosine annealing learning rate scheduler. This lowered our overall pixel acc to 71.58%, increased our loss to 1.34 and our average IoU to 0.0613. After that, we implemented data augmentation techniques mainly mirror flips, rotating the images and cropping them. This gave us better loss function, by decreasing it to 1.316, better IoU, increasing it to 0.0639 but lower pixel acc, 70.98%. Finally we used a weighted loss, which was specifically done to target the class imbalance problem that we were facing. This approach gave us a 2.23 loss function value, a better IoU of almost 0.068 and a lower pixel accuracy of 69.1%. The last problem was basically a way for us to try different architectures and apply transfer learning. We first came up with a personal network, and we did that by making significant changes to the network we had previously. This approach gave us a worse loss of almost 4.4, but better IoU, reaching now 0.0765 and lower pixel acc of 69%. Next thing is that we applied transfer learning to ResNet18 model. We had to do some changes to make it appropriate to the task we have, and these changes were removing the last two fully connected layers, locking the resnet’s weights and adding instead a 4 layer deconvolution network each with a ReLU activation function and batch normalization. This approach also increase dour IoU significantly, making it go to 15%, a good pixel acc of 74.4% and a good loss function as well of 1.8. Finally, we implemented the U-Net architecture that was mentioned in [1]. The architecture of U-Net model was this. This final scheme gave us a loss function of 2.2, an IoU of 0.087 and a pixel acc of 67.747%.

## How to run
### Step 1: Install dependencies (Note: this requirements.txt is a bit dicey (Manually Cut Shorted), should mostly work, but was auto created and had 100+ packages due to the project being made on a shared UCSD environment.)
`pip install -r requirements.txt`

### Step 2: Download data, if not aldready present.
Run the download.py to auto download the voc2012 dataset.

### Step 3: Running the neural network (main.py)
We have created multiple default configs and architecture depending on the specific problem statements which can be ran using the command line as follows:
1. `python train.py`

    a. This command executes a baseline model (description in questionaire)
2. `python train_4_a.py`

    a. This command executes the baseline model with Cosine Anealing Scheduler enabled.
3. `python train_4_b.py`

    a. In this we randomly augment the dataset by introducing flipping, rotations, and cropping. (new train set size 3x original size)
4. `python train_4_c.py`

    a. Runs with weighted loss to fight the class imbalance problem.
5. `python train_5_a.py`

    a. Runs a better version of the baseline model.
    
6. `python train_5_b.py`

    a. Runs by adding multiple deconvolution layers on a pretrained resnet18 layer. (replaces FCN and Avg Pooling layer)
7. `python train_5_c.py`

    a. Runs our interpretation of the U-Net Architecture.

Each of this case has been explained in detail in the report along with its IoU (Intersection over Union) and Pixel Accuracy metric, and we also explain the reason according to us why is the accuracies increasing or decreasing.

Also note, each of the command will result in a plot and it will be automatically stored in the plots folder, while the best models will be stored in the calling directory.

P.S. You may customize any models' hyperparameters or it's architecture in their respective files. It is highly intuitive due to the variable names and comments.


### Runnning the inference
The SegmentationVisualization.ipynb contains the code of running inference of each of the models. You may refer the code there, or modify the modelss array and the number of models variable to add a custom model in the inference loop.


## Any more questions?
Refer the report or contact us on linkedin.




## PA3-Starter (Problem Statement)

The problem statement is present in the CSE_251B__PA3_winter_2023.pdf


## Authors
- [@Jay Jhaveri](https://github.com/JayJhaveri1906)
- [@Andrew Ghafari](https://github.com/AGhafaryy)