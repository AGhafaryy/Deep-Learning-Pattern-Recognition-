# Multilayer Neural Network only using numpy

In this assignment, we were handed a classification task to perform on the CIFAR-100 dataset. Unlike the previous assignment, we did not implement logistic regression and softmax regression, but instead we used multi-layer neural networks with a softmax output activation function, just because we are dealing with a multi-class classification task. We will not be using the full 100 classes that the dataset offers, but we will only restrict ourselves to 20 target labels, which are the 20 superclasses of the dataset. In the last part of the problem we just experimented with 100 classes but this was not the focus of the assignment. We started out with a 2 layer network, ie, an input layer, a hidden layer and an output layer, with the hidden layer having a tanh activation function and the output layer having a softmax activation function. Then throughout the problem improved the accuracy by changing the architecture of the model. We started by adding momentum, then regularization, then we moved on to experimenting with different hidden activation functions, and finally increased the number of hidden layers of the networks. The last task, was to use our best network found throughout the process, and train it onto the 100 classes and check its performance. Let us now walk you through our results and interesting findings. In short, we started with normalizing the data using z normalization, which is popular when dealing with images as training data (we will explain it in depth later), and passed it through a neural network of 2 layers, tanh and softmax activation functions and a momentum of 0.9, we first got an accuracy percent score of 24.46%. We then started improving it, we added  regularization term, which was another hyperparameter that we tuned, and set it finally to 0.01, this improved our accuracy to 24.74%. Next, we experimented with sigmoid and ReLU activation functions for hidden layers which got us 24.17% and 27.84% accuracy respectively. This can be explained by ReLU being simple and effective in not firing up all units simultaneously which leads to better learning and better convergence. Next, we changed the network topology, by first halving the number of hidden units, this reduced our accuracy to 25.73% , which was expected because there is less ‘learning’ happening. Then we doubled the original number of hidden units, and this bumped our accuracy to 28.21%, which was also expected. After that, we kept the number of parameters same, but increasing the number of layers of the overall network, which got us an accuracy of 27.97%. Finally, we projected our best network found throughout this whole process and tried it on the original 100 classes of the CIFAR-100 dataset, this reduced our accuracy to 17.55%.

## How to run
### Step 1: Install dependencies
`pip install -r requirements.txt`

### Step 2: Download data, if not aldready present.
Run the shell file named: get_cifar100data.sh
You may refer the internet to know how to run it on your respective OS/Platforms.\
This will download the data into the data folder.\
Make sure your file structure looks something like the following:\
![File Tree Structure](readmePics/fileTreeStructure.png)

### Step 3: Running the neural network (main.py)
To change any hyperparameter or the architecture of the neural network, the only thing you need to change is the config file present in the config folder.

The config file is pretty self-explanatory. (For Eg. You can add or remove the layers, by directly changing the layer_specs array and mentioning the number of nodes/neurons a layer would have)

To run training directly how the config file is setup, run the following command from the cmd line:\
`python main.py --experiment run_as_is`\
This command will start training the neural network on 20 course class labels with hyperparameters and network architecture from the config file.

If you wanna run the network on 100 fine labels, use the following command:\
`python main.py --experiment run_as_is_100`\
Just make sure to have 100 at the end in layer_specs array in the config file because this indicates the network how many classes will it predict.


There are some defauts we have set according to our problem statement which can be ran using the command line as follows:
1. `python main.py --experiment test_gradients`

    a. This command executes a gradient check by comparing what the delta after one propagation should be according to our neural network V/S Mathematically. It will print out a table showing the results. To learn more about it refer our report on this assignment.
2. `python main.py --experiment test_momentum`

    a. This command executes the neural network with a momentum term of gamma = 0.9
3. `python main.py --experiment test_regularization`

    a. This command introduces a regularization term using L2 regularization of lambda = 0.01
4. `python main.py --experiment test_activation`

    a. This command runs the neural network on ReLU and sigmoid as hidden layer activation function. Our code supports tanh, ReLU, sigmoid as activation functions which can be edited from the config file.
5. `python main.py --experiment test_hidden_units`

    a. There are two parts to this command. The first part runs the neural network on half the number of hidden units i.e. the units drops to 64 instead of 128.
    
    b. Next, we double the number of hidden units to 256, and run the NN on it.
6. `python main.py --experiment test_hidden_layers`

    a. This will add an additional hidden layer to the architecture making it 2 hidden layer of 128 units each and one output layer.
7. `python main.py --experiment test_100_classes`

    a. This will run our best tuned model we could find using hyperparameterized tuning and the best activation function too on 100 fine classes instead of 20 coarse.

Each of this case has been explained in detail in the report, and we also explain the reason according to us why is the accuracies increasing or decreasing.

Also note, each of the command except check_gradients will result in a plot and it will be automatically stored in the plots folder.



## How did we hyperparameterized?
We did it using manually implementing Gridsearch on the three main variables namely: Learning rate, regularization and batch size. You can find the code of this in the ipynb notebook.

## Any more questions?
Refer the report or contact us on linkedin.




## PA2-Starter (Problem Statement)

1. main.py is the driver code that has to be run in order to run different set of experiments
2. run get_cifar100data.sh to download the data. The dataset will be downloaded in 'data' directory 
3. config files need to be in the 'config' directory
4. You are free to create new functions, change existing function
signatures, add/remove member variables/functions in the provided classes but you need to maintain the overall given structure 
of the code. Specifically, you have to mandatorily use the classes provided in neuralnet.py to create your model (although, 
like already mentioned, you can  add or remove variables/functions in these classes or change the function signatures)
5. We have marked sections to be implemented by you as TODO for your convenience



## Authors
- [@Jay Jhaveri](https://github.com/JayJhaveri1906)
- [@Andrew Ghafari](https://github.com/AGhafaryy)