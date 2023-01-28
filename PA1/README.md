## Steps before running
0) Use this drive [link](https://drive.google.com/drive/folders/162ql8iESDd_UQU2N99lAxXaf6XnzpWPz?usp=sharing) to get all the .py files mentioned below.

1) If running from google colab, after mounting the drive, copy the path where the .py files are located (Namely: network, data, networkSmax, image should be in one folder). Paste this path replacing the existing path in the following areas:
    * Line 6 of main.py
    * Line 3 of network.py
    * Line 3 of networkSmax.py

2) Copy the path of the folder where you want to save the plots and images generated. Paste them at:
    * 119, 190, 204 lines in main.py (Keep in mind that you only replace the path and not the file name, else it will throw an exception/error)

3) Also take a note of the path where your data set is located. It would be required while running.

## Steps for running

### The following is a template to run the code from terminal or google colab
`!python main.py --p 30 --batch-size 64 --epochs 100 --learning-rate 0.001 --k-folds 10 --path '/content/drive/MyDrive/251B/PA1'` 

#### hyperparameters:
1) path: paste the path of your dataset here.
2) p: number of principal components
3) batch-size: number of batch size
4) epochs: number of epochs
5) learning-rate: learning rate
6) k-folds: the number of folds you want to divide your training set into for cross k validation.

### When you run it, it will give you 3 options to choose from:
1) Run log reg 2, 7
2) Run log reg 5, 8
3) Run Softmax
Enter the number for the coresponing learning task you want to begin in the input box.

### How it runs
The code will automatically load the dataset and begin training with the given hyperparameters and the choosen training model.
It has early stopping mechanism built in, and after the completion of all folds and epochs it will plot an average loss graph and store it in the designated path.
