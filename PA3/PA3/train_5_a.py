from basic_fcn_5_a import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
from torchvision import transforms
import torchvision
import util
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import PIL
import sys
import matplotlib.pyplot as plt

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



#TODO Get class weights
def getClassWeights():

    raise NotImplementedError


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



# input_transform = standard_transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(15),
#     transforms.RandomRotation(2),
#     transforms.RandomCrop([224, 224]),
#     transforms.ToTensor()
#     ])

# target_transform = standard_transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(15),
#     transforms.RandomRotation(2),
#     transforms.RandomCrop([224, 224]),
#     MaskToTensor(),
#     ])

common_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop([224, 224]),
])
#transform_both = transforms.Lambda(lambda image, label: (common_transform(image), common_transform(label)))

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.Compose([
    MaskToTensor(),
])

train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform, common_transform=common_transform, tcopies = 3)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform, common_transform=common_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform, common_transform=common_transform)

BSize = 8

train_loader = DataLoader(dataset=train_dataset, batch_size= BSize, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= BSize, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= BSize, shuffle=False)

epochs = 30

n_class = 21

fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")   # TODO determine which device to use (cuda or cpu)

    
lr = 0.007
lr_min = 0.0005

bestModelPath = 'best_model_5_a.pth'

fcn_model = fcn_model.to(device) # TODO transfer the model to the device


MOMENTUM = 0.99
OPTIMIZER = "AdamW"

if OPTIMIZER == "AdamW":
    optimizer = torch.optim.AdamW(fcn_model.parameters(), lr = lr)
elif OPTIMIZER == "ADAM":
    optimizer = torch.optim.Adam(fcn_model.parameters(), lr = lr)
elif OPTIMIZER == "SGD":
    optimizer = torch.optim.SGD(fcn_model.parameters(), lr = lr, momentum = MOMENTUM)
    


CRITERION = "Softmax + Cross Entropy"
WEIGHTED = True
if WEIGHTED:
    wts = {0: 7715366, 1: 48254, 15: 907370, 19: 119886, 4: 46910, 2: 43413, 20: 130575, 5: 70999, 11: 131729, 8: 187365, 6: 199300, 7: 74579, 14: 79197, 9: 111148, 18: 168819, 10: 78270, 13: 85963, 12: 143997, 16: 31601, 17: 23115, 3: 88928}
    wtedwts = [1 - (wt / sum(list(wts.values()))) for wt in list(wts.values())]
    
    wtedwts = torch.FloatTensor(wtedwts)
    wtedwts = wtedwts.to(device)
    criterion =  nn.CrossEntropyLoss(weight = wtedwts)

else:
    criterion =  nn.CrossEntropyLoss()
# TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html


SCHEDULER = True
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)


EARLYSTOP = True
patienceLimit = 3
prevLoss = float("inf")
patience = 0

# To Plot
trainLoss = []
valLoss = []
bstop = -1
estop = -1
bestLoss = float("inf")

# TODO
def train():
    global prevLoss
    global bestLoss
    global bstop
    global estop
    best_iou_score = 0.0

    for epoch in range(epochs):
        ts = time.time()
        epochLoss = []
        for iter, (inputs, labels) in enumerate(train_loader):
            # TODO  reset optimizer gradients
        
            optimizer.zero_grad()


            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
            labels =  labels.to(device) # TODO transfer the labels to the same device as the model's

            outputs =  fcn_model(inputs)
            # TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

            loss = criterion(outputs, labels)  #TODO  calculate loss
            epochLoss.append(loss.item())

            # TODO  backpropagate
            loss.backward()

            # TODO  update the weights
            optimizer.step()
            
            if SCHEDULER:
                scheduler.step()

            # Printing
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        print()
        current_miou_score, lossi = val(epoch+1)
        
        # To Plot
        valLoss.append(lossi)
        trainLoss.append(np.mean(epochLoss))
        
        if lossi > prevLoss:
            patience += 1
            print("This loss higher than before, increasing patience to", patience)
            if patience == patienceLimit and EARLYSTOP == True:
                print("Khalas, my patience is up, Early Stopping & saving weights")
                break
        else:
            patience = 0

            if lossi < bestLoss:
                estop = epoch
                bestLoss = lossi

        prevLoss = lossi

        if current_miou_score > best_iou_score:
            bstop = epoch
            best_iou_score = current_miou_score
            
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': fcn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, bestModelPath)
            # save the best model
        print("__"*100)
    
 #TODO
def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    print("Starting val on device for epoch",epoch, "on device", device)
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):
            inputs =  inputs.to(device)
            labels =  labels.to(device)

            outputs =  fcn_model(inputs)
            
            loss = criterion(outputs, labels)
            
            # print(loss)
            # print(type(loss))
            # print(loss.item())
            losses.append(loss.item())
            
            mean_iou_scores.append(util.iou(outputs,labels))
            accuracy.append(util.pixel_acc(outputs,labels))
            




    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores), np.mean(losses)

 #TODO
def modelTest():
    print("Starting Test on best weights")
    
    saved_model_path = bestModelPath
    fcn_model.load_state_dict(torch.load(saved_model_path)['model_state_dict'])

    # loading best wts
    # fcn_model.load_state_dict(saved_model_state_dict)
    
    
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !
    
    
    losses = []
    mean_iou_scores = []
    accuracy = []


    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(test_loader):
            inputs =  inputs.to(device)
            labels =  labels.to(device)

            outputs =  fcn_model(inputs)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            mean_iou_scores.append(util.iou(outputs,labels))
            accuracy.append(util.pixel_acc(outputs,labels))
            




    print(f"Test Loss is {np.mean(losses)}")
    print(f"Test IoU is {np.mean(mean_iou_scores)}")
    print(f"Test Pixel accis is {np.mean(accuracy)}")



    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    
def plotArr():
    varName = "Loss"
    tloss = trainLoss
    vloss = valLoss
    fname = "5_a_self"
    title = fname
    earlyStop = estop
    
    epochs = np.arange(1, len(tloss) + 1, 1)
    plt.plot(epochs, tloss, label='Train ' + varName)
    plt.plot(epochs, vloss, label='Validation ' + varName)
    if varName == "Loss":
        plt.scatter(epochs[earlyStop], vloss[earlyStop], marker='x', c='g', s=300, label='Early Stop Epoch')
        plt.scatter(epochs[bstop], vloss[bstop], marker='o', c='r', s=100, label='Best Accuracy Model')
    # plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10))
    # plt.xlabel('Epoch')
    plt.ylabel(varName)
    plt.title(title + ' Train and Validation ' + varName + ' vs. Epochs')
    plt.legend()
    plt.savefig("plots/" + fname + '.png')
    plt.close()
    

def exportModel(inputs):    
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    saved_model_path = bestModelPath
    fcn_model.load_state_dict(torch.load(saved_model_path)['model_state_dict'])
    
    inputs = inputs.to(device)
    
    outt = fcn_model(inputs)
    
    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return outt


if __name__ == "__main__":

    params = {
        "learning rate":lr,
        "lr_min":lr_min,
        "epochs":epochs,
        "criterion":CRITERION,
        "weighted loss":WEIGHTED,
        "Optimizer":OPTIMIZER,
        "momentum if SGD":MOMENTUM,
        "scheduler":SCHEDULER,
        "early stop":EARLYSTOP,
        "patience":patienceLimit,
        "device":device,
        "bestModelPath":bestModelPath,
    }
        
    print("Metric before Training")
    val(0)  # show the accuracy before training
    print("__"*100)
    print("Starting Training with the following params:\n",params)
    train()
    modelTest()
    plotArr()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()