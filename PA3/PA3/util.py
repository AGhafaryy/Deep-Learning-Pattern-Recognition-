import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

def iou(pred, target, n_classes = 21):
    # Getting the preds for each pixel
    pred = torch.argmax(pred, dim=1)
    IOUs = []
    # eps = 10**-6
    
    for c in range(n_classes):
        # print("class",c)
        TP = torch.sum((pred == c) & (target == c)).item()
        # print(TP)
        TPFPFN = torch.sum((pred == c) | (target == c)).item()
        # print(TPFPFN)
        
        IOU = 0
        if TPFPFN != 0:
            IOU = (TP) / (TPFPFN)
            # print("IOU", IOU)
            IOUs.append(IOU)
        # print(IOUs)
        # print(np.mean(IOUs))
    # sys.exit("nan")
        
    # print(IOUs, np.mean(IOUs))
    return np.mean(IOUs)
        
    

def pixel_acc(pred, target):
    # Getting the preds for each pixel
    pred = torch.argmax(pred, dim=1)
    
    numi = torch.sum(pred == target).item()
    deno = len(pred)*len(pred[0])*len(pred[0][0])
    
    
    return numi/deno
    