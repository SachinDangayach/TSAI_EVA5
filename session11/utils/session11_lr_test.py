import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

import copy

def train(model, device, train_loader, criterion, optimizer, epoch, train_losses, train_acc, lr):
    """Train network"""
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Learning Rate = {lr} epoch={epoch} Train Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def LR_test(max_lr, min_lr,device,iterations,steps,model,criterion,train_loader,momemtum = 0.9,weight_decay=0.05, plot= True ):
    delta = (max_lr - min_lr )/steps
    lr = min_lr
    epochs = math.ceil(iterations/len(train_loader))
    Lrtest_train_acc = []
    LRtest_Lr = []

    for step in range(steps+1):
        testmodel = copy.deepcopy(model)
        optimizer = optim.SGD(testmodel.parameters(), lr=lr ,momentum=momemtum,weight_decay=weight_decay )
        train_acc = []
        train_losses = []
        for epoch in range(epochs):
            train(testmodel, device, train_loader,criterion, optimizer, epoch, train_losses, train_acc, lr)
        Lrtest_train_acc.append(train_acc[-1])
        LRtest_Lr.append(optimizer.param_groups[0]['lr'])
        lr += delta

    if(plot):
        plt.style.use('ggplot')
        plt.plot(LRtest_Lr, Lrtest_train_acc)
        plt.ylabel('train Accuracy')
        plt.xlabel("Learning rate")
        plt.title("Lr v/s accuracy")
        plt.show()
