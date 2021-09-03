import config
import dataset
import model

import numpy as np
import torch
import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)

model = model.LinearRegression()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)



def train_fn(model, optimizer, loss):

    def train_step(x_train, y_train):

        model.train() # set the model to train

        yhat = model(x_train).squeeze()

        loss = loss_fn(yhat, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item() # loss is gradient, changing it to integer

    return train_step

train_step = train_fn(model, optimizer, loss_fn)


def val_fn(model, loss_fn):
    def val_step(x_train, y_train):
        model.eval()

        yhat = model(x_train).squeeze()

        loss = loss_fn(yhat, y_train)

        return loss.item()

    return val_step

val_step = val_fn(model, loss_fn)


def mini_batch(data_loader, step):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        mini_batch_loss = step(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)

    # Computes average loss over all mini-batches
    # That's the epoch loss
    loss = np.mean(mini_batch_losses)
    return loss



losses = []
val_losses = []

for i in range(config.N_EPOCHS):
    train_loss = mini_batch(dataset.train_loader, train_step)
    losses.append(train_loss)

    if (i+1) % 50 == 0:
        print(f"Loss: {train_loss}")
    # validation - no gradients in validation!
    with torch.no_grad():
        val_loss = mini_batch(dataset.val_loader, val_step)
        val_losses.append(val_loss)



checkpoint = {
    'epoch' : config.N_EPOCHS,
    'model_state_dict' : model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': losses,
    'val_loss': val_losses
}


torch.save(checkpoint, 'model_checkpoint.pth')
