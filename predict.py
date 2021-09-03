import torch
import dataset
from model import LinearRegression
from train import mini_batch, val_fn, loss_fn

val_losses = []

model = LinearRegression()

checkpoint = torch.load("model_checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

val_step = val_fn(model, loss_fn)
with torch.no_grad():
    batch = mini_batch(dataset.val_loader, val_step)
    val_losses.append(batch)

print(val_losses)
