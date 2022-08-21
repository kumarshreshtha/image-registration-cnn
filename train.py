
import torch
from torch import nn
from torch import optim
from torch.utils import data
import tqdm

import model
from utils import dataset
from utils import earlystopping


# all hyperparameters have been set to the values used by Stergios et al.

do_affine_transform = True
alpha = beta = 1e-6
num_epochs = 300
batch_size = 2      # works for Nvidia GeForce GTX 1080 (as mentioned in paper)
val_batch_size = 4  # may be able to validate with larger batch, depends on GPU
train_path = './../data/train'
val_path = './../data/val'
checkpoint_path = "./../checkpoint/net.pt"

train_dataset = dataset.RegDataset(train_path)
trainloader = data.DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
val_dataset = dataset.RegDataset(val_path)
valloader = data.DataLoader(val_dataset, batch_size=val_batch_size)

dev = (torch.device("cuda")
       if torch.cuda.is_available() else torch.device("cpu"))

net = model.Model(do_affine=do_affine_transform)
net = net.to(dev)

loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=0.1,
                                                 patience=50,
                                                 verbose=True)
early_stop = earlystopping.EarlyStopping(patience=100)

for epoch in range(num_epochs):
    train_bar = tqdm.tqdm(trainloader)
    net.train()
    for source, target in train_bar:
        source, target = source.to(dev), target.to(dev)
        optimizer.zero_grad()
        output, spatial_grads, theta = net(source, target)
        loss = loss_func(output, target)
        if theta is not None:
            loss = loss + alpha * torch.sum(torch.abs(theta - torch.eye(3, 4)))
        loss = loss + beta * torch.sum(
            torch.abs(spatial_grads - torch.ones_like(spatial_grads)))
        loss.backward()
        optimizer.step()
        train_bar.set_description(
            desc=f"[{epoch}/{num_epochs}] loss: {loss:.4f}")

    net.eval()
    val_bar = tqdm(valloader)
    with torch.no_grad():
        total_val_loss = 0
        for source, target in val_bar:
            out, val_spatial_grads, val_theta = model(source, target)
            val_loss = loss_func(out, target)
            val_loss = (val_loss + alpha * torch.sum(
                torch.abs(val_theta - torch.eye(3, 4))))
            val_loss = (
                val_loss + beta * torch.sum(torch.abs(
                    val_spatial_grads - torch.ones_like(val_spatial_grads))))
            total_val_loss = total_val_loss + val_loss.item()
        total_val_loss /= len(valloader)
        val_bar.set_description(desc=f"val loss: {total_val_loss:.4f}")
    scheduler.step(total_val_loss)
    if early_stop(total_val_loss):
        break

torch.save(net.state_dict(), checkpoint_path)
