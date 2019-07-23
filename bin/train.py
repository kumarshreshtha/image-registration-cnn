import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from models.register_3d import Register3d
from utils.earlystopping import EarlyStopping
from utils.dataset import CTScanDataset
from utils.transforms import IntensityNorm, Rescale, ToTensor
from tqdm import tqdm
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser(description="Train 3D CNN for Image \
    Registration")
parser.add_argument("--linear", default=True, type=bool, help="include/exclude\
     linear registration branch in model")
parser.add_argument("--epochs", default=300, type=int, help="number of epochs\
     to train the model")

opt = parser.parse_args()

# all hyperparameters have been set to the values used by Stergios et al.

affine_transform = opt.linear
alpha = beta = 1e-6
num_epochs = opt.epochs
batch_size = 2      # works for Nvidia GeForce GTX 1080 (as mentioned in paper)
val_batch_size = 4  # may be able to validate with larger batch, depends on GPU
train_path = './../data/train'
val_path = './../data/val'

composed = transforms.Compose([IntensityNorm(), Rescale(), ToTensor()])
train_dataset = CTScanDataset(train_path, transform=composed)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CTScanDataset(val_path, transform=composed)
valloader = DataLoader(val_dataset, batch_size=val_batch_size)

model = Register3d(trainloader[0].size, linear=affine_transform)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                 patience=50, verbose=True)
early_stop = EarlyStopping(patience=100, verbose=True)


for epoch in range(num_epochs):
    train_bar = tqdm(trainloader)
    for source, target in train_bar:
        optimizer.zero_grad()
        model.train()
        if affine_transform:
            output, deform_grads, theta = model(source, target)
        else:
            output, deform_grads = model(source, target)
        loss = loss_func(output, target)
        if affine_transform:
            loss += alpha*torch.sum(torch.abs(theta-torch.eye(3, 4)))
        loss += beta*torch.sum(
            torch.abs(deform_grads-torch.ones_like(deform_grads)))
        loss.backward()
        optimizer.step()
        train_bar.set_description(desc=f"[{epoch}/{num_epochs}] \
            loss: {loss:.4f}")

    model.eval()

    val_bar = tqdm(valloader)
    total_val_loss = 0
    with torch.no_grad():
        for source, target in val_bar:
            out, val_dgrads, val_theta = model(source, target)
            val_loss = loss_func(out, target)
            val_loss += alpha*torch.sum(torch.abs(val_theta-torch.eye(3, 4)))
            + beta*torch.sum(torch.abs(val_dgrads-torch.ones_like(val_dgrads)))
            total_val_loss += val_loss.item()
    total_val_loss /= len(valloader)
    val_bar.set_description(desc=f"val loss: {total_val_loss:.4f}")
    scheduler.step(total_val_loss)
    early_stop(model, total_val_loss, epoch)
    if early_stop.early_stop:
        print("Early Stopping")
        break
print('finished training')
