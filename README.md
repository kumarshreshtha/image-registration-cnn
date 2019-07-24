# Linear and Deformable Image Registration with 3D CNN
A PyTorch implementation of CNN based MRI image registration based on MICCAI 2018 paper [Linear and Deformable Image Registration with 3D Convolutional Neural Networks](https://arxiv.org/abs/1809.06226)

**For a quick introduction to image registration and summary of the paper check out this [presentation](./presentation/Presentation.pdf).**

## Network Architecture
![alt text](./presentation/network_architecture.png "Network Architecture")

## Requirements
- Python 3
- PyTorch
- Torchvision
- Numpy
- Skimage
- Tqdm

## Usage
### Train
```
python ./bin/train.py

optional arguments:
--linear                  whether to include the affine transform branch or not [default value is True]
--epochs                  number of epochs to train [default value is 300]
```

