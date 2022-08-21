# Linear and Deformable Image Registration with 3D CNN

A minimal PyTorch implementation of CNN based MRI image registration from the paper Christodoulidis Stergios et al. "[Linear and Deformable Image Registration with 3D Convolutional Neural Networks](https://arxiv.org/abs/1809.06226)", 2018.

For a quick introduction to image registration and summary of the paper check out this [presentation](./presentation/Presentation.pdf).

*Note: This is not the official implementation. I had to present this paper for an undergrad internship interview back in 2019 and I ended up writing this quick prototype to go with it. Cleaned it up a bit recently but it's still a barebone running prototype than anything else.*

## Network Architecture

![network architecture from the paper](./presentation/network_architecture.png "Network Architecture")

## Requirements

You can download all requirements for this project using `pip` as follows:

```sh
pip3 install -r requirements.txt
```

## Usage

### Train

As mentioned above, the implementation is pretty barebone. Once you have your data directories set up, you can change the hyperparameters in `train.py` and simply launch the train script.

```sh
python3 train.py
```
