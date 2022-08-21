"""Simple tests to check model sanity."""

import torch

import model

def test_model_out_shapes():
    net = model.Model()
    src = torch.randn(1,1,64,192,192)
    tgt = torch.randn(1,1,64,192,192)
    warped_src,spatial_grads,affine_mat = net(src, tgt)
    assert warped_src.shape == tgt.shape
    assert spatial_grads.shape == torch.Size([1,3,64,192,192])
    assert affine_mat.shape == torch.Size([1,3,4])

def test_model_grads():
    net = model.Model()
    src = torch.randn(1,1,64,192,192)
    tgt = torch.randn(1,1,64,192,192)
    warped_src,_,_ = net(src, tgt)
    warped_src.mean().backward()
    for param in net.parameters():
        assert param.grad is not None