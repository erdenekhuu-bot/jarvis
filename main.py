import os
from module.nn import NeuralNetwork
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = NeuralNetwork().to(device)
    print(model)
    

if __name__ == "__main__":
    main()