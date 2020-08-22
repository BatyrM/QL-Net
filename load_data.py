'''
File to load data
ifTrain =True -> load train set
        = False -> load test set
'''
import torch
import torchvision
from torchvision import datasets, transforms

def get_data(args, dataset='mnist', ifTrain=True):
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    if dataset=="mnist":
        dataSet = get_mnist(args, ifTrain)
    elif dataset=='cifar10':
        dataSet = get_cifar10(args, ifTrain)
    else:
        print("I don't know this dataset")
    return dataSet

def get_mnist(args, ifTrain):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=ifTrain, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=ifTrain, **kwargs)
    return loader

def get_cifar10(args, ifTrain):
    
    loader = None
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if ifTrain:
        loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=ifTrain, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=ifTrain, **kwargs)

    else:
        loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root = './data', train=ifTrain,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
                        ])),
            batch_size=args.batch_size, shuffle=ifTrain, **kwargs)
    
    return loader
