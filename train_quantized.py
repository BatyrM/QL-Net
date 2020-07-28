from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from load_data import get_data
from qlnet_model_quantized import BS_Net
from train_utils_quantized import train, test
from  training_parameters import get_params


args = get_params()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader = get_data(args, dataset='mnist', ifTrain=True)
test_loader = get_data(args, dataset='mnist', ifTrain=False)

model = BS_Net()
model.load_state_dict(torch.load('mnist_baseline.pth'))
model.to(device).eval()

layer_id0 = 0
layer0 = 'layer' + str(layer_id0)
tree0 = torch.load('tree_' + layer0)
nodes0 = np.asarray([tree0[i].centroid for i in range (0, np.shape(tree0)[0])])

layer_id1 = 1
layer1 = 'layer' + str(layer_id1)
tree1 = torch.load('tree_' + layer1)
nodes1 = np.asarray([tree1[i].centroid for i in range (0, np.shape(tree1)[0])])

layer_id2 = 2
layer2 = 'layer' + str(layer_id2)
tree2 = torch.load('tree_' + layer2)
nodes2 = np.asarray([tree2[i].centroid for i in range (0, np.shape(tree2)[0])])

lookup_table0 = torch.FloatTensor(nodes0).to(device).unsqueeze(0)
lookup_table1 = torch.FloatTensor(nodes1).to(device).unsqueeze(0)
lookup_table2 = torch.FloatTensor(nodes2).to(device).unsqueeze(0)
# lookup_table = torch.FloatTensor(nodes).to(device)
print(lookup_table0.size())
print(lookup_table1.size())
print(lookup_table2.size())
args.out_name = 'mnist_ql_layer'+str(layer_id1)+'&layer'+str(layer_id2)+'.pth'
print("\n\nWith only first layer + input layer quantized:\n")
train(model, train_loader, test_loader, args, device, layer_id=layer_id1, tree=[[lookup_table0], [lookup_table1]])
print("\n\nWith both layers + input layer quantized:\n")
train(model, train_loader, test_loader, args, device, layer_id=layer_id2, tree=[[lookup_table0], [lookup_table1], [lookup_table2]])

# test(model, test_loader, device)
