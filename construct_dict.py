import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import load_activations as la
import hierarhical_tree_gpu as ht



from load_data import get_data
# from train_utils import train, test
from qlnet_model_quantized import BS_Net
from train_utils_quantized import train, test
from  training_parameters import get_params

def if_exist(path):
    if not os.path.exists(path) :
        os.makedirs(path)

## 1. Load model
args = get_params()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BS_Net()
model.load_state_dict(torch.load('mnist_baseline.pth'))
model.to(device).eval()

layer_id = 1
layer1 = 'layer' + str(layer_id)
activation_folder1 = os.path.join('./activations', layer1)
if_exist(activation_folder1)

layer_id = 2
layer2 = 'layer' + str(layer_id)
activation_folder2 = os.path.join('./activations', layer2)
if_exist(activation_folder2)

### 2. Load train data
train_loader = get_data(args, dataset='mnist', ifTrain=True)
## 2. Extract activations for futher look-up dictionary construction
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    _, activations = model(data)
    activation1 = activations[0].cpu().data.numpy()
    activation2 = activations[1].cpu().data.numpy()
    torch.save(activation1, os.path.join(activation_folder1, layer1 + '_'+str(batch_idx)+'.npy'))
    torch.save(activation2, os.path.join(activation_folder2, layer2 + '_'+str(batch_idx)+'.npy'))
    if batch_idx>6:break

# 3 Construct Look-up Dictionary
# parameters for look-up dictionary construction
n_cl = 10
density = 30
max_depth = 1

# Load activations
print('Load activations')
data1 = la.load_data(activation_folder1) # load patched data1
data2 = la.load_data(activation_folder2) # load patched data2
print('Construct tree 1')
tree1 = ht.construct(data1, n_cl, density, max_depth)
torch.save(tree1, 'tree_' + layer1)
print('Construct tree 2')
tree2 = ht.construct(data2, n_cl, density, max_depth)
torch.save(tree2, 'tree_' + layer2)