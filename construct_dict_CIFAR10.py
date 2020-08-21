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
from vgg_quantized import VGG
from train_utils_quantized_CIFAR10 import train, test
from  training_parameters_CIFAR10 import get_params

def if_exist(path):
    if not os.path.exists(path) :
        os.makedirs(path)

## 1. Load model
args = get_params()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VGG()
model.load_state_dict(torch.load('cifar10_baseline.pth'))
model.to(device).eval()

case_number = args.case_number

if case_number >= 4:
    if case_number == 4:
        layer_id = 0
        layer0 = 'layer' + str(layer_id)
        activation_folder0 = os.path.join('./activations', layer0)
        if_exist(activation_folder0)
        ### 2. Load train data
        train_loader = get_data(args, dataset='cifar10', ifTrain=True)
        ## 2. Extract activations for futher look-up dictionary construction
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            _, activations = model(data)
            #print(len(activations))
            activation0 = activations[0].cpu().data.numpy()
            torch.save(activation0, os.path.join(activation_folder0, layer0 + '_'+str(batch_idx)+'.npy'))
            if batch_idx>6:break

        # 3 Construct Look-up Dictionary
        # parameters for look-up dictionary construction
        n_cl = 10
        max_depth = 1

        # Load activations
        print('Load activations')
        data0 = la.load_data(activation_folder0) # load patched data of input layer
        print('Construct tree input layer')
        tree0 = ht.construct(data0, n_cl, 1, max_depth)
        torch.save(tree0, 'tree_' + layer0)

    elif case_number == 5:
        layer_id = 0
        layer0 = 'layer' + str(layer_id)
        activation_folder0 = os.path.join('./activations', layer0)
        if_exist(activation_folder0)

        layer_id = 1
        layer1 = 'layer' + str(layer_id)
        activation_folder1 = os.path.join('./activations', layer1)
        if_exist(activation_folder1)

        ### 2. Load train data
        train_loader = get_data(args, dataset='cifar10', ifTrain=True)
        ## 2. Extract activations for futher look-up dictionary construction
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            _, activations = model(data)
            activation0 = activations[0].cpu().data.numpy()
            activation1 = activations[1].cpu().data.numpy()
            
            torch.save(activation0, os.path.join(activation_folder0, layer0 + '_'+str(batch_idx)+'.npy'))
            torch.save(activation1, os.path.join(activation_folder1, layer1 + '_'+str(batch_idx)+'.npy'))
            if batch_idx>6:break

        # 3 Construct Look-up Dictionary
        # parameters for look-up dictionary construction
        n_cl = 10
        density = 30
        max_depth = 1

        # Load activations
        print('Load activations')
        data0 = la.load_data(activation_folder0) # load patched data of input layer
        data1 = la.load_data(activation_folder1) # load patched data of layer1
        print('Construct tree input layer')
        tree0 = ht.construct(data0, n_cl, 1, max_depth)
        torch.save(tree0, 'tree_' + layer0)
        print('Construct tree layer 1')
        tree1 = ht.construct(data1, n_cl, density, max_depth)
        torch.save(tree1, 'tree_' + layer1)

    elif case_number == 6:
        layer_id = 0
        layer0 = 'layer' + str(layer_id)
        activation_folder0 = os.path.join('./activations', layer0)
        if_exist(activation_folder0)        
        
        layer_id = 2
        layer2 = 'layer' + str(layer_id)
        activation_folder2 = os.path.join('./activations', layer2)
        if_exist(activation_folder2)
        
        ### 2. Load train data
        train_loader = get_data(args, dataset='cifar10', ifTrain=True)
        ## 2. Extract activations for futher look-up dictionary construction
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            _, activations = model(data)
            activation0 = activations[0].cpu().data.numpy()
            activation2 = activations[2].cpu().data.numpy()
            torch.save(activation0, os.path.join(activation_folder0, layer0 + '_'+str(batch_idx)+'.npy'))
            torch.save(activation2, os.path.join(activation_folder2, layer2 + '_'+str(batch_idx)+'.npy'))
            if batch_idx>6:break

        # 3 Construct Look-up Dictionary
        # parameters for look-up dictionary construction
        n_cl = 10
        density = 30
        max_depth = 1

        # Load activations
        print('Load activations')
        data0 = la.load_data(activation_folder0) # load patched data of input layer
        data2 = la.load_data(activation_folder2) # load patched data of layer2
        print('Construct tree input layer')
        tree0 = ht.construct(data0, n_cl, 1, max_depth)
        torch.save(tree0, 'tree_' + layer0)
        print('Construct tree layer 2')
        tree2 = ht.construct(data2, n_cl, density, max_depth)
        torch.save(tree2, 'tree_' + layer2)

    elif case_number == 7:
        layer_id = 0
        layer0 = 'layer' + str(layer_id)
        activation_folder0 = os.path.join('./activations', layer0)
        if_exist(activation_folder0)

        layer_id = 1
        layer1 = 'layer' + str(layer_id)
        activation_folder1 = os.path.join('./activations', layer1)
        if_exist(activation_folder1)

        layer_id = 2
        layer2 = 'layer' + str(layer_id)
        activation_folder2 = os.path.join('./activations', layer2)
        if_exist(activation_folder2)

        ### 2. Load train data
        train_loader = get_data(args, dataset='cifar10', ifTrain=True)
        ## 2. Extract activations for futher look-up dictionary construction
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            _, activations = model(data)
            #print(len(activations))
            activation0 = activations[0].cpu().data.numpy()
            activation1 = activations[1].cpu().data.numpy()
            activation2 = activations[2].cpu().data.numpy()
            
            torch.save(activation0, os.path.join(activation_folder0, layer0 + '_'+str(batch_idx)+'.npy'))
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
        data0 = la.load_data(activation_folder0) # load patched data of input layer
        data1 = la.load_data(activation_folder1) # load patched data of layer1
        data2 = la.load_data(activation_folder2) # load patched data of layer2
        print('Construct tree input layer')
        tree0 = ht.construct(data0, n_cl, 1, max_depth)
        torch.save(tree0, 'tree_' + layer0)
        print('Construct tree layer 1')
        tree1 = ht.construct(data1, n_cl, density, max_depth)
        torch.save(tree1, 'tree_' + layer1)
        print('Construct tree layer 2')
        tree2 = ht.construct(data2, n_cl, density, max_depth)
        torch.save(tree2, 'tree_' + layer2)

else:
    if case_number == 1:
        layer_id = 1
        layer1 = 'layer' + str(layer_id)
        activation_folder1 = os.path.join('./activations', layer1)
        if_exist(activation_folder1)
        ### 2. Load train data
        train_loader = get_data(args, dataset='cifar10', ifTrain=True)
        ## 2. Extract activations for futher look-up dictionary construction
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            _, activations = model(data)
            activation1 = activations[1].cpu().data.numpy()
            torch.save(activation1, os.path.join(activation_folder1, layer1 + '_'+str(batch_idx)+'.npy'))
            if batch_idx>6:break

        # 3 Construct Look-up Dictionary
        # parameters for look-up dictionary construction
        n_cl = 10
        density = 30
        max_depth = 1

        # Load activations
        print('Load activations')
        data1 = la.load_data(activation_folder1) # load patched data of layer1

        print('Construct tree 1')
        tree1 = ht.construct(data1, n_cl, density, max_depth)
        torch.save(tree1, 'tree_' + layer1)
        
    elif case_number == 2:
        layer_id = 2
        layer2 = 'layer' + str(layer_id)
        activation_folder2 = os.path.join('./activations', layer2)
        if_exist(activation_folder2)
        ### 2. Load train data
        train_loader = get_data(args, dataset='cifar10', ifTrain=True)
        ## 2. Extract activations for futher look-up dictionary construction
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            _, activations = model(data)
            activation2 = activations[2].cpu().data.numpy()
            torch.save(activation2, os.path.join(activation_folder2, layer2+'_'+str(batch_idx)+'.npy'))
            if batch_idx>6:break

        # 3 Construct Look-up Dictionary
        # parameters for look-up dictionary construction
        n_cl = 10
        density = 30
        max_depth = 1

        # Load activations
        print('Load activations')
        data2 = la.load_data(activation_folder2) # load patched data of layer2

        print('Construct tree 2')
        tree2 = ht.construct(data2, n_cl, density, max_depth)
        torch.save(tree2, 'tree_' + layer2)
    
    elif case_number == 3:
        layer_id = 1
        layer1 = 'layer' + str(layer_id)
        activation_folder1 = os.path.join('./activations', layer1)
        if_exist(activation_folder1)

        layer_id = 2
        layer2 = 'layer' + str(layer_id)
        activation_folder2 = os.path.join('./activations', layer2)
        if_exist(activation_folder2)

        ### 2. Load train data
        train_loader = get_data(args, dataset='cifar10', ifTrain=True)
        ## 2. Extract activations for futher look-up dictionary construction
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            _, activations = model(data)
            activation1 = activations[1].cpu().data.numpy()
            activation2 = activations[2].cpu().data.numpy()
            
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
        data1 = la.load_data(activation_folder1) # load patched data of layer1
        data2 = la.load_data(activation_folder2) # load patched data of layer2

        print('Construct tree 1')
        tree1 = ht.construct(data1, n_cl, density, max_depth)
        torch.save(tree1, 'tree_' + layer1)
        print('Construct tree 2')
        tree2 = ht.construct(data2, n_cl, density, max_depth)
        torch.save(tree2, 'tree_' + layer2)

