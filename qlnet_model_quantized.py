import torch.nn.functional as F
import torch.nn as nn
from quantizer_gpu import Quantizer

class BS_Net(nn.Module):
    def __init__(self):
        super(BS_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.fc1 = nn.Linear(16*40, 50) # Fully Connected Layers
        self.fc2 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(40)
        self.bn3 = nn.BatchNorm1d(50)
        self.activation = nn.ReLU()

    def forward(self, x, n=0, tree=None):    
        if n==1:
            tree0, _ = tree
            x = self.quantize_activation(x, True, tree0[n-1], 'lookup_table', True)
        elif n==2:
            tree0, _, _ = tree
            x = self.quantize_activation(x, True, tree0[n-2], 'lookup_table', True)
        
        layer1 = self.activation(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        
        if n==1:
            _, tree1 = tree
            layer1 = self.quantize_activation(layer1, True, tree1[n-1], 'lookup_table', False)

        layer2 = self.activation(F.max_pool2d(self.bn2(self.conv2(layer1)), 2))
        
        if n==2:
            _, tree1, tree2 = tree
            layer1 = self.quantize_activation(layer1, True, tree1[n-2], 'lookup_table', False)
            layer2 = self.quantize_activation(layer2, True, tree2[n-2], 'lookup_table', False)

        out = layer2.view(-1, 16*40) # flatten input to feed it to fully connected layer
        out = self.activation(self.bn3(self.fc1(out)))
        out = F.dropout(out, p=0.25)
        out = self.fc2(out)

        return out, [x, layer1, layer2]

    def quantize_activation(self, input, ifTraining, tree, lookup_table, isInput):
        # return Quantizer(ifQuantizing, ifTraining, tree, lookup_table).apply(input)
        return Quantizer().apply(input, ifTraining, tree, lookup_table, isInput)
