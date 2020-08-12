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

    '''
    
        Case 1: Implement quantization for the 1st layer 
        Case 2: Implement quantization for the 2nd layer
        Case 3: Implement quantization for both layer 1 and 2
        Case 4: Implement quantization for input layer only
        Case 5: Implement quantization for input layer + layer 1
        Case 6: Implement quantization for input layer + layer 2
        Case 7: Implement quantization for input layer + layer 1 + layer 2
    
    '''
    def forward(self, x, n=0, tree=None):    
        if n >= 4:
            if n == 4:
                x = self.quantize_activation(x, True, tree[n-4], 'lookup_table', True)
            elif n == 5:
                tree_input_layer, _ = tree
                x = self.quantize_activation(x, True, tree_input_layer[n-5], 'lookup_table', True)
            elif n == 6:
                tree_input_layer, _ = tree
                x = self.quantize_activation(x, True, tree_input_layer[n-6], 'lookup_table', True)
            elif n == 7:
                tree_input_layer, _, _ = tree
                x = self.quantize_activation(x, True, tree_input_layer[n-7], 'lookup_table', True)

        layer1 = self.activation(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        
        if n == 1:
            layer1 = self.quantize_activation(layer1, True, tree[n-1], 'lookup_table', False)

        elif n == 5:
            _, tree_layer1 = tree
            layer1 = self.quantize_activation(layer1, True, tree_layer1[n-5], 'lookup_table', False)

        layer2 = self.activation(F.max_pool2d(self.bn2(self.conv2(layer1)), 2))
        
        if n == 2:
            layer2 = self.quantize_activation(layer2, True, tree[n-2], 'lookup_table', False)
        elif n == 3:
            tree_layer1, tree_layer2 = tree
            layer1 = self.quantize_activation(layer1, True, tree_layer1[n-3], 'lookup_table', False)
            layer2 = self.quantize_activation(layer2, True, tree_layer2[n-3], 'lookup_table', False)

        elif n == 6:
            _, tree_layer2 = tree
            layer2 = self.quantize_activation(layer2, True, tree_layer2[n-6], 'lookup_table', False)
        
        elif n == 7:
            _, tree_layer1, tree_layer2 = tree
            layer1 = self.quantize_activation(layer1, True, tree_layer1[n-7], 'lookup_table', False)
            layer2 = self.quantize_activation(layer2, True, tree_layer2[n-7], 'lookup_table', False)

        out = layer2.view(-1, 16*40) # flatten input to feed it to fully connected layer
        out = self.activation(self.bn3(self.fc1(out)))
        out = F.dropout(out, p=0.25)
        out = self.fc2(out)

        return out, [x, layer1, layer2]
        
    def quantize_activation(self, input, ifTraining, tree, lookup_table, isInput):
        # return Quantizer(ifQuantizing, ifTraining, tree, lookup_table).apply(input)
        return Quantizer().apply(input, ifTraining, tree, lookup_table, isInput)
