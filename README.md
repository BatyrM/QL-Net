to train the model run QLNet.ipynb in Colab

There are 7 possible cases for quantization:

Case number = 1: Implement quantization for layer 1 only

Case number = 2: Implement quantization for layer 2 only

Case number = 3: Implement quantization for both layer 1 and 2

Case number = 4: Implement quantization for input layer only

Case number = 5: Implement quantization for input layer + layer 1

Case number = 6: Implement quantization for input layer + layer 2

Case number = 7: Implement quantization for input layer + layer 1 + layer 2

Inside the QLNet.ipynb there are 3 different parts:

1) Train baseline model.
2) Construct dictionary for a lookup table: 
In file training_parameters.py there is a parsing argument called case_number. This parameter is needed to distinguish between many cases as written above. Dependending on each case, dictionary is constructed only for needed layers. For ex., if case_number = 1, then dictionary is constructed only for layer 1. If case_number = 3, then dictionary is constructed for both layers 1 and 2.
3) Train quantized model. Use case_number parsing argument to choose one of the cases from above. Depending on that number the layers you want to approximate will be successfully quantized.
