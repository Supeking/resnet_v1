# resnet_v1
This is a code that implements resnet_v1 based on tensorflow. At present, GitHub has a lot of RESNET code, and even tensorlfow and keras have included RESNET modules. We can easily call these modules which have been implemented by others for a pre training. But for a new structured network, it is inconvenient to call the packaged modules directly. I implemented the whole resnet_v1 process with the simplest code, which is more convenient to adapt to a new network when the pre-training weights can be used.
# Instructions
Resnet18_v1 and resnet50_v1 are the implementation of the 18 and 50 layers of RESNET, respectively.

Resnet_50_101_152_v1 implements resnet of 50, 101 and 152 layers at the same time. This code is mainly convenient for training classification tasks directly using resnet. Just by changing the CFG parameters, you can switch freely in the 50, 102 and 152 layers structures.

The weights folder mainly includes the pre-training model of tensorflow version of RESNET. You can download the model by opening the links in the URL file.

The pp_mean.mat is the average pixel of the dataset. In training and testing, the purpose is to de centralization.

This code contains a simple test section to verify the correctness of the model. As for the training part, we can make simple modifications according to the code.
