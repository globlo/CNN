%transfer learning with VGG16 with Seg_train Dataset

% Load Training data
imds = imageDatastore('seg_train', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%Divide the data into training and validation data sets. Use 70% of the images for training and 30% for validation
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%Calculate the number of training class
numClasses = numel(categories(imdsTrain.Labels))

%load the pretrained VGG16
net = vgg16;

% Use analyzeNetwork to display an interactive visualization of the network architecture and detailed information about the network layers
analyzeNetwork(net)

%get the input size of the first layer

inputSize = net.Layers(1).InputSize

% Replace Final Layers
% The last three layers of the pretrained network net are configured for 1000 classes. 
% These three layers must be fine-tuned for the new classification problem.
% Extract all layers, except the last three, from the pretrained network.

layersTransfer = net.Layers(1:end-3);

% Transfer the layers to the new classification task by replacing 
% the last three layers with a fully connected layer, a softmax layer, and a classification output layer.
% Specify the options of the new fully connected layer according to the new data.
% Set the fully connected layer to have the same size as the number of classes in the new data.
% To learn faster in the new layers than in the transferred layers, increase the WeightLearnRateFactor and BiasLearnRateFactor values of the fully connected laye

numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Train Network
% The network requires input images of size 227-by-227-by-3, but the images in the image datastores have different sizes.
% Use an augmented image datastore to automatically resize the training images.
% Specify additional augmentation operations to perform on the training images: randomly flip the training images along the vertical axis, and randomly translate them up to 30 pixels horizontally and vertically.
% Data augmentation helps prevent the network from overfitting and memorizing the exact details of the training images.

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%To automatically resize the validation images without performing further data augmentation, use an augmented image datastore without specifying any additional preprocessing operations.
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%Specify the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',200, ...
    'MaxEpochs',3, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%Train the network that consists of the transferred and new layers. By default, trainNetwork uses a GPU if one is available, otherwise, it uses a CPU.
% Training on a GPU requires Parallel Computing Toolbox� and a supported GPU device

netTransfer = trainNetwork(augimdsTrain,layers,options);

%Load the Test data(this is a little different than previous code) Can you
%tell the diffence inoutput that it makes? 

imds_test_f = imageDatastore('seg_test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%remember, your test data size is also 32x32x3, But alexnet uses 227x227x3.
%So like the validation data , use agmented image datastore
Test_data = augmentedImageDatastore(inputSize(1:2),imds_test_f);

%classify the test data. Note that you can also generate scores for each
%classes

[YPred,scores] = classify(netTransfer,Test_data);

%create the Confusion matrix to calculate the accuracy

confMat = confusionmat(imds_test_f.Labels, YPred);
confMat = confMat./sum(confMat,2);
Accuracy= mean(diag(confMat))



