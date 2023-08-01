%Deep Learning Example: Traning from scratch using CIFAR-10 Dataset
%Copyright 2017 The MathWorks, Inc.
%This example explores creating a convolutional neural network (CNN) from scratch. You will need to download images in order to run this example.
%Please see the file in this directory: DownloadCIFAR10.m Running this file will help you download CIFAR10 if you choose to use those images.

% Running this file will download CIFAR10 and place the images into a training folder and test folder in the current directory
% These will be used for the three demos in this folder. % Please note this will take a few minutes to run, but only needs to be run
% once.
% Copyright 2017 The MathWorks, Inc.
%% Download the CIFAR-10 dataset
% if ~exist('cifar-10-batches-mat','dir')
%     cifar10Dataset = 'cifar-10-matlab';
%     disp('Downloading 174MB CIFAR-10 dataset...');   
%     websave([cifar10Dataset,'.tar.gz'],...
%         ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
%     gunzip([cifar10Dataset,'.tar.gz'])
%     delete([cifar10Dataset,'.tar.gz'])
%     untar([cifar10Dataset,'.tar'])
%     delete([cifar10Dataset,'.tar'])
% end    
   
%% Prepare the CIFAR-10 dataset
% if ~exist('cifar10Train','dir')
%     disp('Saving the Images in folders. This might take some time...');    
%     saveCIFAR10AsFolderOfImages('cifar-10-batches-mat', pwd, true);
% end

%Load training data

% Please note: these are 4 of the 10 categories available
% Feel free to choose which ever you like best!
categories = {'buildings','forest','glacier','mountain','sea','street'};
%% ,'horse','bird','airplane','automobile','ship','truck'

rootFolder = 'seg_train';
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');
% img = imresize(imds,[32 32])


imageSize = [32, 32, 3];
imds.ReadFcn = @(filename)imresize(imread(filename), imageSize(1:2));


%Define Layers
%Training from scratch gives you a lot of freedom to explore the architecture. Take a look at this architecture and see how you might want to alter it: for example, how would you add another convolutional layer?




varSize = 32;
conv1 = convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
conv1.Weights = gpuArray(single(randn([5 5 3 32])*0.0001));


% conv2 = convolution2dLayer(5,32,'Padding',2, 'BiasLearnRateFactor',2);
% conv2.Weights = single(randn([5 5 32 32])*0.0001);
% 
% conv3 = convolution2dLayer(3,128,'Padding',2, 'BiasLearnRateFactor',2);
% conv3.Weights = single(randn([3 3 128 128])*0.0001);



fc1 = fullyConnectedLayer(128,'BiasLearnRateFactor',2);
fc1.Weights = gpuArray(single(randn([128 512])*0.1));
fc2 = fullyConnectedLayer(6,'BiasLearnRateFactor',2);
fc2.Weights = gpuArray(single(randn([6 128])*0.1));

layers = [
    imageInputLayer([varSize varSize 3]);
    conv1;
    maxPooling2dLayer(3,'Stride',1);
    reluLayer();
 
    convolution2dLayer(3,64,'Padding',"same",'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);

    convolution2dLayer(5,128,'Padding',"same",'BiasLearnRateFactor',2);
    reluLayer();
    maxPooling2dLayer(3,'Stride',2);
    

    convolution2dLayer(3,256,'Padding',"same",'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);

    convolution2dLayer(5,512,'Padding',3,'BiasLearnRateFactor',2);
    reluLayer();
    maxPooling2dLayer(3,'Stride',2);

    % convolution2dLayer(3,1024,'Padding',3,'BiasLearnRateFactor',2);
    % reluLayer();
    % maxPooling2dLayer(3,'Stride',2);

    

    fc1;
    reluLayer();
    fc2;
    softmaxLayer()
    classificationLayer()];

%Define training options
%The training options is another parameter that can significantly increase or decrease the accuracy of the network. Try altering some of these values and see what happens to the overall accuracy of the network. 
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 3, ...
    'MiniBatchSize', 300, ...
    'Verbose', true,...
    'plots','training-progress');

%Train! 

%This is where the training happens. This can take a few minutes or longer depending on your hardware. Training on a GPU is recommended.
[net, info] = trainNetwork(imds, layers, opts);

% Load test data

rootFolder = 'seg_test';
imds_test = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

imageSize = [32, 32, 3];
imds_test.ReadFcn = @(filename)imresize(imread(filename), imageSize(1:2));



%Do it all at once
%This section of code will run through all the test data and compare the predicted labels with the actual labels. This will give a feel for how the network is doing overall with one average prediction value.
% This could take a while if you are not using a GPU
labels = classify(net, imds_test);
confMat = confusionmat(imds_test.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))
