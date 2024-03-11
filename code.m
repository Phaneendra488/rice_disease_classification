clear all;

% Define paths to the directories containing images for each class
bacterialBlightDir = 'path\bacterial_leaf_blight';
leafSmutDir = 'path\leaf_smut';
brownspotDir = 'path\brown_spot';

% Count the number of images in each directory
numBacterialBlight = numel(dir(fullfile(bacterialBlightDir, '*.jpg')));
numLeafSmut = numel(dir(fullfile(leafSmutDir, '*.jpg')));
numBrownspot = numel(dir(fullfile(brownspotDir, '*.jpg')));

% Create labels for the images
bacterialBlightLabels = repmat({'bacterialBlight'}, numBacterialBlight, 1);
leafSmutLabels = repmat({'leafSmut'}, numLeafSmut, 1);
brownspotLabels = repmat({'brownspot'}, numBrownspot, 1);

% Create imageDatastore for each class with labels
bacterialBlightDS = imageDatastore(bacterialBlightDir, 'Labels', bacterialBlightLabels);
leafSmutDS = imageDatastore(leafSmutDir, 'Labels', leafSmutLabels);
brownspotDS = imageDatastore(brownspotDir, 'Labels', brownspotLabels);

% Combine all datasets into one
allDS = imageDatastore(cat(1, bacterialBlightDS.Files, leafSmutDS.Files, brownspotDS.Files), ...
    'Labels', categorical(cat(1, bacterialBlightDS.Labels, leafSmutDS.Labels, brownspotDS.Labels)));

% Display the number of images in each class
disp('Combined Dataset:')
countEachLabel(allDS)

% Shuffle the combined dataset
allDS = shuffle(allDS);

% Partition the dataset into training, validation, and testing sets
[trainDS, valTestDS] = splitEachLabel(allDS, 0.596, 'randomized'); % 59.6% for training
[valDS, testDS] = splitEachLabel(valTestDS, 0.139, 'randomized'); % 13.9% for validation, remaining for testing

% Preprocess images before feeding into the CNN model
[trainDS, trainLabels] = preprocessImages(trainDS);
[valDS, valLabels] = preprocessImages(valDS);
[testDS, testLabels] = preprocessImages(testDS);

% Define CNN architecture
layers = [
    imageInputLayer([224 224 3]) % Changed to 3 channels for RGB
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
];

% Specify training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Verbose', true);

% Compile the model
net = trainNetwork(trainDS, layers, options);

% Evaluate the model on validation set
Y_val_pred = classify(net, valDS);
accuracy_val = sum(Y_val_pred == valLabels) / numel(valLabels);

% Display validation accuracy
disp(['Validation Accuracy: ' num2str(accuracy_val)]);

% Evaluate the model on test set
Y_test_pred = classify(net, testDS);
accuracy_test = sum(Y_test_pred == testLabels) / numel(testLabels);

% Display test accuracy
disp(['Test Accuracy: ' num2str(accuracy_test)]);

% Save the model
save('my_model.mat', 'net');


% Load a test image
testImage = imread('test_image\path.jpg');

% Preprocess the test image
processedTestImage = preprocessImage(testImage);

% Classify the test image using the trained model
predictedLabel = classify(net, processedTestImage);

% Display the test image and predicted label
figure;
imshow(testImage);
title(['Predicted Label: ' char(predictedLabel)]);

% Define preprocessImages function
function [augmentedDS, labels] = preprocessImages(ds)
    % Define augmentation settings
    augmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandYReflection', true, ...
        'RandXTranslation', [-20 20], ...
        'RandYTranslation', [-20 20]);

    % Create augmented image datastore
    augmentedDS = augmentedImageDatastore([224 224], ds, 'DataAugmentation', augmenter);
    
    % Extract labels
    labels = ds.Labels;
end

% Define preprocessImage function
function processedImage = preprocessImage(img)
    % Resize the image to 224x224
    processedImage = imresize(img, [224 224]);
end
