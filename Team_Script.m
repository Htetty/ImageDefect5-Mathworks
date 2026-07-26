%% Image-Based Defect Detection for Manufacturing Inspection
% In this project, you will use MATLAB to create a virtual inspection station 
% that processes images of a single part type, produces defect evidence overlays 
% and measurable features, and classifies parts using a pretrained network via 
% transfer learning. You will then evaluate performance and robustness across 
% a repeatable test suite.
% 
% Use this Live Script as a template to get started on your project solution. 
% You may include helper functions at the bottom of this document or as separate 
% files. For a tutorial on how to use Live Scripts, check out this <https://www.youtube.com/watch?v=hUXTyPYRidM 
% video>.
% 
% *Suggested Tasks*
%% 
% # Explore and Organize the Image Dataset
% # Build a Single-Image Inspection Function
% # Train and Integrate an AI Classifier into your Single-Image Inspection Function
% # Evaluate Inspection System Performance
% # Test and Evaluate System Robustness
%% 
% 
%% Break Down the Problem
% In the space below, define the problem statement:
% 
% List the requirements, constraints, and success criteria for the project:
% 
% What is your proposed solution?
% 
% List the steps you will need to take to achieve your proposed solution:
% 
% What challenges do you face in the process of achieving the solution?
%% Task 1: Explore and Organize Image Dataset
% Your goal in this step is to define defect classes and labels (keep it small).
%% 
% * Load and inspect the images and labels from your chosen dataset (see the 
% project page on GitHub for recommended datasets) to understand the part types, 
% defect classes, and data balance
% * Store labels in a table loaded by |readtable(...)|
% * Manage images and labels with |imageDatastore(...)| and the |Labels| property 
% (or use the example workflow for setting up labeled image datastores).
% * Helpful reference: <https://www.mathworks.com/help/deeplearning/ug/create-and-explore-datastore-for-image-classification.html 
% Create and Explore Datastore for Image Classification>.
% * Choose 2-3 defect types (for FAIL) plus PASS

% Insert your code here (or make use of helper functions or additional .m or .mlx files and indicate where
% they can be found). Ensure your code is well-documented.


%fullfile function is to make directory by adding /
dataRoot = fullfile(pwd, 'bottle');
trainGoodDir = fullfile(dataRoot, 'train', 'good');
testDir = fullfile(dataRoot, 'test');
testSubfolders = {'good', 'broken_large', 'broken_small', 'contamination'};

filePaths = {};
labels = {};
defectType = {};

%dir() will list all files inside folder

%trainGoodFiles: have all good training data
%filePaths: store directory of good training data in format folder/name.
%labels: PASS or FAIL
%%get all good images from train/good
trainGoodFiles = dir(fullfile(trainGoodDir, '*.png'));
for i = 1: numel(trainGoodFiles)
    filePaths{end + 1, 1} = fullfile(trainGoodFiles(i).folder, trainGoodFiles(i).name);
    labels{end+1, 1} = 'PASS';
    defectType{end+1, 1} = 'good';
end


%%get all good images from test/good

%read files from each file in test folder
%store the file Paths and defect type and in labels mark it as PASS or FAIL
for k = 1: numel(testSubfolders)
    subfolder = testSubfolders{k};
    folderPath = fullfile(testDir, subfolder);
    files = dir(fullfile(folderPath, '*.png'));

    for i = 1: numel(files)
        filePaths{end+1, 1} = fullfile(files(i).folder, files(i).name);
        defectType{end+1, 1} = subfolder;
        if strcmp(subfolder, 'good')
            labels{end+1, 1} = 'PASS';
        else
            labels{end+1, 1} = 'FAIL';
        end
    end
end

     
% make a table with columns of FilePath | Label | DefectType
imageTable = table(filePaths, labels, defectType, ...
    'VariableNames', {'FilePath', 'Label', 'DefectType'});


writetable(imageTable, 'bottleImageLabels.csv', 'Delimiter', ',');
imageTable = readtable('bottleImageLabels.csv', 'Delimiter', ',');

imageTable.Label = categorical(imageTable.Label);
imageTable.DefectType = categorical(imageTable.DefectType);

imds = imageDatastore(imageTable.FilePath);
imds.Labels = imageTable.Label; 

%check balance
disp('PASS vs FAIL counts')
summary(imageTable.Label)
disp('Defect subtype summary')
summary(imageTable.DefectType)
%% Task 2: Build a Single-Image Inspection Function
% Your goal in this step is to create a function that analyzes and image to 
% identify and quantify suspected defects detected in the image. You're trying 
% to answer the question: "What in the image looks suspicious?"
% 
% To build this function you will need to:
% 
% *STEP 1. Standardize the images*
%% 
% * Read images with |imread(...)| or through |imageDatastore(...)|.
% * Standardize size with |imresize(...)| so metrics and model input are consistent.
% * Convert to grayscale with |rgb2gray(...)| as needed (e.g. classical processing 
% may rely on intensity values)
%% 
% *STEP 2. Preprocess the images*
%% 
% * Apply classical preprocessing, prioritizing stability over complexity. In 
% other words, apply only what you need to reduce sensitivity to normal station 
% variation. Suggested processing includes:
% * Lighting correction: |imflatfield(...)| (or background estimate with |imgaussfilt(...)| 
% and subtraction)
% * Contrast normalization: |adapthisteq(...)|
% * Denoising: |medfilt2(...)| or |imgaussfilt(...)|
% * Save intermediate results during development (e.g., |imshowpair(...)|) for 
% quick debugging.
%% 
% *STEP 3. _Optionally_, define a region of interest (ROI)*
% 
% If each image already contains a centered part with minimal clutter, skip 
% ROI and use the full image:
%% 
% * |roi = I;|
%% 
% If needed, define ROI fully automatically (no per-image manual selection), 
% by using, for example:
%% 
% * Fixed crop defined once (camera is consistent): indexing or |imcrop(...)| 
% OR
% * Auto-crop from the part mask: segment the part, compute |BoundingBox| using  
% |regionprops(...)|, then |imcrop(...)|.
%% 
% *STEP 4. Segment the image for an evidence overlay*
% 
% Your goal is to generate an evidence overlay and a few measurable numbers—not 
% a perfect segmentation. Generate a simple binary mask that highlights suspicious 
% regions in the image. Use a defect-appropriate recipe and keep it short:
%% 
% * Start with thresholding: |imbinarize(...)| or |adaptthresh(...)| + |imbinarize(...)|
% * Clean small specks: |bwareaopen(...)|
% * Bridge gaps (if appropriate): |imclose(...)|
% * Fill holes (only when it matches your goal (e.g., for a clean outer part 
% silhouette): |imfill(...,'holes')|
%% 
% Deliverable: |maskEvidence = defectEvidence(roi)|
% 
% *STEP 5. Extract a small set of interpretable metrics*
% 
% Your goal is to quantify the evidence for suspected defects, which supports 
% traceability and debugging in your workflow. These metrics should answer the 
% question: "What did the algorithm see, and how strong was the evidence?"
% 
% These measurements are mainly for traceability (inspection logs), debugging, 
% and sanity checks. Keep it to 3–4 metrics so it stays easy.
% 
% Recommended minimal metrics (works for many defect types):
%% 
% * |numComponents|: how many separate suspicious regions were found (use |bwconncomp(maskEvidence)| 
% and read |NumObjects|)
% * |maxArea|: size of the largest suspicious region (use |regionprops(maskEvidence,'Area')|)
% * |areaRatio|: fraction of the ROI flagged as suspicious (|nnz(maskEvidence) 
% / numel(maskEvidence)|)
% * _(optional)_ |edgeDensity|: if your evidence comes from edges (chips/scratches), 
% measure how "edgy" the ROI is (|nnz(edgeMask) / numel(edgeMask)|)
%% 
% *STEP 6. _Optionally_, implement a rule-based baseline.*
% 
% Implement 2–3 conservative rules so your system can always produce an explainable 
% fallback decision mechanism. This can also be a useful comparison with AI result.
%% 
% * Example: fail if |maxArea > Amax| or |numComponents > Nmax|
%% 
% Deliverable: |baselineDecision = decideRules(evidence)|

% Insert your code here (or make use of helper functions or additional .m or .mlx files and indicate where
% they can be found). Ensure your code is well-documented.

 % Create a datastore for image data
ds = imds;

% Demonstration
img = read(ds);
inspected_image = inspect_image(img);
% Primarily doing this part for comparison later

goodIdx = find(imageTable.DefectType == categorical("good"),  1, "first");
badIdx = find(imageTable.DefectType == categorical("contamination"),  1, "first");

goodImg = readimage(imds, goodIdx);
badImg = readimage(imds, badIdx);

montage({goodImg, badImg})

function baselineDecision = decideRules(evidence)
% decideRules Apply rule-based defect detection.
%
% Input:
%   evidence - Struct from defectEvidence.
%
% Output:
%   baselineDecision.fail    - True if a defect is found.
%   baselineDecision.reasons - Cell array of defect descriptions.
%% Thresholds

    contamStdMax = 8.5 / 255;      % Contamination threshold
    solidityBreakMax = 0.97; % Small break threshold
    solidityLargeMax = 0.90; % Large break threshold
    
    reasons = {};

% Detect irregular or broken bottle openings
    if evidence.solidity < solidityBreakMax

        if evidence.solidity < solidityLargeMax
            reasons{end+1} = sprintf( ...
                'Large irregularity: solidity %.3f below limit %.3f', ...
                evidence.solidity, solidityLargeMax);
        else
            reasons{end+1} = sprintf( ...
                'Small irregularity: solidity %.3f below limit %.3f', ...
                evidence.solidity, solidityBreakMax);
        end

    end

    baselineDecision.fail = ~isempty(reasons);
    baselineDecision.reasons = reasons;

end

function [maskEvidence, evidence] = defectEvidence(roi)
% defectEvidence Detect bottle-opening defects.
%
% Inputs:
%   roi - Grayscale or RGB bottle-opening image.
%
% Outputs:
%   maskEvidence - Binary mask of the opening.
%   evidence - Defect metrics.
%% Convert to grayscale

if size(roi,3) == 3
    roi = rgb2gray(roi);
end
roi = im2double(roi);
[h,w] = size(roi);
%% Find the bottle opening

darkMask = roi < 0.24;
darkMask = bwareaopen(darkMask,20);
darkMask = imfill(darkMask,'holes');

cc = bwconncomp(darkMask);

% Return default values if no opening is found
if cc.NumObjects == 0
    maskEvidence = false(h,w);
    evidence = struct( ...
        'maxArea',0,...
        'solidity',0,...
        'contamCoreMean',mean(roi(:)),...
        'contamCoreStd',std(roi(:)));
    return
end
%% Keep the largest region

blobSizes = cellfun(@numel,cc.PixelIdxList);
[~,largestIdx] = max(blobSizes);

opening = false(h,w);
opening(cc.PixelIdxList{largestIdx}) = true;
maskEvidence = opening;
%% Measure opening shape

stats = regionprops(opening,'Area','Solidity');
maxArea = stats.Area;
solidity = stats.Solidity;
%% Measure center brightness

[X,Y] = meshgrid(1:w,1:h);
centerX = (w+1)/2;
centerY = (h+1)/2;
radius = 0.22*min(h,w);

coreZone = (X-centerX).^2 + (Y-centerY).^2 <= radius^2;
coreZone = coreZone & opening;

pixels = roi(coreZone);
if isempty(pixels)
    pixels = roi(:);
end

contamCoreMean = mean(pixels);
contamCoreStd = std(pixels);
%% Store results

evidence = struct( ...
    'maxArea',maxArea,...
    'solidity',solidity,...
    'contamCoreMean',contamCoreMean,...
    'contamCoreStd',contamCoreStd);

end

function [inspected_image, evidence, baselineDecision] = inspect_image(image_input)

    % Step 2: Preprocess 
    img_BW = im2gray(image_input); % Turn image to black and white 
    img_BW_Cont = adapthisteq(img_BW, "ClipLimit", 0.01); % Contrast normalization
    img_BW_Cont_Deno = imgaussfilt(img_BW_Cont); % Denoising

    % From inspection, the bottle is already centered so roi can be
    % skipped.

    roi = img_BW_Cont_Deno;

    [maskEvidence, evidence] = defectEvidence(roi);
    baselineDecision = decideRules(evidence);

    % highlight irregularities as red 
    inspected_image = labeloverlay(image_input, maskEvidence, "Colormap", [1 0 0], "Transparency", 0.55);


end


[goodInspect, goodEvidence, goodDecision] = inspect_image(goodImg);
[badInspect, badEvidence, badDecision] = inspect_image(badImg);

montage({goodInspect, badInspect});

disp(goodEvidence);
disp(goodDecision);

disp(badEvidence);
disp(badDecision);

%sanity check that segmentation worked
predictedFail = false(height(imageTable), 1);

for i = 1:height(imageTable)
    currentImg = readimage(imds, i);

    [~, ~, decision] = inspect_image(currentImg);

    predictedFail(i) = decision.fail;
end

actualFail = imageTable.Label == categorical("FAIL");

correct = predictedFail == actualFail;

fprintf("Correct predictions: %d / %d\n", sum(correct), numel(correct));
fprintf("Accuracy: %.2f%%\n", 100 * mean(correct));

truePass  = sum(~actualFail & ~predictedFail);
trueFail  = sum( actualFail &  predictedFail);
falseFail = sum(~actualFail &  predictedFail);
falsePass = sum( actualFail & ~predictedFail);

fprintf("True PASS:  %d\n", truePass);
fprintf("True FAIL:  %d\n", trueFail);
fprintf("False FAIL: %d\n", falseFail);
fprintf("False PASS: %d\n", falsePass);
%% Task 3: Train and Integrate an AI Classifier into your Single-Image Inspection Function
% Your goal in this step is to use a MathWorks provided pretrained network (recommended: 
% |resnet18|) to classify each part as either a PASS or FAIL, that will be part 
% of the output of the function you began to build in the previous step.
%% 
% * Load the recommended starter network: use |imagePretrainedNetwork| and select  
% |resnet18| from the Deep Learning Toolbox.
% * Resize/augment input with |augmentedImageDatastore(...)| to match the network 
% input size.
% * Fine-tune the last layers for your classes (start with |PASS| vs |FAIL|), 
% using |trainNetwork(...)| or a guided workflow such as <https://www.mathworks.com/help/deeplearning/gs/get-started-with-transfer-learning.html 
% Get started with transfer learning> or <https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html 
% pretrained network workflows>.
% * The output of your AI classification model should be a predicted label (PASS/FAIL) 
% and a confidence score in that label
%% 
% Deliverable: |[aiLabel, aiScore] = classify(net, roiForNet)|

% Insert your code here (or make use of helper functions or additional .m or .mlx files and indicate where
% they can be found). Ensure your code is well-documented.

function roiForNet = prepareForNet(I)

I = imresize(I, [224 224]);
if size(I,3) == 1
    I = cat(3, I, I, I);
end
roiForNet = I; 


end

function [aiLabel, aiScore] = classifyImage(net, roiForNet)

scores = minibatchpredict(net, roiForNet);
classNames = ["FAIL", "PASS"];
[aiLabel, aiScore] = scores2label(scores, classNames);

end


% split data to train and test ds
rng(40); % seed value
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized'); % 80% train + 20% test
disp('Train set:')
countEachLabel(imdsTrain)
disp('Test set:')
countEachLabel(imdsTest)

% Class imbalance handled via WEIGHTED LOSS instead of oversampling/duplication -
% avoids the network seeing repeated FAIL images, penalizes FAIL mistakes more instead.
classNames = ["FAIL","PASS"];
counts = countcats(imdsTrain.Labels);           % [FAIL count, PASS count]
classWeights = sum(counts) ./ (numel(counts) * counts);
fprintf('Class weights -> FAIL: %.3f, PASS: %.3f\n', classWeights(1), classWeights(2));

% Resize inputs to match network input size
inputSize = [224 224 3];

% augmentation to prevent overfitting
augmenter = imageDataAugmenter('RandRotation',[-10 10], 'RandXReflection',true);
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation',augmenter);
augimdsTest = augmentedImageDatastore(inputSize, imdsTest);

% load and adapt the recommended model to 2 classes
net = imagePretrainedNetwork('resnet18', NumClasses = 2);
options = trainingOptions("adam", 'MaxEpochs',8, 'MiniBatchSize',16, 'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsTest, 'ValidationFrequency',10, 'Verbose', true, 'Plots','training-progress');

net = trainnet(augimdsTrain, net, @(Y,T) crossentropy(Y,T,classWeights,'WeightsFormat','C'), options);
% --------test it on an image ------
testIdx = 1;
I = imread(imageTable.FilePath{testIdx});
roiForNet = prepareForNet(I);
[aiLabel, aiScore] = classifyImage(net, roiForNet);

fprintf('Predicted: %s (confidence: %.2f%%)\n', aiLabel, aiScore*100);
fprintf('True label: %s\n', string(imageTable.Label(testIdx)));

% ---- test several FAIL Cases
testFailIdx = find(imdsTest.Labels == 'FAIL');
sampleFailIdx = testFailIdx(1:min(10, numel(testFailIdx)));

correct = 0;
for i = 1:numel(sampleFailIdx)
    I = readimage(imdsTest, sampleFailIdx(i));
    roiForNet = prepareForNet(I);
    [aiLabel, aiScore] = classifyImage(net, roiForNet);
    isCorrect = strcmp(string(aiLabel), 'FAIL');
    correct = correct + isCorrect;
    fprintf('FAIL sample %d -> predicted=%s (%.2f%%) %s\n', ...
        i, aiLabel, aiScore*100, string(isCorrect));
end
fprintf('\n%d/%d FAIL images correctly identified\n', correct, numel(sampleFailIdx));
%% Task 3 Checkpoint: Combine outputs into a hybrid inspection result
% Your image inspection function (e.g. |inspectPar(I)|) should combine the classical 
% evidence with the AI classification output.
% 
% In other words, your image inspection function should integrate both classical 
% evidence from image processing and an AI classification decision. The AI classifier 
% provides the main decision (PASS/FAIL) while the classical evidence extraction 
% provides interpretability, traceability, and sanity checks for that decision.
% 
% Your function should return:
%% 
% * |finalLabel| (from AI, in Task 3)
% * |confidenceScore| (from AI, in Task 3)
% * |evidenceOverlay| (from Step 4 in Task 2)
% * |evidenceMetrics| (from Step 5 in Task 2)
% * (optional) |baselineDecision| and a "disagreement flag" if optional rules 
% in Step 6 from Task 2 and AI output disagree
%% 
% Optionally, you can also choose to overlay these results on the image using 
% |insertShape(...)| / |insertText(...)|, and save rejects with |imwrite(...)|

% Insert your code here (or make use of helper functions or additional .m or .mlx files and indicate where
% they can be found). Ensure your code is well-documented.
%% --- Test inspectPart across the full test set, check accuracy ---
numTest = numel(imdsTest.Files);
predictedLabels = strings(numTest, 1);
trueLabels = strings(numTest, 1);
disagreementCount = 0;

for i = 1:numTest
    I = readimage(imdsTest, i);
    result = inspectPart(I, net, false);  % saveRejects=false, just checking accuracy
    % figure
    % imshow(result.annotatedImage)
    predictedLabels(i) = string(result.finalLabel);
    trueLabels(i) = string(imdsTest.Labels(i));

    if result.disagreementFlag
        disagreementCount = disagreementCount + 1;
    end
end


function result = inspectPart(I, net, saveRejects, rejectFolder)

if nargin < 4
    saveRejects = false;
end
if nargin < 5
    rejectFolder = 'rejected_parts';
end

[evidenceOverlay, evidence, baselineDecision] = inspect_image(I);
roiForNet = prepareForNet(I);
[aiLabel, aiScore] = classifyImage(net, roiForNet);

result.finalLabel = aiLabel;
result.confidenceScore = aiScore;
result.evidenceOverlay = evidenceOverlay;
result.evidenceMetrics = evidence;
result.baselineDecision = baselineDecision;
if baselineDecision.fail
    baselineLabel = "FAIL";
else
    baselineLabel = "PASS";
end

result.disagreementFlag = string(aiLabel) ~= baselineLabel;

%annotate the image with the results using insertText and insertShape

displayImage = imresize(I, [256 256]);
if size(displayImage, 3) == 1
    displayImage = cat(3, displayImage, displayImage, displayImage);
end

% drawing a color coded box by decision
% red -> FAIL, green -> PASS
if strcmp(string(aiLabel), 'PASS')
    boxColor = 'green';
else
    boxColor = 'red';
end

annotated = insertShape(displayImage, "rectangle", [2 2 252 252], "Color", boxColor, "LineWidth", 4);

labelText = sprintf('%s (%.1f%%)', aiLabel, aiScore*100);
annotated = insertText(annotated, [10 10], labelText, "FontSize",14, "BoxColor", boxColor, "TextColor", "white");

if result.disagreementFlag
    annotated = insertText(annotated, [10 40], 'DISAGREEMENT: check manually', 'FontSize', 14, 'BoxColor', 'yellow', 'TextColor', 'black');
end
result.annotatedImage = annotated;

% save rejected parts to disk

if saveRejects && strcmp(string(aiLabel), 'FAIL')
    if ~exist(rejectFolder, 'dir')
        mkdir(rejectFolder);
    end
    filename = fullfile(rejectFolder, ...
        sprintf('reject_%s.png', string(datetime('now', ...
        'Format','yyyy-MM-dd_HH-mm-ss-SSS'))));
    imwrite(annotated, filename);
    fprintf("Saved rejected pat to: %s\n", filename);
end
end
%% Task 4: Evaluate Inspection System Performance
% Now that you've built a hybrid inspection system that integrates classical 
% evidence with an AI decision, your goal in this step is to test the system on 
% a batch of images.
% 
% The results of this test should be summarized with a confusion matrix, report 
% yield, and defect rates across the entire batch of images.
% 
% Write a script (e.g. |runInspectionSuite.m|) that:
%% 
% * Splits data into train/test (e.g., |splitEachLabel(...)| or a fixed split)
% * Evaluates the trained AI model and trains the inspection system on a test 
% set and logs outcomes
% * Produces a confusion matrix with |confusionmat(...)| / |confusionchart(...)|
% * Summarizes yield and defect counts in tables and plots
% * Visualizes common failure cases (e.g. by creating a montage of common errors 
% using |montage(...)|)

% Here, it is recommended to create a separate runInspectionSuite.m file
% (or include as a local helper function) that combines the work you've done 
% so far here and additionally performs the steps outlined in the task description. 
% You can run it from here.
%% Task 5: Test and Evaluate System Robustness
% Your goal in this step is to assess how your system performs under simulated 
% variations. You can simulate image inspection station variation by altering 
% the lighting, blur, or noise in the images. Here are some suggested distortions 
% to apply:
%% 
% * Brightness/contrast: |imadjust(...)|
% * Blur: |imgaussfilt(...)|
% * Noise: |imnoise(...)|
%% 
% Then, re-run the evaluation. In other words, run |runInspectionSuite.m| again 
% with the same image set after applying distortions to the images.
% 
% Compare how performance changes across conditions by reporting how accuracy 
% and false-reject rates change under simulated variations.

% Insert your code here (or make use of helper functions or additional .m or .mlx files and indicate where
% they can be found). Ensure your code is well-documented.
%% Interpretation of Results
% Summarize your findings by interpreting their physical/engineering meaning:
% 
% What are some limitations of your work?
% 
% What are practical next steps?