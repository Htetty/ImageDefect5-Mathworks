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
% 
% 
% List the requirements, constraints, and success criteria for the project:
% 
% What is your proposed solution?      
% 
% 
% 
% List the steps you will need to take to achieve your proposed solution:
% 
% 
% 
% What challenges do you face in the process of achieving the solution?
% 
% 
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
% question: “What did the algorithm see, and how strong was the evidence?”
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
% measure how “edgy” the ROI is (|nnz(edgeMask) / numel(edgeMask)|)
%% 
% *STEP 6. _Optionally_, implement a rule-based baseline.*
% 
% Implement 2–3 conservative rules so your system can always produce an explainable 
% fallback decision mechanism. This can also be a useful comparison with AI result.
%% 
% * Example: fail if |maxArea > Amax| or |numComponents > Nmax|
%% 
% Deliverable: |baselineDecision = decideRules(evidence)|





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
% * (optional) |baselineDecision| and a “disagreement flag” if optional rules 
% in Step 6 from Task 2 and AI output disagree
%% 
% Optionally, you can also choose to overlay these results on the image using 
% |insertShape(...)| / |insertText(...)|, and save rejects with |imwrite(...)|


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

conditions = {'Baseline', 'Brightness', 'Blur', 'Noise'}
robustnessResults = table();
for c = 1:numel(conditions)
condition = conditions{c}
predictedLabels = strings(numTest, 1);
trueLabels = strings(numTest,1);

for i = 1:numTest
I = readimage(imdsTest,i);

% apply the suggested distortions
switch condition
case 'Baseline'
Idist = I; %no distortion
case 'Brightness'
% Pixels with intensities below 0.3 become black, those above 0.7 become
% white, and values in between are stretched to simulate illumination changes.
% [] maps output to range between 0 and 1
Idist = imadjust(I, [], [], 0.7);
case 'Blur'
% The larger the sigma value, the blurrier the image becomes.
Idist = imgaussfilt(I, 3);
case 'Noise'
%% Add Gaussian noise with mean = 0 and variance = 0.01 (higher variance adds more noise, lower variance adds less) to simulate realistic sensor noise.
Idist = imnoise(I, 'gaussian', 0, 0.01);
end

result = inspectPart(Idist, referenceImage, net, false);
predictedLabels(i) = string(result.finalLabel);
trueLabels(i) = string(imdsTest.Labels(i));
end


correct = sum(predictedLabels == trueLabels);
accuracy = correct / numTest * 100;

passIdx = trueLabels == 'PASS';
failIdx = trueLabels == 'FAIL';
passAccuracy = sum(predictedLabels(passIdx) == trueLabels(passIdx)) / sum(passIdx) * 100;
failAccuracy = sum(predictedLabels(failIdx) == trueLabels(failIdx)) / sum(failIdx) * 100;

falseRejects = sum(trueLabels == 'PASS' & predictedLabels == 'FAIL');
falseAccepts = sum(trueLabels == 'FAIL' & predictedLabels == 'PASS');
falseRejectRate = falseRejects / sum(passIdx) * 100;
falseAcceptRate = falseAccepts / sum(failIdx) * 100;
robustnessResults = [robustnessResults; table(string(condition), accuracy, passAccuracy, failAccuracy, ...
falseRejects, falseRejectRate, falseAccepts, falseAcceptRate, ...
'VariableNames', {'Condition','Accuracy_pct','PassAccuracy_pct','FailAccuracy_pct', ...
'FalseRejects','FalseRejectRate_pct','FalseAccepts','FalseAcceptRate_pct'})];
end

disp('=== Robustness Summary Across Conditions ===')
disp(robustnessResults)

%% --- Visualize how accuracy and false-reject rate change across conditions ---
conditionOrder = categorical(robustnessResults.Condition, conditions);

figure
subplot(2,1,1)
bar(conditionOrder, robustnessResults.Accuracy_pct)
ylabel('Accuracy (%)')
title('Overall Accuracy Across Simulated Conditions')
grid on

subplot(2,1,2)
bar(conditionOrder, robustnessResults.FalseRejectRate_pct)
ylabel('False Reject Rate (%)')
title('False Reject Rate Across Simulated Conditions')
grid on

%% --- Optional: visualize an example of each distortion for the report ---
sampleI = readimage(imdsTest, 1);
figure
subplot(1,4,1), imshow(sampleI), title('Baseline')
subplot(1,4,2), imshow(imadjust(sampleI, [0.3 0.7], [])), title('Brightness')
subplot(1,4,3), imshow(imgaussfilt(sampleI, 3)), title('Blur')
subplot(1,4,4), imshow(imnoise(sampleI, 'gaussian', 0, 0.01)), title('Noise')
sgtitle('Example Distortions Applied for Robustness Testing')




%% Interpretation of Results
% Summarize your findings by interpreting their physical/engineering meaning:
% 
% 
% 
% What are some limitations of your work?
% 
% 
% 
% What are practical next steps?