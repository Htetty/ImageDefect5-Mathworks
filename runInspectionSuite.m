%% Task 4: Evaluate Inspection System Performance

actualLabels = categorical(trueLabels, ["FAIL", "PASS"]);
modelLabels = categorical(predictedLabels, ["FAIL", "PASS"]);

% confusion matrix
figure;
confusionchart(actualLabels, modelLabels);
title("Inspection System Confusion Matrix");

% good and defect rates
numImages = numel(modelLabels);
numPass = sum(modelLabels == "PASS");
numFail = sum(modelLabels == "FAIL");

goodRate = 100 * numPass / numImages;
defectRate = 100 * numFail / numImages;

summaryTable = table( ...
    numImages, ...
    numPass, ...
    numFail, ...
    goodRate, ...
    defectRate, ...
    'VariableNames', ...
    {'TotalImages', 'PredictedPASS', 'PredictedFAIL', ...
     'YieldPercent', 'DefectRatePercent'});

disp(summaryTable);

% plot PASS and FAIL counts
figure;
bar(categorical(["PASS", "FAIL"]), [numPass, numFail]);
ylabel("Number of Images");
title("Inspection Outcomes");

% show misclassified images
errorIdx = find(modelLabels ~= actualLabels);

if isempty(errorIdx)
    disp("No misclassified images were found.");
else
    numToShow = min(8, numel(errorIdx));
    errorImages = cell(1, numToShow);

    for i = 1:numToShow
        idx = errorIdx(i);
        I = readimage(imdsTest, idx);
    
        labelText = sprintf( ...
            "True: %s | Predicted: %s", ...
            string(actualLabels(idx)), ...
            string(modelLabels(idx)));
    
        errorImages{i} = insertText( ...
            I, ...
            [10 10], ...
            labelText, ...
            "FontSize", 18, ...
            "BoxColor", "black", ...
            "TextColor", "white");
    end

    figure;
    montage(errorImages);
    title("Common Misclassified Images");
end