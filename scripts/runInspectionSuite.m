%% Task 4: Evaluate Inspection System Performance

actualLabels = categorical(trueLabels, ["FAIL", "PASS"]);
modelLabels = categorical(predictedLabels, ["FAIL", "PASS"]);

% confusion matrix
figConfusion = figure;
confusionchart(actualLabels, modelLabels);
title("Inspection System Confusion Matrix");

saveProjectFigure(figConfusion, "task4_confusion_matrix.png");

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
saveProjectTable(summaryTable, "task4_summary.csv");

% plot PASS and FAIL counts
figOutcomes = figure;
bar(categorical(["PASS", "FAIL"]), [numPass, numFail]);
ylabel("Number of Images");
title("Inspection Outcomes");

saveProjectFigure(figOutcomes, "task4_inspection_outcomes.png");

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

    figErrors = figure;
    montage(errorImages);
    title("Common Misclassified Images");
    saveProjectFigure(figErrors, "task4_misclassified_images.png");
end

function saveProjectFigure(figHandle, fileName)

outputFolder = fullfile(pwd, "project_results");

if ~exist(outputFolder, "dir")
    mkdir(outputFolder);
end

exportgraphics( ...
    figHandle, ...
    fullfile(outputFolder, fileName), ...
    "Resolution", 300);

end

function saveProjectTable(tableData, fileName)

outputFolder = fullfile(pwd, "project_results");

if ~exist(outputFolder, "dir")
    mkdir(outputFolder);
end

writetable(tableData, fullfile(outputFolder, fileName));

end