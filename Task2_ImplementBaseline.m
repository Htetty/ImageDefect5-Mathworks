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

%[appendix]{"version":"1.0"}
%---
