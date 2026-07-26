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
    contamStdMax = 8.5;      % Contamination threshold
    solidityBreakMax = 0.97; % Small break threshold
    solidityLargeMax = 0.90; % Large break threshold

    reasons = {};

    %% Check contamination first
    isContaminated = evidence.contamCoreStd > contamStdMax;

    if isContaminated
        reasons{end+1} = sprintf( ...
            'Contamination: core brightness variation %.1f exceeds limit %.1f', ...
            evidence.contamCoreStd, contamStdMax);
    end

    %% Check for breaks
    if ~isContaminated && evidence.solidity < solidityBreakMax

        if evidence.solidity < solidityLargeMax
            reasons{end+1} = sprintf( ...
                'Broken (large): solidity %.3f below limit %.3f', ...
                evidence.solidity, solidityLargeMax);
        else
            reasons{end+1} = sprintf( ...
                'Broken (small): solidity %.3f below limit %.3f', ...
                evidence.solidity, solidityBreakMax);
        end

    end

    %% Store results
    baselineDecision.fail = ~isempty(reasons);
    baselineDecision.reasons = reasons;

end

%[appendix]{"version":"1.0"}
%---
