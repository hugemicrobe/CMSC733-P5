function [tfList, mseList] = pcListTf(pcList)

n = numel(pcList);

tfList = cell(1, n-1);
mseList = zeros(1, n-1);

% compute transformation for pairs of pointclouds
parfor i = 2:n
    
    fixed = pcList{i-1};
    moving = pcList{i};
    
    %[tfList{i-1}, ~, mseList(i-1)] = pcregrigid(moving, fixed, 'Metric', 'pointToPlane', 'Extrapolate', false, 'InlierRatio', 0.8);
    [tfList{i-1}, mseList(i-1)] = efficientICP(moving.Location, fixed.Location, 'Extrapolate', false, 'InlierRatio', 0.9, 'MaxIterations', 60);
    %[tfList{i-1}, mseList(i-1)] = icpMultiScale(moving.Location, fixed.Location);
    
    i
end


end