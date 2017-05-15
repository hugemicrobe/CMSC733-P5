clear; clc

% 0:Single 1:Multiple
mode = 0;
SceneNum = 23;


addpath('./ICP/');


if mode == 0
    Path = '../Dataset/SingleObject/';
    saveName = 'single_scene_'; 
else
    Path = '../Dataset/MultipleObjects/';
    saveName = 'multiple_scene_';
end    

SceneName = sprintf('%0.3d', SceneNum);

FramePath = dir([Path, 'scene_', SceneName, '/frames/*_rgb.png']);
DepthPath = dir([Path, 'scene_', SceneName, '/frames/*_depth.png']);

frameNum = zeros(1, length(FramePath));
for i = 1:length(FramePath)
    frameName = FramePath(i).name;
    tokens = strsplit(frameName, '_');
    frameNum(i) = str2num(tokens{2});
end

depthNum = zeros(1, length(DepthPath));
for i = 1:length(DepthPath)
    depthName = DepthPath(i).name;
    tokens = strsplit(depthName, '_');
    depthNum(i) = str2num(tokens{2}); 
end

pcNumbers = intersect(frameNum, depthNum);

nPointClouds = numel(pcNumbers);

ptCloudList = cell(1, nPointClouds);
tableModelList = cell(1, nPointClouds);
wallModelList = cell(1, nPointClouds);
distributionList = cell(1, nPointClouds);
sampleList = cell(1, nPointClouds);

if isempty(gcp('nocreate'))
    parpool(8);
end

% parse image and depth files into point cloud objects
parfor i = 1:nPointClouds

    if mod(i, 2) ~= 1
        continue;
    end

    framefile = [Path, 'scene_', SceneName, '/frames/frame_', num2str(pcNumbers(i)),'_rgb.png'];
    depthfile = [Path, 'scene_', SceneName, '/frames/frame_', num2str(pcNumbers(i)),'_depth.png'];
    
    if exist(framefile, 'file') && exist(depthfile, 'file')
    
        I = imread(framefile);
        ID = imread(depthfile);
        
        [pcx, pcy, pcz, r, g, b] = depthToCloud_full_RGB(ID, I, './params/calib_xtion.mat');
        
        pts = [pcx, pcy, pcz];
        colors = uint8([r g b]);

        % =================================================================
        % remove as much table as possible using larger threshold (0.05)
        % and then estimate rough range of depths of the objects
        % =================================================================
        [~, inliers_1] = fitPlaneRANSAC(pts, 500, 0.05); 
        [~, inliers_2] = fitPlaneRANSAC(pts(~inliers_1, :), 500, 0.05);
        inliers_1(~inliers_1) = inliers_2;
        
        % compute distance to the origin in x-y coordinates
        xyDists = sqrt(sum(pts(~inliers_1, 1:2).^2, 2));
        
        % estimate kernel density from 1D samples
        [f, xi] = ksdensity(xyDists);
        [~, localMaxInd] = findpeaks(f);
        [~, localMinInd] = findpeaks(-f);
        ind = find((xi > xi(localMaxInd(1))) & (f < 1e-5), 1);
        
        if ~isempty(localMinInd)
            threshold = xi(min(ind, localMinInd(1)));
        else
            threshold = xi(ind);
        end    
        distributionList{i} = f; sampleList{i} = xi;        
        inliers_1(~inliers_1) = (xyDists > threshold);

        % estimate rough range of the object in Z direction (depth)
        midZ = median(pts(~inliers_1, 3), 1);
        stdZ = std(pts(~inliers_1, 3), 1);
             
        % =================================================================
        % keep more points of the objects using smaller threshold (0.005)
        % and remove points that are not in the estimated depth
        % =================================================================
        [tableModelList{i}, inliers_1] = fitPlaneRANSAC(pts, 500, 0.05); 
        [wallModelList{i}, inliers_2] = fitPlaneRANSAC(pts(~inliers_1, :), 500, 0.05);
        inliers_1(~inliers_1) = inliers_2;
        
        % compute distance to the origin in x-y coordinates
        xyDists = sqrt(sum(pts(~inliers_1, 1:2).^2, 2));
        
        % estimate kernel density from 1D samples
        [f, xi] = ksdensity(xyDists);
        [~, localMaxInd] = findpeaks(f);
        [~, localMinInd] = findpeaks(-f);
        ind = find((xi > xi(localMaxInd(1))) & (f < 1e-5), 1);
        
        if ~isempty(localMinInd)
            threshold = xi(min(ind, localMinInd(1)));
        else
            threshold = xi(ind);
        end    
              
        inliers_1(~inliers_1) = (xyDists > threshold);
        inliers_1 = inliers_1 | pts(:, 3) > midZ + 4*stdZ | pts(:, 3) < midZ - 3*stdZ;
        ptCloudList{i} = pointCloud(pts(~inliers_1, :), 'Color', colors(~inliers_1, :));
    end            
end

% remove empty cells
ptCloudList = ptCloudList(~cellfun('isempty', ptCloudList));
tableModelList = tableModelList(~cellfun('isempty', tableModelList));
wallModelList = wallModelList(~cellfun('isempty', wallModelList));
distributionList = distributionList(~cellfun('isempty', distributionList));
sampleList = sampleList(~cellfun('isempty', sampleList));

save(['../Results/', saveName, SceneName, '.mat'], 'ptCloudList', 'tableModelList', 'wallModelList');
