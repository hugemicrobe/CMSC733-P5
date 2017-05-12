clear; clc

addpath('./ICP/');

Path = '../Dataset/SingleObject/'; 
SceneNum = 1;
SceneName = sprintf('%0.3d', SceneNum);

FramePath = dir([Path, 'scene_', SceneName, '/frames/*_rgb.png']);
DepthPath = dir([Path, 'scene_', SceneName, '/frames/*_depth.png']);

nPointClouds = max(length(FramePath), length(FramePath));

ptCloudList = cell(1, nPointClouds);
distributionList = cell(1, nPointClouds);
sampleList = cell(1, nPointClouds);

if isempty(gcp('nocreate'))
    parpool(8);
end

% parse image and depth files into point cloud objects
parfor i = 1:nPointClouds

%     if mod(i, 3) ~= 1
%         continue;
%     end

    framefile = [Path, 'scene_', SceneName, '/frames/image_', num2str(i-1),'_rgb.png'];
    depthfile = [Path, 'scene_', SceneName, '/frames/image_', num2str(i-1),'_depth.png'];
    
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
        [~, inliers_1] = fitPlaneRANSAC(pts, 500, 0.005); 
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
              
        inliers_1(~inliers_1) = (xyDists > threshold);
        inliers_1 = inliers_1 | pts(:, 3) > midZ + 4*stdZ | pts(:, 3) < midZ - 3*stdZ;
        ptCloudList{i} = pointCloud(pts(~inliers_1, :), 'Color', colors(~inliers_1, :));
    end            
end

% remove empty cells
ptCloudList = ptCloudList(~cellfun('isempty', ptCloudList));
distributionList = distributionList(~cellfun('isempty', distributionList));
sampleList = sampleList(~cellfun('isempty', sampleList));
nPointClouds = numel(ptCloudList);


tformList = cell(1, nPointClouds-1);
mseList = zeros(1, nPointClouds-1);

% compute transformation for pairs of pointclouds
parfor i = 2:nPointClouds
    gridSize = 0.1;
    fixed = ptCloudList{i-1};
    moving = ptCloudList{i};
%     fixed = pcdownsample(ptCloudList{i-1}, 'gridAverage', gridSize);
%     moving = pcdownsample(ptCloudList{i}, 'gridAverage', gridSize);
%      fixed = pcdownsample(ptCloudList{i-1}, 'random', 0.8);
%      moving = pcdownsample(ptCloudList{i}, 'random', 0.8);
      
    %[tformList{i-1}, ~, mseList(i-1)] = pcregrigid(moving, fixed, 'Metric', 'pointToPlane', 'Extrapolate', false, 'InlierRatio', 0.8);
    %[tformList{i-1}, mseList(i-1)] = efficientICP(moving.Location, fixed.Location, 'Extrapolate', false, 'InlierRatio', 0.9, 'MaxIterations', 60);
    [tformList{i-1}, mseList(i-1)] = icpMultiScale(moving.Location, fixed.Location);
    
    i
end    


% compute accumulated transform and merge point clouds
mergeSize = 0.015;
accumTform = affine3d();
ptCloudScene = ptCloudList{1};

for i = 2:nPointClouds    
    ptCloudCurrent = ptCloudList{i};

    accumTform = affine3d(tformList{i-1}.T * accumTform.T);
        
    ptCloudAligned = pctransform(ptCloudCurrent, accumTform);

    % Update the world scene.
    ptCloudScene = pcmerge(ptCloudScene, ptCloudAligned, mergeSize); 
        
    i
end 

ptCloudScene = pctransform(ptCloudScene, affine3d([1 0 0 0; 0 0 -1 0; 0 1 0 0; 0 0 0 1]));
figure;
showPointCloud(ptCloudScene);
