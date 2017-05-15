clear; clc;

% 0:Single 1:Multiple
mode = 1;
SceneNum = 50;
SceneName = sprintf('%0.3d', SceneNum);

if mode == 0
    saveName = 'single_scene_'; 
else
    saveName = 'multiple_scene_';
end 

load(['../Results/', saveName, SceneName, '.mat'], 'ptCloudList', 'tableModelList', 'wallModelList');

% downsample frames
%ptCloudList = ptCloudList([1:10:numel(ptCloudList)-1, numel(ptCloudList)]);

% merge point clouds
[tformList, mseList] = pcListTf(ptCloudList);

ptCloudScene = pcListMerge(ptCloudList, tformList, mseList);
notTable = removePlane(ptCloudScene.Location, tableModelList{1}, 0.005);
notWall = removePlane(ptCloudScene.Location, wallModelList{1}, 0.02);
ptCloudScene = select(ptCloudScene, find(notTable & notWall));

% =========================================================================
% convert and save in pcd file
% =========================================================================
locData = ptCloudScene.Location';

% normalize to [-1, 1]
minValue = min(locData(:));
maxValue = max(locData(:));

locData = (locData - minValue) ./ (maxValue - minValue);
locData = locData*2 - 1;

rgbData = double(ptCloudScene.Color') / 255;
savepcd(['../Results/pcd/', saveName, SceneName, '.pcd'], [locData; rgbData]);

% visualize result
pp = pctransform(ptCloudScene, affine3d([1 0 0 0; 0 0 -1 0; 0 1 0 0; 0 0 0 1]));
figure;
showPointCloud(pp);