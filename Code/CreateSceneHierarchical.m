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

% merge point clouds
n = numel(ptCloudList);

nEpoch = ceil(n / 10);

subScene = cell(1, nEpoch);
parfor k = 1:nEpoch
    ii = 1+(k-1)*10;
    jj = min(10*k, n);

    [tformList, mseList] = pcListTf(ptCloudList(ii:jj));
    s = pcListMerge(ptCloudList(ii:jj), tformList, mseList);
    notTable = removePlane(s.Location, tableModelList{ii}, 0.05);
    subScene{k} = select(s, find(notTable));
end

[tformList, mseList] = pcListTf(subScene);

ptCloudScene = pcListMerge(subScene, tformList, mseList);
notTable = removePlane(ptCloudScene.Location, tableModelList{1}, 0.005);
notWall = removePlane(ptCloudScene.Location, wallModelList{1}, 0.02);
ptCloudScene = select(ptCloudScene, find(notTable & notWall));

% convert and save in pcd file
locData = ptCloudScene.Location';
rgbData = double(ptCloudScene.Color') / 255;
savepcd(['../Results/pcd_hierarchical/', saveName, SceneName, '.pcd'], [locData; rgbData]);

% visualize result
pp = pctransform(ptCloudScene, affine3d([1 0 0 0; 0 0 -1 0; 0 1 0 0; 0 0 0 1]));
figure;
showPointCloud(pp);