clear; clc

Path = '../Dataset/SingleObject/'; 
SceneNum = 1;
SceneName = sprintf('%0.3d', SceneNum);

FramePath = dir([Path, 'scene_', SceneName, '/frames/*_rgb.png']);
DepthPath = dir([Path, 'scene_', SceneName, '/frames/*_depth.png']);

nPointClouds = length(FramePath);

ptCloudList = cell(1, nPointClouds);
tformList = cell(1, nPointClouds-1);


if isempty(gcp('nocreate'))
    parpool(8);
end


% parse image and depth files into point cloud objects
for i = 1:nPointClouds
    I = imread([Path, 'scene_', SceneName, '/frames/image_', num2str(i-1),'_rgb.png']);
    ID = imread([Path, 'scene_', SceneName, '/frames/image_', num2str(i-1),'_depth.png']);
    
    % [pcx, pcy, pcz, r, g, b, D_, X, Y, validInd]
    [pcx, pcy, pcz, r, g, b] = depthToCloud_full_RGB(ID, I, './params/calib_xtion.mat');
    
    ptCloudList{i} = pointCloud([pcx pcy pcz], 'Color', uint8([r g b]));
end

% compute transformation for pairs of pointclouds
parfor i = 2:nPointClouds
    gridSize = 0.1;
    fixed = pcdownsample(ptCloudList{i-1}, 'gridAverage', gridSize);
    moving = pcdownsample(ptCloudList{i}, 'gridAverage', gridSize);

    tformList{i-1} = pcregrigid(moving, fixed, 'Metric', 'pointToPlane', 'Extrapolate', true);
    i
end    


% compute accumulated transform and merge point clouds
mergeSize = 0.015;
accumTform = affine3d(eye(4));
ptCloudScene = ptCloudList{1};

for i = 2:3
    ptCloudCurrent = ptCloudList{i};

    accumTform = affine3d(tformList{i-1}.T * accumTform.T);
    ptCloudAligned = pctransform(ptCloudCurrent, accumTform);

    % Update the world scene.
    ptCloudScene = pcmerge(ptCloudScene, ptCloudAligned, mergeSize);
    i
end 

ptCloudScene = pctransform(ptCloudScene, affine3d([1 0 0 0; 0 0 -1 0; 0 1 0 0; 0 0 0 1]));
showPointCloud(ptCloudScene);
