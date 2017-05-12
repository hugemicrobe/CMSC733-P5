function [tform, mse] = icpMultiScale(srcPts, modelPts)

n = size(srcPts, 1);
doScaling = false;


% Scaling
if doScaling
    srcMean = mean(srcPts, 1);
    modelMean = mean(modelPts, 1);
    avgMean = 0.5 * (srcMean + modelMean);
    srcPts = bsxfun(@minus, srcPts, srcMean);
    modelPts = bsxfun(@minus, modelPts, modelMean);
    
    avgDist = 0.5 * (sum(sqrt(sum(srcPts.^2, 2))) + sum(sqrt(sum(modelPts.^2, 2))));
    
    maxDist = max([sqrt(sum(srcPts.^2, 2)); sqrt(sum(modelPts.^2, 2))]);

    scale = 1 / maxDist;
    srcPts = srcPts * scale;
    modelPts = modelPts * scale;
end

% compute normals for model point cloud
modelNormals = ptCloudNormal(modelPts, 5);


% Apply multi-scale ICP
maxIter = 200;
nScales = 4;
totalTform = affine3d();

for i = nScales-1:-1:0
    nPts = round(n / (2^i));
    iters = ceil(maxIter / (2^i));

    % transform points
    srcPtsMoved = rigidTransform(srcPts, totalTform.T(1:3, 1:3)', totalTform.T(4, 1:3)');

    % sampling
    srcPtsMoved = uniformSample(srcPtsMoved, nPts);
    
%     [tf, mse] = efficientICP(srcPtsMoved, modelPts, 'NormalsData', modelNormals, ...
%                             'MaxIterations', iters, 'Extrapolate', true, 'InlierRatio', 0.75);
    [tf, ~, mse] = pcregrigid(pointCloud(srcPtsMoved), pointCloud(modelPts, 'Normal', modelNormals), ...
                                                                           'Metric', 'pointToPlane', ...
                                                                           'MaxIterations', iters, ...
                                                                           'Extrapolate', false, ...
                                                                           'InlierRatio', 0.9);
    totalTform = affine3d(totalTform.T * tf.T);

end


if doScaling
    totalTform.T(4, 1:3) = totalTform.T(4, 1:3) / scale + modelMean - srcMean * totalTform.T(1:3, 1:3);
end

tform = totalTform;


if ~(isRigidTransform(tform))
    error('results in non-rigid transformation');
end


end


function normal = ptCloudNormal(xyzPoints, K)

normal = zeros(size(xyzPoints, 2), size(xyzPoints, 1), 'like', xyzPoints);
xyzPoints = single(xyzPoints);

indices = multiQueryKNNSearchImpl(pointCloud(xyzPoints), xyzPoints, K);

numNeighbors = K-1;

for n = 1:size(xyzPoints, 1)
    
    % Get all neighbor points (not including itself)
    x = xyzPoints(indices(2:end, n), :);
    % Compute the centroid
    xMean = sum(x, 1)/numNeighbors;
    % Compute the offset to the centroid
    y = x - repmat(xMean, numNeighbors, 1);
    % Compute the covariance
    P =  y' * y;
    % Choose the eigenvector with the smallest eigenvalue
    [V, D] = eig(P, 'vector');
    [~, idx] = min(D);
    normal(:, n) = V(:, idx);
end
normal = normal';

end

%==========================================================================
% Inputs to rigidTransform are in pre-multiplication format
%==========================================================================
function B = rigidTransform(A, R, T)
    B = A * R';
    B(:,1) = B(:,1) + T(1);
    B(:,2) = B(:,2) + T(2);
    B(:,3) = B(:,3) + T(3);
end


function [samples] = uniformSample(pts, nPts)

n = size(pts, 1);
sampledIndices=1:fix(n/nPts):n;
samples = pts(sampledIndices, :);

end

function tf = isRigidTransform(tform)

singularValues = svd(tform.T(1:tform.Dimensionality,1:tform.Dimensionality));
tf = max(singularValues)-min(singularValues) < 100*eps(max(singularValues(:)));
tf = tf && abs(det(tform.T)-1) < 100*eps(class(tform.T));

end