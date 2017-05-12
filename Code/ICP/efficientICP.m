function [tform, rmse] = efficientICP(srcPts, modelPts, varargin)

[modelNormal, doExtrapolate, inlierRatio, maxIterations, tolerance, ...
    initialTransform] = validateAndParseOptInputs(srcPts, modelPts, varargin{:});

modelObj = pointCloud(modelPts);

% compute normals for modelPts
if isnan(modelNormal)
    modelNormal = ptCloudNormal(modelPts, 5);
end


Rs = zeros(3, 3, maxIterations+1);
Ts = zeros(3, maxIterations+1);
% Quaternion and translation vector
qs = [ones(1, maxIterations+1); zeros(6, maxIterations+1)];
% The difference of quaternion and translation vector in consecutive
% iterations
dq = zeros(7, maxIterations+1);
% The angle between quaternion and translation vectors in consecutive
% iterations
dTheta = zeros(maxIterations+1, 1);
% RMSE
Err = zeros(maxIterations+1, 1);

% Apply the initial condition.
% We use pre-multiplication format in this algorithm.
Rs(:,:,1) = initialTransform.T(1:3, 1:3)';
Ts(:,1) = initialTransform.T(4, 1:3)';
qs(:,1) = [rotationToQuaternion(Rs(:,:,1)); Ts(:,1)];

srcPts_moved = srcPts;
if qs(1) ~= 0 || any(qs(2:end,1))
    srcPts_moved = rigidTransform(srcPts, Rs(:,:,1), Ts(:,1));
end

stopIteration = maxIterations;
upperBound = max(1, round(inlierRatio * size(srcPts, 1)));

for i = 1:maxIterations
    % Find correspondence
    [indices, dists] = multiQueryKNNSearchImpl(modelObj, srcPts_moved, 1);
    
    % Reject outliers
    src_inlier_ind = 1:size(srcPts_moved, 1);
    [~, idx] = sort(dists, 'descend');
    src_inlier_ind = src_inlier_ind(idx);
    src_inlier_ind = src_inlier_ind(end-upperBound:end);
    model_inlier_ind = indices(idx);
    model_inlier_ind = model_inlier_ind(end-upperBound:end);

    % Picky ICP
    % If one model point is matched to two source points,
    % reject the match with larger distance
    hashTable = containers.Map(model_inlier_ind, src_inlier_ind);
    src_inlier_ind = cell2mat(hashTable.values);
    model_inlier_ind = cell2mat(hashTable.keys);
    inlierDist = dists(src_inlier_ind);

    if i == 1
        Err(i) = sqrt(sum(inlierDist)/length(inlierDist));
    end
    
    % Estimate transformation
    [R, T] = minimizePointToPlaneMetric(srcPts_moved(src_inlier_ind, :), ...
                    modelPts(model_inlier_ind, :), modelNormal(model_inlier_ind, :));
                
    % Update the total transformation
    Rs(:,:,i+1) = R * Rs(:,:,i);
    Ts(:,i+1) = R * Ts(:,i) + T;            
    
    % RMSE
    srcPts_moved = rigidTransform(srcPts, Rs(:,:,i+1), Ts(:,i+1));
    squaredError = sum((srcPts_moved(src_inlier_ind, :) - modelPts(model_inlier_ind, :)).^2, 2);
    Err(i+1) = sqrt(sum(squaredError)/length(squaredError));
    
    % Convert to vector representation
    qs(:,i+1) = [rotationToQuaternion(Rs(:,:,i+1)); Ts(:,i+1)];
    
    % With extrapolation, we might be able to converge faster
    if doExtrapolate
        extrapolateInTransformSpace;
    end
    
    % Check convergence    
    % Compute the mean difference in R/T from the recent three iterations.
    [dR, dT] = getChangesInTransformation;
    
    % Stop ICP if it already converges
    if dT <= tolerance(1) && dR <= tolerance(2)
        stopIteration = i;
        fprintf('Early stop at %d/%d...\n', i, maxIterations);
        break;
    end
end

% Make the R to be orthogonal as much as possible
R = Rs(:,:,stopIteration+1)';
[U, ~, V] = svd(R);
R = U * V';
tformMatrix = [R, zeros(3,1);...
               Ts(:, stopIteration+1)',  1];
tform = affine3d(tformMatrix);
rmse = Err(stopIteration+1);

    %======================================================================
    % Nested function to perform extrapolation
    % Besl, P., & McKay, N. (1992). A method for registration of 3-D shapes. 
    % IEEE Transactions on pattern analysis and machine intelligence, p245.
    %======================================================================
    function extrapolateInTransformSpace        
        dq(:,i+1) = qs(:,i+1) - qs(:,i);
        n1 = norm(dq(:,i));
        n2 = norm(dq(:,i+1));
        dTheta(i+1) = (180/pi)*acos(dot(dq(:,i),dq(:,i+1))/(n1*n2));

        angleThreshold = 10;
        scaleFactor = 25;
        if i > 2 && dTheta(i+1) < angleThreshold && dTheta(i) < angleThreshold
            d = [Err(i+1), Err(i), Err(i-1)];
            v = [0, -n2, -n1-n2];
            vmax = scaleFactor * n2;
            dv = extrapolate(v,d,vmax);
            if dv ~= 0
                q = qs(:,i+1) + dv * dq(:,i+1)/n2;
                q(1:4) = q(1:4)/norm(q(1:4));
                % Update transformation and data
                qs(:,i+1) = q;
                Rs(:,:,i+1) = quaternionToRotation(q(1:4));
                Ts(:,i+1) = q(5:7);
                srcPts_moved = rigidTransform(srcPts, Rs(:,:,i+1), Ts(:,i+1));
            end
        end
    end

    %======================================================================
    % Nested function to compute the changes in rotation and translation
    %======================================================================
    function [dR, dT] = getChangesInTransformation
        dR = 0;
        dT = 0;
        count = 0;
        for k = max(i-2,1):i
            % Rotation difference in radians
            rdiff = acos(dot(qs(1:4,k),qs(1:4,k+1))/(norm(qs(1:4,k))*norm(qs(1:4,k+1))));
            % Euclidean difference
            tdiff = sqrt(sum((Ts(:,k)-Ts(:,k+1)).^2));
            dR = dR + rdiff;
            dT = dT + tdiff;
            count = count + 1;
        end
        dT = dT/count;
        dR = dR/count;
    end


end

%==========================================================================
% Parameter validation
%==========================================================================
function [normalsData, doExtrapolate, inlierRatio, maxIterations, tolerance, ...
            initialTransform] = validateAndParseOptInputs(srcPts, modelPts, varargin)
        
    % Parse the input P-V pairs
    defaults = struct(...
        'NormalsData', NaN, ...
        'Extrapolate',  false, ...
        'InlierRatio', 1.0,...
        'MaxIterations', 20,...
        'Tolerance', [0.01, 0.009],...
        'InitialTransform', affine3d());

    parser = inputParser;
    parser.CaseSensitive = false;
    
    parser.addRequired('srcPts', @(x)isreal(x) && size(x, 2) == 3);
    parser.addRequired('modelPts', @(x)isreal(x) && size(x, 2) == 3);
    parser.addParameter('NormalsData', defaults.NormalsData, ...
                @(x)isreal(x) && size(x, 2) == 3);
    parser.addParameter('Extrapolate', defaults.Extrapolate, ...
                @(x)validateattributes(x,{'logical'}, {'scalar','nonempty'}));
    parser.addParameter('InlierRatio', defaults.InlierRatio, ...
                @(x)validateattributes(x,{'single', 'double'}, {'real','nonempty','scalar','>',0,'<=',1}));
    parser.addParameter('MaxIterations', defaults.MaxIterations, ...
                @(x)validateattributes(x,{'single', 'double'}, {'scalar','integer'}));
    parser.addParameter('Tolerance', defaults.Tolerance, ...
                @(x)validateattributes(x,{'single', 'double'}, {'real','nonnegative','numel', 2}));        
    parser.addParameter('InitialTransform', defaults.InitialTransform, ...
                @(x)validateattributes(x,{'affine3d'}, {'scalar'}));   
            
    parser.parse(srcPts, modelPts, varargin{:});

    normalsData     = parser.Results.NormalsData;
    doExtrapolate   = parser.Results.Extrapolate;
    inlierRatio     = parser.Results.InlierRatio;
    maxIterations   = parser.Results.MaxIterations;
    tolerance       = parser.Results.Tolerance;
    initialTransform = parser.Results.InitialTransform;
    if ~(isRigidTransform(initialTransform))
        error(message('vision:pointcloud:rigidTransformOnly'));
    end
end

%==========================================================================
% Determine if transformation is rigid transformation
%==========================================================================
function tf = isRigidTransform(tform)

singularValues = svd(tform.T(1:tform.Dimensionality,1:tform.Dimensionality));
tf = max(singularValues)-min(singularValues) < 100*eps(max(singularValues(:)));
tf = tf && abs(det(tform.T)-1) < 100*eps(class(tform.T));

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

function quaternion = rotationToQuaternion(R)

Qxx = R(1,1);
Qxy = R(1,2);
Qxz = R(1,3);
Qyx = R(2,1);
Qyy = R(2,2);
Qyz = R(2,3);
Qzx = R(3,1);
Qzy = R(3,2);
Qzz = R(3,3);

t = Qxx+Qyy+Qzz;

if t >= 0,
    r = sqrt(1+t);
    s = 0.5/r;
    w = 0.5*r;
    x = (Qzy-Qyz)*s;
    y = (Qxz-Qzx)*s;
    z = (Qyx-Qxy)*s;
else
    maxv = max(Qxx, max(Qyy, Qzz));
    if maxv == Qxx
        r = sqrt(1+Qxx-Qyy-Qzz);
        s = 0.5/r;
        w = (Qzy-Qyz)*s;
        x = 0.5*r;
        y = (Qyx+Qxy)*s;
        z = (Qxz+Qzx)*s;
    elseif maxv == Qyy
        r = sqrt(1+Qyy-Qxx-Qzz);
        s = 0.5/r;
        w = (Qxz-Qzx)*s;
        x = (Qyx+Qxy)*s;
        y = 0.5*r;
        z = (Qzy+Qyz)*s;
    else
        r = sqrt(1+Qzz-Qxx-Qyy);
        s = 0.5/r;
        w = (Qyx-Qxy)*s;
        x = (Qxz+Qzx)*s;
        y = (Qzy+Qyz)*s;
        z = 0.5*r;
    end
end

quaternion = [w;x;y;z];

end

function R = quaternionToRotation(quaternion)
% quaternionToRotation Converts (unit) quaternion to (orthogonal) rotation matrix.
% 
% quaternion is a 4-by-1 vector
% R is a 3x3 orthogonal matrix of corresponding rotation matrix
%
% Note
% ----
% R is rotation of vectors anti-clockwise in a right-handed system by pre-multiplication

% Copyright 2014 The MathWorks, Inc.

% References
% ----------
% http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#From_a_quaternion_to_an_orthogonal_matrix

q0 = quaternion(1);
qx = quaternion(2);
qy = quaternion(3);
qz = quaternion(4);

R = [q0.^2+qx.^2-qy.^2-qz.^2, 2*qx.*qy-2*q0.*qz,       2*qx.*qz+2*q0.*qy; ...
     2*qx.*qy+2*q0.*qz,       q0.^2-qx.^2+qy.^2-qz.^2, 2*qy.*qz-2*q0.*qx; ...
     2*qx.*qz-2*q0.*qy,       2*qy.*qz+2*q0.*qx,       q0.^2-qx.^2-qy.^2+qz.^2];
end

%==========================================================================

function B = rigidTransform(A, R, T)
    B = A * R';
    B(:,1) = B(:,1) + T(1);
    B(:,2) = B(:,2) + T(2);
    B(:,3) = B(:,3) + T(3);
end

%==========================================================================

function [R, T] = minimizePointToPlaneMetric(p, q, nv)
    % Set up the linear system
    cn = [cross(p,nv,2),nv];
    C = cn'*cn;
    qp = q-p;
    b =  [sum(sum(qp.*repmat(cn(:,1),1,3).*nv, 2));
          sum(sum(qp.*repmat(cn(:,2),1,3).*nv, 2));
          sum(sum(qp.*repmat(cn(:,3),1,3).*nv, 2));
          sum(sum(qp.*repmat(cn(:,4),1,3).*nv, 2));
          sum(sum(qp.*repmat(cn(:,5),1,3).*nv, 2));
          sum(sum(qp.*repmat(cn(:,6),1,3).*nv, 2))];
    % X is [alpha, beta, gamma, Tx, Ty, Tz]
    X = C\b;

    cx = cos(X(1)); 
    cy = cos(X(2)); 
    cz = cos(X(3)); 
    sx = sin(X(1)); 
    sy = sin(X(2)); 
    sz = sin(X(3)); 

    R = [cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz;
         cy*sz, cx*cz+sx*sy*sz, cx*sy*sz-sx*cz;
           -sy,          sx*cy,          cx*cy];

    T = X(4:6);
end

%==========================================================================

function dv = extrapolate(v,d,vmax)
    p1 = polyfit(v,d,1);    % linear fit
    p2 = polyfit(v,d,2);    % parabolic fit
    v1 = -p1(2)/p1(1);      % linear zero crossing point
    v2 = -p2(2)/(2*p2(1));  % polynomial top point

    if (issorted([0 v2 v1 vmax]) || issorted([0 v2 vmax v1]))
        % Parabolic update
        dv = v2;
    elseif (issorted([0 v1 v2 vmax]) || issorted([0 v1 vmax v2])...
            || (v2 < 0 && issorted([0 v1 vmax])))
        % Line update
        dv = v1;
    elseif (v1 > vmax && v2 > vmax)
        % Maximum update
        dv = vmax;
    else
        % No extrapolation
        dv = 0;
    end
end