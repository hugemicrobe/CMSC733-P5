function [best_model, best_inliers] = fitPlaneRANSAC(X, maxIters, threshold)

N = size(X, 1);
X_ = [X, ones(N, 1)];


max_ninliers = 0;
best_model = [];
best_inliers = [];


for i = 1:maxIters
    ind = randperm(N, 9);
    
    % fit plane to three points A,B,C
%     AB = X(ind(2), :) - X(ind(1), :);
%     AC = X(ind(3), :) - X(ind(1), :);
%     
%     normal = cross(AB, AC)';
%     d = -X(ind(1), :) * normal;
%     
%     model = [normal; d];
%     model = model / norm(model);
    
    model = fitLS(X_(ind, :));
    err = evaluate(X_, model);
    
    inliers = err < threshold;
    
    if sum(inliers) > max_ninliers
        max_ninliers = sum(inliers);
        best_model = model;
        best_inliers = inliers;
    end
end



end


function model = fitLS(X_)

[~, ~, V] = svd(X_, 0);
model = V(:, end);

end

function err = evaluate(X_, model)
% model: ax + by + cz + d = 0
err = abs(X_ * model);

end

function samples = randperm2(N, K)
x = false(N, 1);
sumx = 0;
while sumx < K
    x(randi(N,1,K-sumx)) = true;
    sumx = sum(x);
end
samples = find(x);
samples = samples(randperm(K));

end