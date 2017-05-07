clear; clc;

m = 100; % width of grid
n = m^2; % number of points

[X,Y] = meshgrid(linspace(-2,2,m), linspace(-2,2,m));

X = reshape(X,1,[]);
Y = reshape(Y,1,[]);

Z = sin(X).*cos(Y);
%Z = sin(X);

% Create the data point-matrix
D = [X; Y; Z];

% Translation values (a.u.):
Tx = 0.5;
Ty = -0.3;
Tz = 0.2;

% Translation vector
T = [Tx; Ty; Tz];

% Rotation values (rad.):
rx = 0.3;
ry = -0.2;
rz = 0.05;

Rx = [1 0 0;
      0 cos(rx) -sin(rx);
      0 sin(rx) cos(rx)];
  
Ry = [cos(ry) 0 sin(ry);
      0 1 0;
      -sin(ry) 0 cos(ry)];
  
Rz = [cos(rz) -sin(rz) 0;
      sin(rz) cos(rz) 0;
      0 0 1];

% Rotation matrix
R = Rx*Ry*Rz;

% Transform data-matrix plus noise into model-matrix 
M = R * D + repmat(T, 1, n);

% Add noise to model and data
rng(2912673);
M = M + 0.01*randn(3,n);
D = D + 0.01*randn(3,n);


% point cloud 
pc_M = pointCloud(M', 'Color', repmat(uint8([0 0 225]), size(M, 2), 1));
pc_D = pointCloud(D', 'Color', repmat(uint8([225 0 0]), size(M, 2), 1));

figure;
showPointCloud(pc_M);
hold on
showPointCloud(pc_D);

pc_D_sampled = pcdownsample(pc_D, 'random', 0.6);
figure;
showPointCloud(pc_M);
hold on
showPointCloud(pc_D_sampled);
tform = pcregrigid(pc_D_sampled, pc_M);

pcOut = pctransform(pc_D_sampled, tform);
figure;
showPointCloud(pc_M);
hold on
showPointCloud(pcOut);







