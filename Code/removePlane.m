function toRemove = removePlane(xyzPoints, model, tolerance)
n = size(xyzPoints, 1);

X = [xyzPoints, ones(n, 1)];
d = abs(X * model);
toRemove = (d > tolerance);

end