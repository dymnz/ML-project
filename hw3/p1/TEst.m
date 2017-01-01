I = eye(TrainSize, TrainSize);
A = zeros(TrainSize, 1);

% Build W
for i = 1 : 5
    Sigma = sigmoid(A);
    W = diag(Sigma.*(1-Sigma));
    A = Cov*inv(I + W*Cov)*(TrainTarget - Sigma + W*A);
    B = (TrainTarget-Sigma-inv(Cov)*A);
    disp(B(1));
end