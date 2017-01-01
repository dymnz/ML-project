function Y = sigmoid(X)
% Softmax function
% INPT: X: DxI matrix. The dataset with D dimensions and I samples
% OUPT: Y: Ix1 matrix. The result of Sigmoid of I samples

Y = 1 ./ (1 + exp(-X));

end