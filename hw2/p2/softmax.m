function y = softmax(x, theta)
% Softmax function
% INPT: x: Dx1
%       theta: DxK
% OUPT: t: Kx1

weight = exp(theta.' * x);
y = weight / sum(weight);

end