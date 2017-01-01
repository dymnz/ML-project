function [val] = GPRegressionKernel(X1, X2, Theta, Eta)
% TODO: Make this MIMO
    val = Theta(1)*exp( -0.5*(((X1-X2).^2)*Eta) ) + Theta(2);
end