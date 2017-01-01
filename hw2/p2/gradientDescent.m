function [L, g] = gradientDescent(target, X, oldTheta)

L = 0;
g = zeros(size(oldTheta));
I = size(X, 1);
K = size(oldTheta, 2);

for i = 1 : I    
    xi = X(i, :).';
    yi = softmax(xi, oldTheta);
    L = L - log( yi(target(i) + 1) );

    for n = 1 : K
        if target(i) + 1 == n
            g(:, n) = g(:, n) + (yi(n) - 1)*xi;
        else
            g(:, n) = g(:, n) + yi(n)*xi;
        end            
    end        
end

end