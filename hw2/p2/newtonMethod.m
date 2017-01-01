function [L, G, H] = newtonMethod(target, X, Theta)

L = 0;
I = size(X, 1);
K = size(Theta, 2);
D = size(Theta, 1);

g = zeros(size(Theta));
h = zeros(K, K, D, D);

for i = 1 : I    
    xi = X(i, :).';
    yi = softmax(xi, Theta);
    L = L - log( yi(target(i) + 1) );
    
    
    for n = 1 : K
        if target(i) + 1 == n
            g(:, n) = g(:, n) + (yi(n) - 1)*xi;
        else
            g(:, n) = g(:, n) + yi(n)*xi;
        end
        
        for m = 1 : K
            if m == n
                h(n, m, :, :) = squeeze(h(n, m, :, :)) + yi(n)*(1-yi(m))*xi*xi';
            else
                h(n, m, :, :) = squeeze(h(n, m, :, :)) - yi(n)*yi(m)*xi*xi';
            end
        end
        
    end        
end

H = zeros(D*K, D*K);
G = zeros(D*K, 1);
for n = 1 : K
    for m = 1 : K
        H( 1+(D*(n-1)) : (D*n) , 1+(D*(m-1)) : (D*m) ) =  ...
            squeeze(h(n, m, :, :));
    end
    G( 1+(D*(n-1)) : D*n ) = g(:, n);
end

end