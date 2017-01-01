clear; close all;
load('t3.mat')
load('x3.mat')
TrainX = x3_v2.train_x;
TrainT = t3_v2.train_y;
TestX = x3_v2.test_x;
TestT = t3_v2.test_y;
clear t3_v2 x3_v2;

MAX = 9;

splitTestX = [TrainX(11:15) TrainX(1:5) TrainX(6:10)];
splitTestT = [TrainT(11:15) TrainT(1:5) TrainT(6:10)];

splitTrainX = [cat(1, TrainX(1:5), TrainX(6:10)) ...
                cat(1, TrainX(6:10), TrainX(11:15))...
                cat(1, TrainX(11:15), TrainX(1:5))];
splitTrainT = [cat(1, TrainT(1:5), TrainT(6:10)) ...
                cat(1, TrainT(6:10), TrainT(11:15))...
                cat(1, TrainT(11:15), TrainT(1:5))];

TrainError = zeros(MAX, 3);
TrainErrorX = zeros(MAX, 1);

BestW = cell(MAX, 1);
for M = 1:MAX
    W = zeros(M+1, 3);
    for i = 1:3        
        
        A = ones(10, 1);
        for m = 1:M
            A = cat(2, A, splitTrainX(:, i).^m);
        end
        W(:, i) = A\splitTrainT(:, i);
        
        A = ones(5, 1);
        for m = 1:M
            A = cat(2, A, splitTestX(:, i).^m);
        end
        
        Y = A*W(:, i);
        
        TrainError(M, i) = sqrt(sum((Y-splitTestT(:, i)).^2)/5);
    end
    [minError, minIndex] = min(TrainError(M, :));
    TrainErrorX(M) = minError;
    BestW{M} = W(:, minIndex);
end
plot(TrainErrorX);
hold on

for i = 1:9
    disp(sprintf('%.2f\t%.2f\t%.2f', TrainError(i, 1), TrainError(i, 2), TrainError(i, 3)));    
end


TestError = zeros(MAX, 1);
for M = 1:MAX
    A = ones(10, 1);
    for m = 1:M
        A = cat(2, A, TestX.^m);
    end
    Y = A*(BestW{M});
    TestError(M) = sqrt((sum((Y-TestT).^2))/5);
end
plot(TestError);



