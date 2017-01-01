clear; close all;
load('t3.mat')
load('x3.mat')
TrainX = x3_v2.train_x;
TrainT = t3_v2.train_y;
TestX = x3_v2.test_x;
TestT = t3_v2.test_y;
clear t3_v2 x3_v2;

TrainError = zeros(9, 1);
AllW = cell(9, 1);
for M = 1:9      
    A = ones(15, 1);
    for m = 1:M
        A = cat(2, A, TrainX.^m);
    end
    W = A\TrainT;

    A = ones(15, 1);
    for m = 1:M
        A = cat(2, A, TrainX.^m);
    end
    Y = A*W;

    TrainError(M) = sqrt((sum((Y-TrainT).^2))/15);
    
    AllW{M} = W;
end
plot(TrainError);
hold on;

TestError = zeros(9, 1);
for M = 1:9
    A = ones(10, 1);
    for m = 1:M
        A = cat(2, A, TestX.^m);
    end
    Y = A*(AllW{M});
    TestError(M) = sqrt((sum((Y-TestT).^2))/10);
end
plot(TestError);
xlabel('Order') % x-axis label
ylabel('Erms') % y-axis label