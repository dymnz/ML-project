clear; close all;
load('t3.mat')
load('x3.mat')
TrainX = x3_v2.train_x;
TrainT = t3_v2.train_y;
TestX = x3_v2.test_x;
TestT = t3_v2.test_y;
clear t3_v2 x3_v2;

TrainError = zeros(1, 25/0.25+1);
AllW = cell(1, 1);
I = 0;
for lambda = -20:0.25:5
    I = I+1;
    for M = 9:9      
        A = ones(15, 1);
        for m = 1:M
            A = cat(2, A, TrainX.^m);
        end
        W = (A.'*A+exp(lambda).*eye(10))\A.'*TrainT;

        Y = A*W;       
        TrainError(1, I) = sqrt(sum((Y-TrainT).^2)/15);
        
        AllW{I} = W;
    end    
end
plot(-20:0.25:5, TrainError);
hold on;

TestError = zeros(1, 25/0.25+1);
I = 0;
for lambda = -20:0.25:5
    I = I+1;
    for M = 9:9
        A = ones(10, 1);
        for m = 1:M
            A = cat(2, A, TestX.^m);
        end
        Y = A*(AllW{I});
        TestError(1, I) = sqrt(sum((Y-TestT).^2)/10);
         
    end
end
plot(-20:0.25:5, TestError);
xlabel('ln(lambda)') % x-axis label
ylabel('Erms') % y-axis label