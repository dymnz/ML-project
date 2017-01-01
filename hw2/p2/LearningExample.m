clear; close all;

%[Dst_bytes	Flag Root_shell Su_attempted 
% Dst_host_count Dst_host_srv_count
% Dst_host_same_srv_rate Dst_host_diff_srv_rate	
% Dst_host_same_src_port_rate Dst_host_srv_diff_host_rate Class]

TrainSet = csvread('../Data/kdd99_training_data.csv', 1, 0);
TestSet = csvread('../Data/kdd99_testing_data.csv', 1, 0);

DIMENSION = size(TrainSet, 2) - 1;
MAX_CLASS = 5;

TrainTarget = TrainSet(:, 11);
TrainSet = TrainSet(:, 1:10);
TrainCount = size(TrainSet, 1);

TestTarget = TestSet(:, 11);
TestSet = TestSet(:, 1:10);
TestCount = size(TestSet, 1);

%% Gradient Descent
LEARNING_RATE = 5;
theta = 0.005 * randn(DIMENSION, MAX_CLASS);
for i = 1 : 1000
   [L, g] = gradientDescent(TrainTarget, TrainSet, theta);   
   theta = theta - LEARNING_RATE.*g/TranCount;
   disp(L);
end

%% Newton Method
LEARNING_RATE = 0.05;
theta = 0.005 * randn(DIMENSION, MAX_CLASS);
for i = 1 : 100
	[L, G, H] = newtonMethod(TrainTarget, TrainSet, theta) ;
    
    rTheta = reshape(theta, [DIMENSION*MAX_CLASS 1]);
    rTheta = rTheta - LEARNING_RATE*pinv(H) * G;
    theta = reshape(rTheta, [DIMENSION MAX_CLASS]);
    	
	disp(sprintf('%d %.2f', i, L));
end




