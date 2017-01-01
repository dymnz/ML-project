% Gaussian Process for Regression
clear; close all;

TrainSize = 1200;
TestSize = 400;
Dimension = 2;

%% Read data
Data = csvread('../Data/gp_data.csv', 0, 0);

TrainSet = Data(1:TrainSize, :);
TestSet = Data(TrainSize+1:TrainSize+TestSize, :);

TrainTarget = TrainSet(:, Dimension+1);
TrainSet = TrainSet(:, 1:Dimension);

TestTarget = TestSet(:, Dimension+1);
TestSet = TestSet(:, 1:Dimension);

%% Build Cov
Alpha = 0.01;
Theta = [1; 0.5];
Eta = [1; 1];

Cov = zeros(TrainSize);
for i = 1 : TrainSize
    for r = i : TrainSize
        Cov(i, r) = GPRegressionKernel( ...
            TrainSet(i, :), TrainSet(r, :), Theta, Eta);
        Cov(r, i) = Cov(i, r);
    end
end
Cov = Cov + Alpha*eye(size(Cov));

%% Prediction
TestPred = zeros(size(TestTarget));
TestError = zeros(size(TestTarget));
k = zeros(size(TrainTarget));
c = 0;

for i = 1 : TestSize
    for r = 1 : TrainSize
        k(r) = GPRegressionKernel( ...
            TestSet(i, :), TrainSet(r, :), Theta, Eta);
    end
    c = GPRegressionKernel( ...
            TestSet(i, :), TestSet(i, :), Theta, Eta) + Alpha;
    
    V = k' * inv(Cov);
    mu = V * TrainTarget;
    sigma = c - V*k;
    TestPred(i) = mu;
end

%% Show result
TestError = TestPred - TestTarget;
TestErms = sqrt(2*sum(TestError.^2)/TestSize);
disp(sprintf('Erms: %.4f', TestErms));
scatter3(TestSet(:, 1), TestSet(:, 2), TestTarget);
hold on;
scatter3(TestSet(:, 1), TestSet(:, 2), TestPred, '*');
xlabel('x1'); ylabel('x2'); zlabel('value');
legend('Target','Prediction');


