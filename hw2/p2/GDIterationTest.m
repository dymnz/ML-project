clear; close all;

TrainSet = csvread('../Data/kdd99_training_data.csv', 1, 0);
TestSet = csvread('../Data/kdd99_testing_data.csv', 1, 0);

ITERATION_COUNTS = [0 1:1:9 10:10:90 100:100:900 1000:1000:25000 100000];
LOSS_THRESHOLD = 6;

errorPTrain = zeros(1, numel(ITERATION_COUNTS));
errorPTest =zeros(1, numel(ITERATION_COUNTS));
lossTrain = zeros(1, numel(ITERATION_COUNTS));

TrainTarget = TrainSet(:, 11);
TrainSet = TrainSet(:, 1:10);
TrainCount = size(TrainSet, 1);

TestTarget = TestSet(:, 11);
TestSet = TestSet(:, 1:10);
TestCount = size(TestSet, 1);

DIMENSION = size(TrainSet, 2);
MAX_CLASS = 5;
theta = 0.005 * randn(DIMENSION, MAX_CLASS);

endingIteration = 0;

for C = 2 : numel(ITERATION_COUNTS)

%% Learning    
MAX_ITERATION = ITERATION_COUNTS(C);
LEARNING_RATE = 5;	% Learning rate

% Gradient Descent process
endingIteration = MAX_ITERATION;
for i = ITERATION_COUNTS(C-1) : MAX_ITERATION
   [L, g] = gradientDescent(TrainTarget, TrainSet, theta);   
   theta = theta - LEARNING_RATE.*(g./TrainCount);
   
   if L < LOSS_THRESHOLD
       endingIteration = i;
       break;
   end
   
end
lossTrain(C-1) = L;
disp(sprintf('Iterations: %d Final loss: %.2f', i, L));

%% Training Set error
I = size(TrainSet, 1);
K = size(theta, 2);

errorCount = zeros(1, K);

for i = 1 : I   
    xi = TrainSet(i, :).';
    yi = softmax(xi, theta);
    [v, ind] = max(yi);
    if ind ~= TrainTarget(i) + 1       
        errorCount(TrainTarget(i) + 1) = errorCount(TrainTarget(i) + 1) + 1;
    end
end

errorPTrain(C-1) = (sum(errorCount)/sum(TrainCount)) * 100;
disp(sprintf('TrainSet error: %.2f%%', ...
    (sum(errorCount)/sum(TrainCount)) * 100));

%% Testing Set error
I = size(TestSet, 1);
K = size(theta, 2);

errorCount = zeros(1, K);

for i = 1 : I   
    xi = TrainSet(i, :).';
    yi = softmax(xi, theta);
    [v, ind] = max(yi);
    if ind ~= TrainTarget(i) + 1       
        errorCount(TrainTarget(i) + 1) = errorCount(TrainTarget(i) + 1) + 1;
    end
end

errorPTest(C-1) = (sum(errorCount)/sum(TestCount)) * 100;
disp(sprintf('TestSet error: %.2f%%', ...
    (sum(errorCount)/sum(TestCount)) * 100));

if L < LOSS_THRESHOLD
    break;
end

end
%%
figure; 
plot([ITERATION_COUNTS(2:C) endingIteration], errorPTrain(1:C));
hold on;
plot([ITERATION_COUNTS(2:C) endingIteration], errorPTest(1:C));
hold off;
axis([0, endingIteration, 0, 100]);
set(gca,'xscale','log');
legend('Train','Test');
title('Miss Classification Rate vs. # of Iterations');
%%
figure; 
plot([ITERATION_COUNTS(2:C) endingIteration], lossTrain(1:C));
hold on;
% plot(ITERATION_COUNTS(2:end), lossTrain);
hold off;
axis([0, endingIteration, 0, max(lossTrain(1:C))]);
% set(gca,'xscale','log');
legend('Train','Test');
title('Loss vs. # of Iterations');
