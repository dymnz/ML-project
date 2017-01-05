% One-VS-Rest 3-class SVM w/ Polynomial Kernel
clear; close all;

% OPTIONS
DrawSupportVector = false;
DrawTestingData = true;

load Iris.mat;

Class = 3;
Dimension = 2;
KernelDimension = 3;

C = 1000;
Tol = 0.001;

TrainSize = size(trainLabel, 1);
TestSize = size(testLabel, 1);
K = zeros(TrainSize, TrainSize);
alphas = zeros(Class, TrainSize);
biases = zeros(Class);
Ws = zeros(Class, KernelDimension);
SupportVectors = cell(Class, 1);

%% Build sets
TrainSet = trainFeature(trainLabel==1, :);
TestSet = testFeature(testLabel==1, :);
TrainTarget = ones(size(find(trainLabel==1)));
TestTarget = ones(size(find(testLabel==1)));
for i = 2 : Class
    TrainSet = cat(1, TrainSet, trainFeature(trainLabel==i, :));
    TestSet = cat(1, TestSet, testFeature(testLabel==i, :));
    TrainTarget = cat(1, TrainTarget, i*ones(size(find(trainLabel==i))));
    TestTarget = cat(1, TestTarget, i*ones(size(find(testLabel==i))));
end

%% Learning
% Build K
for m = 1 : TrainSize
    for n = m : TrainSize
        K(m, n) = (TrainSet(m, :) * TrainSet(n, :)')^2;
        K(n, m) = K(m, n);
    end
end    

% SMO
Phi = [TrainSet(:, 1).^2 ...
        sqrt(2)*TrainSet(:, 1).*TrainSet(:, 2)...
        TrainSet(:, 2).^2];
for i = 1 : Class
    Target = -ones(1, TrainSize);
    Target(TrainTarget==i) = 1;
    [alphas(i, :),biases(i)] = smo(K, Target, C, Tol);
    Ws(i, :) = alphas(i, :).*Target*Phi;   
end

%% Find support vector
for i = 1 : Class
    SupportVectors{i} = TrainSet(alphas(i, :)~=0, :);
end

%% Train set testing
Phi = [TrainSet(:, 1).^2 ...
        sqrt(2)*TrainSet(:, 1).*TrainSet(:, 2)...
        TrainSet(:, 2).^2];
Preds = zeros(TrainSize, Class);
for i = 1 : TrainSize
    for r = 1 : Class
        Preds(i, r) = Ws(r, :) * Phi(i, :)' + biases(r);
    end
end

[val Results] = max(Preds');
ErrorCount = double(size(find(Results'~=TrainTarget), 1));
disp(sprintf('\nTrain:\nErrorCount: \t%d\nError%%: \t%.2f%%\nCorrect%%: \t%.2f%%', ...
    ErrorCount, (ErrorCount/TrainSize)*100, (1-ErrorCount/TrainSize)*100));    

%% Test set testing
Phi = [TestSet(:, 1).^2 ...
        sqrt(2)*TestSet(:, 1).*TestSet(:, 2)...
        TestSet(:, 2).^2];
Preds = zeros(TestSize, Class);
for i = 1 : TestSize
    for r = 1 : Class
        Preds(i, r) = Ws(r, :) * Phi(i, :)' + biases(r);
    end
end

[val Results] = max(Preds');
ErrorCount = double(size(find(Results'~=TestTarget), 1));
disp(sprintf('\nTest:\nErrorCount: \t%d\nError%%: \t%.2f%%\nCorrect%%: \t%.2f%%', ...
    ErrorCount, (ErrorCount/TestSize)*100, (1-ErrorCount/TestSize)*100));    


%% Draw decision boundary
if DrawTestingData || DrawSupportVector
    
% Set plot range
xrange = [4 8];
yrange = [2 5];

% Step size
inc = 0.01;

% Get many x and ys
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
d1 = size(x, 1);
d2 = size(x, 2);
length = d1*d2;

% Vectorize
x = reshape(x, length, 1); 
y = reshape(y, length, 1);
values = [x y];

% Get their label
Phi = [values(:, 1).^2 ...
        sqrt(2)*values(:, 1).*values(:, 2)...
        values(:, 2).^2];
Preds = zeros(length, Class);
for i = 1 : size(values, 1)
    for r = 1 : Class
        Preds(i, r) = Ws(r, :) * Phi(i, :)' + biases(r);
    end
end
[val ind] = max(Preds');

% Back to matrix
decisionmap = reshape(ind, [d1 d2]);

% Draw boundary
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal'); % matlab weird
cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
colormap(cmap);

end

%% Draw training set and support vectors
if DrawSupportVector
    
symbols = ['*' '+' 'x'];
colors = hsv(Class);

% Draw trainig sample
for i = 1 : Class
    scatter(TrainSet(TrainTarget==i, 1), ...
        TrainSet(TrainTarget==i, 2), 36, colors(i, :), symbols(i));
    hold on;
end
% Draw support vector
for i = 1 : Class
    scatter(SupportVectors{i}(:, 1), ...
        SupportVectors{i}(:, 2), 50, [0 0 0], 'o');
end

end




%% Draw testing set
if DrawTestingData
    
symbols = ['*' '+' 'x'];
colors = hsv(Class);

% Draw testing sample
for i = 1 : Class
    scatter(TestSet(TestTarget==i, 1), ...
        TestSet(TestTarget==i, 2), 36, colors(i, :), symbols(i));
    hold on;
end

end

