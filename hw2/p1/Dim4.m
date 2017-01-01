%% Init
clear; close all;
[Data RawLabel] = xlsread('../Data/Irisdat .xls');

% Data = Data(2:end, :);
RawLabel = RawLabel(2:end, end);

Label = zeros(size(RawLabel));
for i = 1 : numel(RawLabel)
    switch RawLabel{i}
        case 'SETOSA'
            Label(i) = 1;
        case 'VIRGINIC'
            Label(i) = 2;
        case 'VERSICOL'
            Label(i) = 3;
        otherwise
            error('??????????');
    end
end

MaxClass = 3;
TrainCount = 120;
TestCount = 30;

TrainSet = Data(1:TrainCount, :);   % I x D
TestSet = Data(TrainCount+1:TrainCount+TestCount, :);   

TrainTarget = Label(1:TrainCount, :);   % I x 1
TestTarget = Label(TrainCount+1:TrainCount+TestCount, :);

%% Learning - Dim = 4
Dim = 4;
priors = zeros(MaxClass, 1);    % K x 1
means = zeros(MaxClass, Dim);   % K x D
covs = zeros(Dim, Dim, MaxClass);   % D x D x K

% Prior
counts = zeros(MaxClass, 1);
for i = 1 : TrainCount
    counts(TrainTarget(i)) = counts(TrainTarget(i)) + 1;
end
priors = counts./TrainCount;

% Mean
acc = zeros(size(means));   % K x D
for i = 1 : TrainCount
    acc(TrainTarget(i), :) = acc(TrainTarget(i), :) + TrainSet(i, :); 
end
means = acc./repmat(counts, 1, Dim);

% Cov
for i = 1 : TrainCount
    covs(:, :, TrainTarget(i)) =  covs(:, :, TrainTarget(i)) + ...
        (TrainSet(i, :) - means(TrainTarget(i), :))' * ...
        (TrainSet(i, :) - means(TrainTarget(i), :));    
end
sharedCov = zeros(Dim, Dim);
for i = 1 : MaxClass
    sharedCov = sharedCov + covs(:, :, i)./repmat(TrainCount, Dim, Dim);
end

% Training Inference
errorCounts = zeros(MaxClass, 1);
likelihoods = zeros(TrainCount, MaxClass);
probs = zeros(TrainCount, MaxClass);
for k = 1 : MaxClass
    likelihoods(:, k) = mvnpdf(TrainSet, means(k, :), sharedCov);
    probs(:, k) = priors(k) * likelihoods(:, k);
end

% Training Class chart
trainClassChart = zeros(MaxClass);
[val ind] = max(probs');
for i = 1 : TrainCount
    trainClassChart(TrainTarget(i), ind(i)) = trainClassChart(TrainTarget(i), ind(i)) + 1;
    if ind(i)~=TrainTarget(i)
        errorCounts(TrainTarget(i)) = errorCounts(TrainTarget(i)) +1;
    end
end
disp('Training Class Chart');
disp(trainClassChart);
disp(100 - sum(errorCounts)/TrainCount*100);

% Testing Inference
errorCounts = zeros(MaxClass, 1);
likelihoods = zeros(TestCount, MaxClass);
probs = zeros(TestCount, MaxClass);
for k = 1 : MaxClass
    likelihoods(:, k) = mvnpdf(TestSet, means(k, :), sharedCov);
    probs(:, k) = priors(k) * likelihoods(:, k);
end

% Testing Class chart
testClassChart = zeros(MaxClass);
[val ind] = max(probs');
for i = 1 : TestCount
    testClassChart(TestTarget(i), ind(i)) = testClassChart(TestTarget(i), ind(i)) + 1;
    if ind(i)~=TestTarget(i)
        errorCounts(TestTarget(i)) = errorCounts(TestTarget(i)) +1;
    end
end
disp('Testing Class Chart');
disp(testClassChart);
disp(100 - sum(errorCounts)/TestCount*100);

