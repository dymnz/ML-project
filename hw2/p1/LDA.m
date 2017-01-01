%% Init
clear; close all;
TargetDim = 2;
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

OriginalDim = 4;
MaxClass = 3;
TrainCount = 120;
TestCount = 30;

TrainSet = Data(1:TrainCount, :);   % I x D
TestSet = Data(TrainCount+1:TrainCount+TestCount, :);   

TrainTarget = Label(1:TrainCount, :);   % I x 1
TestTarget = Label(TrainCount+1:TrainCount+TestCount, :);

%% LDA to N-d
means = zeros(MaxClass, OriginalDim);   % K x D
covs = zeros(OriginalDim, OriginalDim, MaxClass);   % D x D x K

% Set Mean
setMean = sum(TrainSet)./TrainCount;

% Set Cov
Component = TrainSet - repmat(setMean, TrainCount, 1);
setCov = Component' * Component;

% Class Count
counts = zeros(MaxClass, 1);
for i = 1 : MaxClass
   counts(i) = sum(logical(TrainTarget==i));
end

% Class Means
acc = zeros(size(means));
for i = 1 : TrainCount
    acc(Label(i), :) = acc(Label(i), :) + TrainSet(i, :); 
end
means = acc./repmat(counts, 1, OriginalDim);

% Class Covs
for i = 1 : TrainCount
    covs(:, :, Label(i)) =  covs(:, :, Label(i)) + ...
        (TrainSet(i, :) - means(Label(i), :))' * ...
        (TrainSet(i, :) - means(Label(i), :));    
end

% Sw = Sum of Class Covs
Sw = zeros(OriginalDim, OriginalDim);
for i = 1 : MaxClass
    Sw = Sw + covs(:, :, i)./repmat(TrainCount, OriginalDim, OriginalDim);
end

% Sb
Sb = zeros(OriginalDim, OriginalDim);
for i = 1 : MaxClass
    Sb = Sb + counts(i)*(means(i, :)-setMean)'*(means(i, :)-setMean);
end

% Eigen Decomp
[V D] = eig(inv(Sw)*Sb);
[B I] = sort(sum(D), 'descend');

% Construct W
W = zeros(OriginalDim, TargetDim);
for i = 1 : TargetDim
    W(:, i) = V(:, I(i));
end

% Transform
TrainSet = (TrainSet*W);
TestSet = (TestSet*W);

%% Learning - Dim = TargetDim
Dim = TargetDim;
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
    trainClassChart(TrainTarget(i), ind(i)) = ...
        trainClassChart(TrainTarget(i), ind(i)) + 1;
    if ind(i)~=TrainTarget(i)
        errorCounts(TrainTarget(i)) = errorCounts(TrainTarget(i)) +1;
    end
end
disp('Training Class Chart');
disp(trainClassChart);
disp(100 - sum(errorCounts)/TrainCount*100);

% Testing Inference
likelihoods = zeros(TestCount, MaxClass);
probs = zeros(TestCount, MaxClass);
for k = 1 : MaxClass
    likelihoods(:, k) = mvnpdf(TestSet, means(k, :), sharedCov);
    probs(:, k) = priors(k) * likelihoods(:, k);
end

% Testing Class chart
errorCounts = zeros(MaxClass, 1);
testClassChart = zeros(MaxClass);
[val ind] = max(probs');
for i = 1 : TestCount
    testClassChart(TestTarget(i), ind(i)) = ...
        testClassChart(TestTarget(i), ind(i)) + 1;
    if ind(i)~=TestTarget(i)
        errorCounts(TestTarget(i)) = errorCounts(TestTarget(i)) +1;
    end
end
disp('Testing Class Chart');
disp(testClassChart);
disp(100 - sum(errorCounts)/TestCount*100);

%% Plot
TransData = (Data*W);
figure;
if TargetDim == 3 
for i = 1 : MaxClass
    scatter3(TransData(logical(Label == i), 1), ...
    TransData(logical(Label == i), 2), ...
    TransData(logical(Label == i), 3));
    hold on;
end
end

if TargetDim == 2 
for i = 1 : MaxClass
    scatter(TransData(logical(Label == i), 1), ...
        TransData(logical(Label == i), 2));
    hold on;
end
end

if TargetDim == 1 
for i = 1 : MaxClass
    plot(TransData(logical(Label == i), 1));
    hold on;
end
end

title('LDA');
legend('SET','VIR', 'VER')
hold off;

