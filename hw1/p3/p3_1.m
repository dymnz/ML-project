clear; close all;

%% CHANGE THIS to switch between 1st, 2nd or 3rd order
cOrder = 3;

%% Constants and data reading %%
strFileName = 'data.xlsx';
cRowOffset = 1;
cDimension = 4;
cTrainSetStart = 1; cTrainSetSize = 400;
cTestSetStart = 401; cTestSetSize = 100;

strTrainXRange = sprintf('A%d:D%d', cRowOffset + cTrainSetStart, cRowOffset + cTrainSetStart + cTrainSetSize - 1);
strTrainTRange = sprintf('E%d:E%d', cRowOffset + cTrainSetStart, cRowOffset + cTrainSetStart + cTrainSetSize - 1);
strTestXRange = sprintf('A%d:D%d', cRowOffset + cTestSetStart, cRowOffset + cTestSetStart + cTestSetSize - 1);
strTestTRange = sprintf('E%d:E%d', cRowOffset + cTestSetStart, cRowOffset + cTestSetStart + cTestSetSize - 1);

TrainX = xlsread(strFileName, strTrainXRange);
TrainT = xlsread(strFileName, strTrainTRange);
TestX = xlsread(strFileName, strTestXRange);
TestT = xlsread(strFileName, strTestTRange);


%% 2/3nd order linear regression by partial derivation %%
%% Build the Normal Equation: AW = Y 
cOrder = min(cOrder, 3);
S = (1 - cDimension^(cOrder+1))/(1 - cDimension);
W = zeros(S);
Y = zeros(length(W), 1);
A = zeros(length(W), length(W));

% Build the Weight from partial derivative
% If I had more time, I would have written a shorter code
Weight = ones(cTrainSetSize, 1);
for i = 1 : cDimension
    Weight = cat(2, Weight, TrainX(:, i));
end
if cOrder > 1
for i = 1 : cDimension
    for r = 1 : cDimension
        Weight = cat(2, Weight, TrainX(:, i).*TrainX(:, r));
    end
end
end
if cOrder > 2
for i = 1 : cDimension
    for r = 1 : cDimension
        for p = 1 : cDimension
        Weight = cat(2, Weight, TrainX(:, i).*TrainX(:, r).*TrainX(:, p));
%         disp(sprintf('ind%d:X%d%d%d', (i-1)*cDimension*cDimension+(r-1)*cDimension+p+5,i, r, p));
        end
    end
end
end

% Build the A and Y matrix
for i = 1 : length(W)
        A(i, :) = sum(Weight.*repmat(Weight(:, i), 1, S));
        Y(i) = sum(TrainT.*Weight(:, i));
end

%% Find W using W = pinv(A)Y
% W = A\Y;
W = pinv(A)*Y;

%% Testing with test set
% Construct X for test set
X = ones(cTestSetSize, 1);
for i = 1 : cDimension
    X = cat(2, X, TestX(:, i));
end
if cOrder > 1
for i = 1 : cDimension
    for r = 1 : cDimension
        X = cat(2, X, TestX(:, i).*TestX(:, r));
    end
end
end
if cOrder > 2
for i = 1 : cDimension
    for r = 1 : cDimension
        for p = 1 : cDimension
        X = cat(2, X, TestX(:, i).*TestX(:, r).*TestX(:, p));
        end
    end
end
end

% Predict
Predictions = X*W;

% Erms
Erms = sqrt( sum((Predictions-TestT).^2)/cTestSetSize );
disp(sprintf('Erms for testing set: %f\n', Erms));

%% Testing with train set
% Construct X for training set
X = ones(cTrainSetSize, 1);
for i = 1 : cDimension
    X = cat(2, X, TrainX(:, i));
end
if cOrder > 1
for i = 1 : cDimension
    for r = 1 : cDimension
        X = cat(2, X, TrainX(:, i).*TrainX(:, r));
    end
end
end
if cOrder > 2
for i = 1 : cDimension
    for r = 1 : cDimension
        for p = 1 : cDimension
        X = cat(2, X, TrainX(:, i).*TrainX(:, r).*TrainX(:, p));
        end
    end
end
end

% Predict
Predictions = X*W;

% Erms
Erms = sqrt( sum((Predictions-TrainT).^2)/cTrainSetSize );
disp(sprintf('Erms for training set: %f\n', Erms));



