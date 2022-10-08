clear; clc;

addpath('EMGC2F');
addpath('finchpp');
addpath('funs');

% load dataset
load('ORL.mat');
X = cellfun(@(x) (x - mean(x, 2)) ./ std(x, 0, 2), X, 'uni', 0);
n = size(X{1}, 1);
c = numel(unique(Y));

As = cellfun(@(x) constructW_PKN(x, 10), X, 'uni', 0);

[y_pred, obj, coeff, n_g, y_coar, evaltime] = run_EMGC2F(As, c, true);
result = ClusteringMeasure_new(Y, y_pred);
fprintf('time=%f\n', evaltime);
disp(result);
