function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


tmp = 0;
sig = 0;
tmp1 = 0;

for i = 1 : m
  sig = sigmoid(theta' * X'(:, i));
  tmp = tmp - y(i) * log (sig) - (1 - y(i)) * log (1 - sig);
end

for j = 2 : size(theta)
  tmp1 = tmp1 + theta(j) ^ 2;
end

J = (1 / m) * tmp + (lambda / (2 * m)) * tmp1;

tmp = 0;
for i = 1 : m
  tmp = tmp + (sigmoid(theta' * X'(:, i)) - y(i)) * X(i, 1);
end
grad(1) = (1 / m) * tmp;


for j = 2 : size(theta)
  tmp = 0;
  for i = 1 : m
    tmp = tmp + (sigmoid(theta' * X'(:, i)) - y(i)) * X(i, j);
  end
  grad(j) = (1 / m) * tmp + (lambda / m) * theta(j);
end



% =============================================================

end
