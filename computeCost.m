function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

temp1 = X' * X(:,2) - [ones(1,m) * X(:,2) - m; 0];
temp2 = X' * y;
temp3 = ones(1,m) * (y .^ 2);

J = (1 / (2 * m)) * ((theta' .^ 2) * temp1 - 2 * theta' * temp2 + temp3);



% =========================================================================

end
