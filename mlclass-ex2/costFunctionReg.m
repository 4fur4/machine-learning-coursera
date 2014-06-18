function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
n = size(theta);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

factor = 0;
for i=2:n
    factor = factor + theta(i)^2;
endfor
factor = (lambda/(2*m))*factor;
for i=1:m
    J = J + (- y(i)*log(sigmoid(X(i,:)*theta)) - (1-y(i))*log(1 -sigmoid(X(i,:)*theta)));
endfor
J = (1/m)*J + factor;

for j=1:n
 factor =0;
   for i=1:m
    factor = factor + (sigmoid(X(i,:)*theta) - y(i))*X(i,j);
   endfor
   if(j==1)
     grad(j) = (1/m)* factor;
    else
     grad(j) = (1/m)* factor +(lambda/m)*theta(j);
    endif
endfor


% =============================================================

end
