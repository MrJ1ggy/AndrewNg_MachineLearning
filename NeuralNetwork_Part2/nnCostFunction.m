function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%%%%%%Calculating Ho(x)

a1 = [ones(m, 1) X]; % 5000x401

z2 = a1*(Theta1)'; % 5000x401 * 401x25

Hid_X = sigmoid(z2); 

a2 = [ones(m,1) Hid_X];  %  Hid_X   5000x26

z3 = a2*(Theta2)'; % 5000x26 26x10 

a3 = sigmoid(z3);  


%%%%%%%%%%%%%%%%%%%55

y_new = zeros(m,num_labels);

for i = 1:size(y,1),
  y_new(i,y(i)) = 1;  
endfor

for i = 1:m,
  for k = 1:num_labels,
    J = J + ((y_new(i,k)*log(a3(i,k))) + ((1-y_new(i,k))*log(1-a3(i,k)))) ;
  endfor
endfor

J = (-1*J)/m + ((lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))); %calculating cost with regularization
%%%%%using (:,2:end) would mean that theta of bias unit is not calculated means theta0 is ignored
%%%%%%%%%%%%%%%%%%%%Part 1 complete%%%%%%%%%%%%%%%%%%%%%%%%%55
%a2 = 5000x26
% a3 = 5000x10 
% y_new = 5000x10
% theta2 = 10x26
%Hid_x = 5000x25
%z2 = 5000x25

Intuition3 = a3 - y_new;   %5000x10

Intuition2 = ((Intuition3*Theta2(:,2:end)).*sigmoidGradient(z2)); % 5000x25

delta1 = (Intuition2)'*a1;  %25x401
delta2 = (Intuition3)'*a2;  %10x26


rg1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
rg2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];


Theta1_grad = delta1./m + rg1;

Theta2_grad = delta2./m + rg2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
