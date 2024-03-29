function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m, 1) X];


Hid_X = sigmoid((X*(Theta1)')); % 5000x401 * 401x25

Hid_X = [ones(m,1) Hid_X];  %  Hid_X   5000x26

Out_X = sigmoid((Hid_X*(Theta2)'));  % 5000x26 26x10 

[p_max,p_pos] = max(Out_X,[],2); % position of max value in p gives the label 

p = p_pos;









% =========================================================================


end
