%% Radial Basis Function
% Takes inputs X and Y, and query point Xq, and predicts the value of Y at Xq

function [model] = RBF2(X,Y)

%LOOCV to find error sums for each validation
%res = 1e-4;
%length_range = 0:res:5;
%Error_Sum = Gaussian_LOOCV(X,Y,length_range);
%Calculate L using Franke's formula
L_Frankes = Frankes_Formula2(X,Y);
%Determine optimum L, combining the result of Franke's and LOOCV error sums 
%[~,Optimum_L] = min(Error_Sum((L_Frankes==Error_Sum)-20:res:(L_Frankes==Error_Sum)+20));
%Optimum_L = L_Frankes + (Optimum_L - 2);
% Find the index of the minimum error sum
%[~, min_error_index] = min(Error_Sum);

%LOOCV_L = length_range(min_error_index);

Optimum_L = L_Frankes;

%formula to predict a 'y' point for any query point 
phi = buildPhi(X,Optimum_L);
Weights = (phi\Y).';
%Yhat = predictRBF(X,Xq,Weights,Optimum_L);

% define model structure
model.X = X;
model.Y = Y;
model.theta = Optimum_L; % RBF width (length scale)
model.beta = Weights; % RBF weights

end