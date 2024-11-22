function Error_Sum = Gaussian_LOOCV(X,Y,G_Length)
%Error_Sum is an array where each element is the magnitudal error sum for all sample points for a given Length
%X = sample point coordinates
%Y = Result points
%G_Length = Array of test lengths
    
    %initialise variables
    Num_of_Data_Points = size(X,1);
    Num_of_G_Lengths = length(G_Length);
    Errors = zeros(Num_of_Data_Points,length(G_Length));

    for i = 1:Num_of_Data_Points %loop through all rows of X

        X_test = X(i,:);%choose row to leave out and use as a query point after fitting
        X_fit = X;
        X_fit(i,:) = []; %delete one row from X
        
        Y_test = Y(i);%choose Y value to compare predicted value to
        Y_fit = Y;
        Y_fit(i) = []; %delete one element from Y

        for j = 1:Num_of_G_Lengths %loop through all L values
            phi = buildPhi(X_fit,G_Length(j));
            Weights = (phi\Y_fit).';
            Y_hat= predictRBF(X_fit,X_test,Weights,G_Length(j));
            Errors(i,j) = abs(Y_test - Y_hat);
        end
    end
    Error_Sum = sum(Errors);
end