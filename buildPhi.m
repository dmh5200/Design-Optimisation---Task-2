function phi = buildPhi(X,L)
%phi is a N x N matrix
%X is matrix of sample point
%L is length scaling
    Num_of_Data_Points = size(X,1);
    phi = zeros(Num_of_Data_Points);
    for i = 1:Num_of_Data_Points
        for j = 1:Num_of_Data_Points
            phi(i,j) = GaussianRBF(X(i,:),X(j,:),L);% call your gaussian RBF function wit input X(i,:) and X(j,:), along with length scale L
        end
    end
end