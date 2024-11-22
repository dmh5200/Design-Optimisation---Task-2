function Yhat = predictRBF(X,Xq,W,L)
%X known data point position
%Xq query point
%W is RBF weights
%L is length scale
    Yhat = zeros(1,size(Xq,1));%initialise yhat array
    for i = 1:size(Xq,1) %for query point, sum the scaled guassian predicitons
        for j = 1:length(W)
            Yhat(i) = Yhat(i) + W(j)*GaussianRBF(Xq(i,:),X(j,:),L); %call your gaussian RBF function with input Xq(i,:) and X(j,:), along with length scaleL
        end
    end
end