%% Predict value of RBF at a point x

function [Yhat] = predictRBF3(model, x)
    % RBF centers
    X = model.X;
    % RBF weights
    beta = model.beta;
    % RBF width (length scale)
    theta = model.theta;

    % Calculate the predicted value
    Yhat = predictRBF(X,x,beta,theta);
end