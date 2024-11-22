%% Predict value of RBF at a point x

function [Yhat] = predictRBF4(model, x)
    % RBF centers
    X = model.X;
    % RBF weights
    beta = model.beta;
    % RBF width (length scale)
    theta = model.theta;

    % Calculate the predicted value
    Yhat = predictRBF(X,x,beta,theta);

    % Calculate the predicted mean and standard deviation
    [mu, sigma] = predictRBF2(model, x);

    % Normalise the predicted value
    Yhat = ((Yhat + mu)/ max(model.Y)) / sigma;

end