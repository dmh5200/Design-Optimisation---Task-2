%% Predict mean and standard deviation of RBF at a point x

function [mu, sigma] = predictRBF2(model, x)
    % RBF centers
    X = model.X;
    % RBF weights
    beta = model.beta;
    % RBF width (length scale)
    theta = model.theta;
    % Observed values
    Y = model.Y;

    % Calculate the kernel between the query point and the RBF centers
    K = exp(-theta * pdist2(X, x).^2);
    
    % Calculate the mean
    mu = sum(K' .* beta);
    
    % Calculate the kernel matrix for the RBF centers
    Phi = exp(-theta * pdist2(X, X).^2);
    
    % Calculate the inverse of the kernel matrix
    Phi_inv = inv(Phi + 1e-6 * eye(size(Phi))); % Add a small regularization term for numerical stability
    
    % Calculate the variance at the query point
    k_star = exp(-theta * pdist2(X, x).^2);
    sigma = sqrt(1 - k_star' * Phi_inv * k_star);

    % Ensure sigma is not too small to avoid numerical issues
    sigma = max(sigma, 1e-9);
end