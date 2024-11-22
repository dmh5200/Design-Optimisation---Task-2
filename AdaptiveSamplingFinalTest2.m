%% Adaptive sampling strategy based on Bayesian optimisation with RBF model

% Final version for use in black box exercise

% The LHS algorithm is iteratively run externally 10,000 times.
% The best LHS sample is saved to preserve consistency between strategy
% test runs.
% LHS samples will be used to generate the first 5 outputs.
% The radial basis function (RBF) model will be used to predict the output
% Following this, the strategy will use an adaptive algorithm that balances
% exploration and exploitation until 15 samples remain. At this point,
% the algorithm will switch to a purely exploitative strategy to find the
% global minimum.

function [final_min_value, final_min_location, sample_locations, sample_values] = AdaptiveSamplingFinalTest2(ub, lb, res)

% Inputs
% ub: upper bounds of the input variables
% lb: lower bounds of the input variables
% res: resolution of the input variables

% Define strategy parameters
num_variables = 3; % Number of input variables
total_samples = 35; % Total number of samples
initial_samples = 5; % Number of initial samples
final_samples = 15; % Number of final samples
r = 0.4; % Rate of exponential decay for kappa

% specify normalisation function
normalisationFunction = @(x) (x - lb) ./ (ub - lb);

% specify denormalisation function
denormalisationFunction = @(x) x .* (ub - lb) + lb;

% LHS samples

load Latin_Hypercube_3D_10000_iterations_5samples.mat Xout;
lhs_samples = Xout;


% Objective function (black box) outputs manually inputted
% Prompt the user to manually input the black box outputs for the LHS inputs
num_lhs_samples = size(lhs_samples, 1);
initial_outputs = zeros(num_lhs_samples, 1);

% denormalise the LHS samples
lhs_samples_denorm = denormalisationFunction(lhs_samples);

% round LHS samples to correct resolution if required
lhs_samples_denorm = round(lhs_samples_denorm, res);

% normalise the rounded values
lhs_samples = normalisationFunction(lhs_samples_denorm);

% 3D test function for testing
% 3D Rosenbrock function
f = @(x) 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2 + 100*(x(3) - x(2)^2)^2 + (1 - x(2))^2;
for i = 1:num_lhs_samples
    initial_outputs(i) = f(lhs_samples_denorm(i,:));
end

% Fit RBF model
rbf_model = RBF2(lhs_samples, initial_outputs);

%% Adaptive Sampling Strategy

% Define the
%number of additional samples
num_additional_samples = total_samples - initial_samples - final_samples;

% Initialize variables to track the sample values and locations
sample_values = zeros(total_samples, 1);
sample_locations = zeros(total_samples, num_variables);

sample_values(1:initial_samples) = initial_outputs;
sample_locations(1:initial_samples, :) = lhs_samples;

% Set options for fmincon to suppress output
options = optimoptions('fmincon', 'Display', 'off');

if num_additional_samples > 0

    for i = initial_samples+1:initial_samples+num_additional_samples
        % Define acquisition function (EI+)
        acquisition_function = @(x) -expectedImprovementPlus(x, rbf_model, sample_values(1:i), i, r);

        % Define the number of starting points
        num_starting_points = 10;

        % Initialize variables to track the best sample point
        best_sample = [];
        best_acquisition_value = Inf;

        for k = 1:num_starting_points
            % Generate a random starting point
            start_point = rand(1, num_variables);

            % Optimize acquisition function from the starting point
            [sample, acquisition_value] = fmincon(acquisition_function, start_point, [], [], [], [], normalisationFunction(lb), normalisationFunction(ub), [], options);

            % Update the best sample point if a better one is found
            if acquisition_value < best_acquisition_value
                best_sample = sample;
                best_acquisition_value = acquisition_value;
            end
        end

        % Use the best sample point found
        next_sample = best_sample;

        % Evaluate the objective function at the new sample point
        % Denormalize the next sample point
        next_sample_denorm = denormalisationFunction(next_sample);

        % round the next sample to correct resolution if required
        next_sample_denorm = round(next_sample_denorm, res);

        % normalise the rounded value
        next_sample = normalisationFunction(next_sample_denorm);

        % Prompt the user to manually input the black box output for the next sample point
        % Test function
        next_output = f(next_sample_denorm);

        % Update the RBF model with the new sample
        sample_values(i) = next_output;
        sample_locations(i,:) = next_sample;
        rbf_model = RBF2(sample_locations(1:i,:), sample_values(1:i));

    end

end

if final_samples > 0

    % For final 5 samples, switch to purely exploitative strategy
    % Actually only take 4 samples - save final sample for final optimisation
    for i = initial_samples+num_additional_samples+1:total_samples-1
        % Define acquisition function (EI - purely exploitative)
        acquisition_function = @(x) -expectedImprovement(x, rbf_model, sample_values(1:i));

        % Define the number of starting points
        num_starting_points = 10;

        % Initialize variables to track the best sample point
        best_sample = [];
        best_acquisition_value = Inf;

        for k = 1:num_starting_points
            % Generate a random starting point
            start_point = rand(1, num_variables);

            % Optimize acquisition function from the starting point
            [sample, acquisition_value] = fmincon(acquisition_function, start_point, [], [], [], [], normalisationFunction(lb), normalisationFunction(ub), [], options);

            % Update the best sample point if a better one is found
            if acquisition_value < best_acquisition_value
                best_sample = sample;
                best_acquisition_value = acquisition_value;
            end
        end

        % Use the best sample point found
        next_sample = best_sample;

        % Evaluate the objective function at the new sample point
        % Denormalize the next sample point
        next_sample_denorm = denormalisationFunction(next_sample);

        % round the next sample to correct resolution if required
        next_sample_denorm = round(next_sample_denorm, res);

        % normalise the rounded value
        next_sample = normalisationFunction(next_sample_denorm);

        % Prompt the user to manually input the black box output for the next sample point
        % Test function
        next_output = f(next_sample_denorm);

        % Update the RBF model with the new sample
        sample_values(i) = next_output;
        sample_locations(i,:) = next_sample;
        rbf_model = RBF2(sample_locations(1:i,:), sample_values(1:i));

    end

end

%% Minimise based on the full sample set
% Use final sample to check the actual value of this guess

% Define the objective function for the surrogate model
surrogate_objective = @(x) predictRBF3(rbf_model, x);

% Set options for the optimization
options = optimoptions('fmincon', 'Display', 'off');

num_starting_points = 20;
% Initialize variables to track the best sample point
best_sample = [];
best_acquisition_value = Inf;


for k = 1:num_starting_points
    % Generate a random starting point
    start_point = rand(1, num_variables);

    % Optimize the surrogate model from the starting point
    [sample, min_value] = fmincon(surrogate_objective, start_point, [], [], [], [], normalisationFunction(lb), normalisationFunction(ub), [], options);

    % Update the best sample point if a better one is found
    if min_value < best_acquisition_value
        best_sample = sample;
        best_acquisition_value = min_value;
    end
end

        % Use the best sample point found
        next_sample = best_sample;

        % Evaluate the objective function at the new sample point
        % Denormalize the next sample point
        next_sample_denorm = denormalisationFunction(next_sample);

        % round the next sample to correct resolution if required
        next_sample_denorm = round(next_sample_denorm, res);

        % Prompt the user to manually input the black box output for the next sample point
        % Test function
        next_output = f(next_sample_denorm);

        % Update the arrays with the new sample
        sample_values(end) = next_output;
        sample_locations(end,:) = next_sample;

        % Best sample location and value
        final_min_value = min(sample_values);
        final_min_location = sample_locations(sample_values == final_min_value, :);
        final_min_location = denormalisationFunction(final_min_location);


    % Output the final minimum value and location
    disp(['Final minimum value of the surrogate model: ', num2str(final_min_value)]);
    try
        disp(['Location of the final minimum: ', num2str(final_min_location)]);
    catch
        disp('Location of the final minimum: Could not be determined as multiple samples have the same value');
    end


    % % Plot the value of the global minimum over the iterations
    figure;
    plot(1:(total_samples), sample_values, 'r-', 'LineWidth', 2);
    legend('Global Minimum Value');
    xlabel('Iteration');
    ylabel('Value');
    title('Global Minimum Value vs Iteration');

end

%% Helper Functions

function ei_plus = expectedImprovementPlus(x, model, y, i, r)
% Exploration parameter kappa
%r = 0.4; % rate of exponential decay for kappa
kappa = (1/(200^2))*std(y)*exp(-r*i);
[mu, sigma] = predictRBF2(model, x);
y_best = min(y);
z = (y_best - mu - kappa) / sigma;
ei_plus = (y_best - mu - kappa) * normcdf(z) + sigma * normpdf(z);
% plot the expected improvement for visualisation

end

function ei = expectedImprovement(x, model, y)
[mu, sigma] = predictRBF2(model, x);
y_best = min(y);
z = (y_best - mu) / sigma;
ei = (y_best - mu) * normcdf(z);
end


