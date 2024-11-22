%% Adaptive sampling strategy based on Bayesian optimisation with RBF model

% Version 4 includes improved Latin Hypercube Sampling (LHS) and a different overall strategy
% The LHS algorithm is iteratively run externally 10,000 times.
% The best LHS sample is saved to preserve consistency between strategy
% test runs.
% Additionally, the startegy will use an adaptive algorithm that balances
% exploration and exploitation until only 5 samples remain. At this point,
% the algorithm will switch to a purely exploitative strategy to find the
% global minimum.

function [final_min_value, final_min_location, final_goodness_score, goodness_scores, sample_locations, sample_values] = AdaptiveSampling13(blackBoxFunction, ub, lb, visualisation, known_global_min, initial_samples, final_samples, r)

num_variables = 2; % Number of input variables
total_samples = 35; % Total number of samples
%initial_samples = 15; % Number of initial samples
%final_samples = 5; % Number of final samples

% specify normalisation function
normalisationFunction = @(x) (x - lb) ./ (ub - lb);

% specify denormalisation function
denormalisationFunction = @(x) x .* (ub - lb) + lb;

% LHS samples
if initial_samples == 15
    load Latin_Hypercube_2D_10000_iterations.mat X_Out;
    lhs_samples = X_Out;
elseif initial_samples == 5
    load Latin_Hypercube_2D_10000_iterations_5samples.mat Xout;
    lhs_samples = Xout;
elseif initial_samples == 25
    load Latin_Hypercube_2D_10000_iterations_25samples.mat Xout;
    lhs_samples = Xout;
elseif initial_samples == 35
    load Latin_Hypercube_2D_10000_iterations_35samples.mat Xout;
    lhs_samples = Xout;
elseif initial_samples == 2
    lhs_samples = [0,0;1,1];
else
    lhs_samples = lhsdesign(initial_samples, num_variables);

end

% Objective function (black box) outputs manually inputted
% For testing, replace with toy function: ...
% Using constrained Rosenbrock function

[initial_outputs] = blackBoxFunction(lhs_samples, lb, ub);

% Fit RBF model
rbf_model = RBF2(lhs_samples, initial_outputs);

%% Testing visualisation for 2d functions

if visualisation == true
    if num_variables ~= 2
        disp('Visualisation is only supported for 2D functions');
    else
        % Generate a grid of points
        [x, y] = meshgrid(linspace(0, 1 , 200), linspace(0, 1, 200));
        z = zeros(size(x));
        for i = 1:length(x)
            for j = 1:length(x)
                z(i,j) = blackBoxFunction([x(i,j), y(i,j)], lb, ub);
            end
        end

        % Evaluate the predicted function values over the grid of points
        predicted_z = zeros(length(x), length(y));
        for i = 1:size(x, 1)
            for j = 1:size(x, 2)
                predicted_z(i, j) = predictRBF3(rbf_model, [x(i, j), y(i, j)]);
            end
        end

        % Plot the test function as a surface
        figure;
        surf(x, y, z, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        % Apply a colormap to the actual function
        colormap('parula');
        hold on;
        %
        % Plot the predicted function as a mesh on top of the see-through surface
        mesh(x, y, predicted_z, 'EdgeColor', 'k');

        xlabel('x', 'FontSize', 18);
        ylabel('y', 'FontSize', 18);
        zlabel('Objective Function Value', 'FontSize', 18);

        % Set the text size for the axis numbers (tick labels)
        ax = gca;
        ax.FontSize = 16;

        % Apply a different colormap to the predicted function
        colormap('hot');
        % Add lighting to enhance contrast
        lightangle(-45, 30);
        lighting gouraud;
        material dull;
        hold off;
        legend('Test Function', 'Predicted Function', 'FontSize', 18);
    end
end

%% Adaptive Sampling Strategy

% Define the
%number of additional samples
num_additional_samples = total_samples - initial_samples - final_samples;

% Initialize variables to track the sample values and locations
sample_values = zeros(total_samples, 1);
sample_locations = zeros(total_samples, num_variables);

sample_values(1:initial_samples) = initial_outputs;
sample_locations(1:initial_samples, :) = lhs_samples;

% Introduce 'goodness' score for testing purposes
% Normalised score based on Euclidean distance to known global minimum
goodness_scores = zeros(total_samples, 1);
% Known global minimum location for the test function
known_global_min_location = known_global_min;
% normalise the known global minimum location
known_global_min_location = normalisationFunction(known_global_min_location);




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
    next_output = blackBoxFunction(next_sample, lb, ub);

    % Update the RBF model with the new sample
    sample_values(i) = next_output;
    sample_locations(i,:) = next_sample;
    rbf_model = RBF2(sample_locations(1:i,:), sample_values(1:i));


    % Update the 'goodness' score based on the known global minimum
    goodness_scores(i) = 1 - sqrt((known_global_min_location(1) - sample_locations(i,1))^2 + (known_global_min_location(2) - sample_locations(i,2))^2);
end

end

if final_samples > 0

% For final 5 samples, switch to purely exploitative strategy
for i = initial_samples+num_additional_samples+1:total_samples
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
    next_output = blackBoxFunction(next_sample, lb, ub);

    % Update the RBF model with the new sample
    sample_values(i) = next_output;
    sample_locations(i,:) = next_sample;
    rbf_model = RBF2(sample_locations(1:i,:), sample_values(1:i));


    % Update the 'goodness' score based on the known global minimum
    goodness_scores(i) = 1 - sqrt((known_global_min_location(1) - sample_locations(i,1))^2 + (known_global_min_location(2) - sample_locations(i,2))^2);
end

end

%% Minimise based on the full sample set

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

final_min_location_norm = best_sample;
final_goodness_score = 1 - sqrt((known_global_min_location(1) - final_min_location_norm(1))^2 + (known_global_min_location(1) - final_min_location_norm(2))^2);
% denormalise the final minimum location
final_min_location = denormalisationFunction(final_min_location_norm);
final_min_value =blackBoxFunction(final_min_location_norm, lb, ub);


if visualisation == true
    % Output the final minimum value and location
    disp(['Final minimum value of the surrogate model: ', num2str(final_min_value)]);
    disp(['Location of the final minimum: ', num2str(final_min_location)]);
    disp(['Goodness score of the final minimum: ', num2str(final_goodness_score)]);

    % Plot the 'goodness' scores over the iterations
    figure;
    subplot(2, 1, 2);
    plot(1:(total_samples), goodness_scores, 'b-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Goodness Score');
    % % Plot the value of the global minimum over the iterations
    subplot(2, 1, 1);
    plot(1:(total_samples), sample_values, 'r-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Sample Output Value');
end

%% Testing visualisation

if visualisation == true
    if num_variables ~= 2
        disp('Visualisation is only supported for 2D functions');
    else
        % Generate a grid of points
        [x, y] = meshgrid(linspace(0, 1, 200), linspace(0, 1, 200));
        for i = 1:size(x, 1)
            for j = 1:size(x, 2)
                z(i, j) = blackBoxFunction([x(i, j), y(i, j)], lb, ub);
            end
        end

        % Evaluate the predicted function values over the grid of points
        predicted_z = zeros(size(x));
        for i = 1:size(x, 1)
            for j = 1:size(x, 2)
                predicted_z(i, j) = predictRBF3(rbf_model, [x(i, j), y(i, j)]);
            end
        end

        % Plot the test function as a surface
        figure;
        surf(x, y, z, 'EdgeColor', 'none');
        colormap('parula');
        hold on;

        % Plot the predicted function as a mesh on top of the see-through surface
        mesh(x, y, predicted_z,'FaceAlpha', 0.3, 'EdgeColor', 'k');

        % Apply a different colormap to the predicted function
        colormap('hot');
        % Add lighting to enhance contrast
        lightangle(-45, 30);
        lighting gouraud;
        material dull;

        % Plot the progression of the predicted global minimum locations
        plot3([sample_locations(:, 1); final_min_location_norm(1)], [sample_locations(:, 2); final_min_location_norm(2)], [sample_values; final_min_value], 'r.-', 'LineWidth', 2, 'MarkerSize', 15);

        % Highlight the final predicted global minimum location
        plot3(final_min_location_norm(1), final_min_location_norm(2), final_min_value, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');

        % Highlight the known global minimum location
        plot3(known_global_min_location(1), known_global_min_location(2), blackBoxFunction(known_global_min_location, lb, ub), 'mo', 'MarkerSize', 15, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k');
        % Set the text size for the axis numbers (tick labels)
        ax = gca;
        ax.FontSize = 22;
        
        % % Add labels and title
        xlabel('x', 'FontSize', 26);
        ylabel('y', 'FontSize', 26);
        zlabel('Objective Function Value', 'FontSize', 26);
        legend('Test Function', 'Predicted Function', 'Progression of sample locations','Predicted Global Minimum', 'Known Global Minimum', 'FontSize', 26);    


        hold off;

    end
end


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

function model = fitrbf(X, y)
% Fit an RBF model (you can use MATLAB's fitrgp or other toolboxes)
model = fitrgp(X, y);
end
