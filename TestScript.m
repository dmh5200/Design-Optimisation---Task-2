%% Test script

% Run test script 100 times to evaluate the performance of the adaptive sampling strategy
num_runs = 100;
global_min_values = zeros(num_runs, 1);
global_min_locations = zeros(num_runs, 2);
goodness_scores = zeros(num_runs, 1);
best_goodness_scores = zeros(num_runs, 1);

% Display menu for user to select the black box function
choice = menu('Select the black box function:', 'Rosenbrock w/ circle constraint', 'BlackBox2');

% Set the black box function handle based on user choice
switch choice
    case 1
        blackBoxFunction = @Rosenbrock_circle_constraint;
    case 2
        blackBoxFunction = @BlackBox2;
    otherwise
        error('Invalid selection');
end

% Display menu for user to select the strategy function
choice = menu('Select the strategy function:', 'Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4', 'Strategy 5', 'Strategy 6', 'Strategy 7', 'Strategy 8', 'Strategy 9', 'Strategy 10', 'Strategy 11', 'Strategy 12', 'Strategy 13');

% Set the strategy function handle based on user choice
switch choice
    case 1
        strategyFunction = @AdaptiveSampling;
        Out = 1;
    case 2
        strategyFunction = @AdaptiveSampling2;
        Out = 1;
    case 3
        strategyFunction = @AdaptiveSampling3;
        Out = 1;
    case 4
        strategyFunction = @AdaptiveSampling4;
        Out = 1;
    case 5
        strategyFunction = @AdaptiveSampling5;
        Out = 1;
    case 6
        strategyFunction = @AdaptiveSampling6;
        Out = 1;
    case 7
        strategyFunction = @AdaptiveSampling7;
        Out = 2;
    case 8
        strategyFunction = @AdaptiveSampling8;
        Out = 2;
    case 9
        strategyFunction = @AdaptiveSampling9;
        Out = 2;
    case 10
        strategyFunction = @AdaptiveSampling10;
        Out = 2;
    case 11
        strategyFunction = @AdaptiveSampling11;
        Out = 2;
    case 12
        strategyFunction = @AdaptiveSampling12;
        Out = 2;
    case 13
        strategyFunction = @AdaptiveSampling13;
        Out = 2;
    otherwise
        error('Invalid selection');
end

if Out == 1
    for i = 1:num_runs
        [global_min_values(i), global_min_locations(i, :), goodness_scores(i), best_goodness_scores(i)] = strategyFunction(blackBoxFunction);
        fprintf('Run %d: End Goodness Score = %.4f Best Goodness Score = %.4f\n', i, goodness_scores(i), best_goodness_scores(i));
    end
elseif Out == 2
    for i = 1:num_runs
        [global_min_values(i), global_min_locations(i, :), goodness_scores(i), best_goodness_scores(i)] = strategyFunction(blackBoxFunction);
        fprintf('Run %d: End Goodness Score = %.4f Best Goodness Score = %.4f\n', i, goodness_scores(i), best_goodness_scores(i));
    end
end

% Calculate the mean and standard deviation of the goodness scores
mean_goodness_score = mean(goodness_scores);
std_goodness_score = std(goodness_scores);

% Plot the distribution of goodness scores and overlay the mean and standard deviation
figure;
histogram(goodness_scores, 20);
xlabel('Goodness Score');
ylabel('Frequency');
title('Distribution of Goodness Scores');
hold on;
line([mean_goodness_score, mean_goodness_score], ylim, 'Color', 'r', 'LineWidth', 2);
line([mean_goodness_score - std_goodness_score, mean_goodness_score - std_goodness_score], ylim, 'Color', 'g', 'LineWidth', 2);
line([mean_goodness_score + std_goodness_score, mean_goodness_score + std_goodness_score], ylim, 'Color', 'g', 'LineWidth', 2);
legend('Mean', 'Mean - Std Dev', 'Mean + Std Dev');
hold off;

% Calculate the mean and standard deviation of the best goodness scores
mean_best_goodness_score = mean(best_goodness_scores);
std_best_goodness_score = std(best_goodness_scores);

% Plot the distribution of best 'goodness' scores
figure;
histogram(best_goodness_scores, 20);
xlabel('Best Goodness Score');
ylabel('Frequency');
title('Distribution of Best Goodness Scores');
hold on;
line([mean_best_goodness_score, mean_best_goodness_score], ylim, 'Color', 'r', 'LineWidth', 2);
line([mean_best_goodness_score - std_best_goodness_score, mean_best_goodness_score - std_best_goodness_score], ylim, 'Color', 'g', 'LineWidth', 2);
line([mean_best_goodness_score + std_best_goodness_score, mean_best_goodness_score + std_best_goodness_score], ylim, 'Color', 'g', 'LineWidth', 2);
legend('Mean', 'Mean - Std Dev', 'Mean + Std Dev');
hold off;



