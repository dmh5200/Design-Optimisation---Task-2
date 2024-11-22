function plotIsoSurface()

load final_min_location.mat final_min_location
load final_min_value.mat final_min_value
load sample_locations.mat sample_locations
load sample_values.mat sample_values

% % Plot the value of the global minimum over the iterations
figure;
plot(1:36, sample_values, 'r-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 28);
ylabel('Value', 'FontSize', 28);

% Set the text size for the axis numbers (tick labels)
ax = gca;
ax.FontSize = 22;


%% Minimise based on the full sample set
% Use final sample to check the actual value of this guess

% Define the objective function for the surrogate model
rbf_model = RBF2(sample_locations(1:end-1,:), sample_values(1:end-1));







%this script will give you an example of how to plot IsoSurface - perfect
%for plotting the objective function topology with 3 input variables!

%fine sample in the three inout variables
x = linspace(0,1,100);
y = linspace(0,1,100);
z = linspace(0,1,100);

%create grid sampling
[X,Y,Z] = meshgrid(x,y,z);

%outout objective value - in this case it's the spherical toy function
%F = X.^2 + Y.^2 + Z.^2;

% Generate output values for the grid
X_input = [X(:), Y(:), Z(:)];
F = predictRBF3(rbf_model, X_input);

F = reshape(F,size(X));

figure;

%find the min and max objective funciton levels
minF = min(min(min(F)));
maxF = max(max(max(F)));

%set the levels between the max and min F value
noLevels = 20;
levels = linspace(minF+0.01*(maxF-minF),maxF,noLevels);
for i = 1:length(levels)
    p{i} = patch(isosurface(X, Y, Z, F, levels(i)));
    p{i}.FaceColor = [0.55 0.55 0.95]; % RGB color
    p{i}.EdgeColor = 'none'; % No edges
    %make higher levels of the objective funciton more transparent
    p{i}.FaceAlpha = 0.2/i; % Adjust transparency (0 = fully transparent, 1 = opaque)
end

% Highlight the predicted minimum location
hold on;
scatter3(final_min_location(1)+0.05, final_min_location(2)+0.05, final_min_location(3), 200, 'k', 'filled'); % Increase size to 200 and change color to black

h1 = text(final_min_location(1)+0.05, final_min_location(2)+0.05, final_min_location(3) + 0.1, ...
    sprintf('Min value: %.2f', final_min_value), 'FontSize', 28, 'Color', 'k', 'BackgroundColor', 'w');

% Label the predicted minimum location with its location
h2 = text(final_min_location(1), final_min_location(2), final_min_location(3) - 0.05, ...
    sprintf('Min location(%.2f, %.2f, %.2f)', round(final_min_location(1),2), round(final_min_location(2),2), round(final_min_location(3),2)), 'FontSize', 28, 'Color', 'k', 'BackgroundColor', 'w');

% Bring the text labels to the front
uistack(h1, 'top');
uistack(h2, 'top');

% Set the text size for the axis numbers (tick labels)
ax = gca;
ax.FontSize = 22;

xlabel('X1','FontSize',28)
ylabel('X2','FontSize',28)
zlabel('X3','FontSize',28)

% Adjust the view angle for better visualization
view(3);
axis tight;
camlight;
lighting gouraud;

hold off;

end