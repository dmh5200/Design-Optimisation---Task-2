function [output] = Rosenbrock_2(lhs_samples,lb,ub)

% Convert to denormalised values in the range [lb,ub]
% specify denormalisation function
denormalisationFunction = @(x) x .* (ub - lb) + lb;
lhs_samples = denormalisationFunction(lhs_samples);

x = lhs_samples(:,1);
y = lhs_samples(:,2);


f = ((1-x).^2)+(100.*((y - x.^2).^2));
output = f;

end