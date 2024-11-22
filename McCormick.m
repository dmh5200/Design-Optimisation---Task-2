function [output] = McCormick(lhs_samples,lb,ub)

    % Convert to denormalised values in the range [lb,ub]
    % specify denormalisation function
    denormalisationFunction = @(x) x .* (ub - lb) + lb;
    lhs_samples = denormalisationFunction(lhs_samples);
    
    x = lhs_samples(:,1);
    y = lhs_samples(:,2);
    
    
    f = sin(x + y) + (x - y).^2 - 1.5*x + 2.5*y + 1;
    output = f;
    
    end