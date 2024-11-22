function [output] = ThreeCamel(lhs_samples,lb,ub)

    % Convert to denormalised values in the range [lb,ub]
    % specify denormalisation function
    denormalisationFunction = @(x) x .* (ub - lb) + lb;
    lhs_samples = denormalisationFunction(lhs_samples);
    
    x = lhs_samples(:,1);
    y = lhs_samples(:,2);
    
    
    f = 2*x.^2 - 1.05*x.^4 + x.^6/6 + x.*y + y.^2;
    output = f;
    
    end