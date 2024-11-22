function [X_Out] = LatinHyperCube(Sample_Points, Dimensions, Minimums, Maximums,Rounding,Runs)
    %Outputs: 
        %X_Out - matrix containing the best generation of scaled sample from
        %latinhyper cube. Rows are the Ref. of sample point and columns are
        %dimensions
    %Inputs:
        %Sample_Points - number of sample points to generate per dimension
        %Dimensions - number of dimensions
        %Minimums - array of minimums for each range, i is dimension no.
        %Maximums - array of maximums for each range ,i is dimension no.
        %Rounding - 0 is integer rounding, >0 or <0 moves the decimal place
        %Runs - number of generations
   
    for k = 1:Runs
        %generate latin hypercube sample points
        X = lhsdesign(Sample_Points,Dimensions);
        X_Size = size(X);
        
        %Validate and process scaling input data 
        if((Dimensions ~= length(Minimums)) || (Dimensions ~= length(Maximums)))
            disp('Error: Incorrect Dimensions/Minimums/Maximums input, check array sizes!')
            return
        else
            New_Range = Maximums - Minimums;
        end
        
        %scale the sample points to desider range
        for i = 1:X_Size(1) %iterate through rows
            for j = 1:X_Size(2) %iterate through cloumns
                %First multiply sample point (0-1) by maximum value of
                %range, then add it to the minimum value of the range,
                %finally round the value
                X(i,j) = round((X(i,j)*New_Range(j)) + Minimums(j), Rounding);
            end
        end
        
        %generate star delta rating on sample point filling quality
        sD = starD(X);

        if(k==1) %On the first loop initialise X_out and sD_Lowest
            X_Out = X;
            sD_Lowest = sD;
        else % Update variables on further iterations
            %compare current sd to previously lowest sd
            %current lowest sd found then assign X to X_Out
            if(sD<sD_Lowest)
                sD_Lowest = sD;
                X_Out = X;
            end
        end
    end

    %plot histogram
    histogram(sD)
end