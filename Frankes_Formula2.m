function L_Frankes = Frankes_Formula2(X,Y)
%L_Frankes is the Length value determined by frankes formula
%X = sample point coordinates
%Y = resulting values
    %X = [X,Y];
    Pairs = nchoosek(1:size(X,1),2);
    Frankes_D = 0;
    for i = 1:size(Pairs,1)
        Euclean_D = norm(X(Pairs(i,1),:) - X(Pairs(i,2),:));
        if Euclean_D>Frankes_D
            Frankes_D = Euclean_D;
        end
    end
    L_Frankes = Frankes_D/(0.8*nthroot(size(X,1),4));
end
