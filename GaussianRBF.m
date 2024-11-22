function RBF = GaussianRBF(x1,x2,L)
     RBF = exp(-((norm(x1 - x2))/L)^2);
end