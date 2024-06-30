function [w,rho] = myCORCA(X,Y)
%% Correlated component Analysis
% Input:  X -- EEG signal (channels x points) 
%         Y -- reference signal(2*Nh x points) " Nh= number of Harmonics"
% Output: W -- spatial filters
%         rho --  maximum correlation between the EEG signal and the reference signal

% by    Mohammad Norizadeh Cherloo,
%       Homa Kashefi Amiri,
%       Amir Mohammad Mijani,
%       Liang Zhan,
%       Mohammad Reza Daliri

% Rerefence: 
% A comprehensive study for template-based frequency detection methods in SSVEP-based BCIs

%% calculate covariance matrixs
XY= [X,Y];
Cv= cov(XY);
p= size(X,2);
R11= Cv(1:p,1:p);
R22= Cv(p+1:end,p+1:end);
R12= Cv(1:p,p+1:end);
R21= Cv(p+1:end,1:p);
%% build your eigen value decomposition problem
C= pinv(R22+R11) * (R12+R21) ;
%% eigen value decomposition
[U,r]= eig(C);
%% diag,sort, sqrt
V = fliplr(U);		% reverse order of eigenvectors
r = flipud(diag(r));	% extract eigenvalues and reverse their order
[r,I]= sort((real(r)));	% sort reversed eigenvalues in ascending order
r = flipud(r);		% restore sorted eigenvalues into descending order
for j = 1:length(I)
  U(:,j) = V(:,I(j));  % sort reversed eigenvectors in ascending order
end
U = fliplr(U);	% restore sorted eigenvectors into descending order
w=U(:,1);
r= sort(r,'descend');
rho= real(sqrt(r(1:p)));
end

