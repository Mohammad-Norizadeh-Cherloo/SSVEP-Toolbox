function [Wx,Wy,r] = myCCA(X,Y)
%% Standard CCA
% Input:  X -- EEG signal (channels x points) 
%         Y -- reference signal(2*Nh x points) " Nh= number of Harmonics"
% Output: Wx -- spatial filters(weigh vector of X) 
%         Wy -- spatial filters(weigh vector of Y)
%         r --  maximum correlation between the EEG signal and the reference signal

% by    Mohammad Norizadeh Cherloo,
%       Homa Kashefi Amiri,
%       Amir Mohammad Mijani,
%       Liang Zhan,
%       Mohammad Reza Daliri

% Rerefence: 
% A comprehensive study for template-based frequency detection methods in SSVEP-based BCIs

%% calculate covariance matrixs
z = [X;Y];
C = cov(z.');
sx = size(X,1);
sy = size(Y,1);
Cxx = C(1:sx, 1:sx) + 10^(-8)*eye(sx);
Cxy = C(1:sx, sx+1:sx+sy);
Cyx = Cxy';
Cyy = C(sx+1:sx+sy, sx+1:sx+sy) + 10^(-8)*eye(sy);
invCyy = inv(Cyy);
C=inv(Cxx)*Cxy*invCyy*Cyx;

%% eigen value decomposition
[Wx,r]= eig(C);
%% diag,sort, sqrt
r = sqrt(real(r));      % Canonical correlations
% --- Sort correlations ---
V = fliplr(Wx);		% reverse order of eigenvectors
r = flipud(diag(r));	% extract eigenvalues and reverse their order
[r,I]= sort((real(r)));	% sort reversed eigenvalues in ascending order
r = flipud(r);		% restore sorted eigenvalues into descending order
for j = 1:length(I)
    Wx(:,j) = V(:,I(j));  % sort reversed eigenvectors in ascending order
end
Wx = fliplr(Wx);	% restore sorted eigenvectors into descending order
% --- Calcualte Wy  ---
Wy = inv(Cyy)*Cyx*Wx;     % Basis in Y
Wx=Wx(:,1:end);
end
