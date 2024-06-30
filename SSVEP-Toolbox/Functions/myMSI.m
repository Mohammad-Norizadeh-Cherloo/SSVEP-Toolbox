function S = myMSI(X,Y)
%% Multivariate synchronization index (MSI)
% Input:  X -- EEG signal (channels x points) 
%         Y -- reference signal(2*Nh x points) " Nh= number of Harmonics"
% Output:      
%         S -- Synchronization index between EEG signal and reference signal

% by    Mohammad Norizadeh Cherloo,
%       Homa Kashefi Amiri,
%       Amir Mohammad Mijani,
%       Liang Zhan,
%       Mohammad Reza Daliri

% Rerefence: 
% A comprehensive study for template-based frequency detection methods in SSVEP-based BCIs


[n,m]=size(X);
% covariance matrix calculation according to equation 35-37
% covariance matrix: equation (35)
k=(1/m);
c12=k*(X*Y');
c21=c12';
c11=k*(X*X');
c22=k*(Y*Y');
C=[c11 c12;c21 c22];
C11=c11^-0.5;
C22=c22^-0.5;
% linear transformation: equation (36)
U=[C11, zeros(size(C11,2),size(C22,2));
    zeros(size(C22,2),size(C11,2)),C22];
% transformed covariance matrix: equation (37)
R=U*C*U';

% eigen value decomposition
[~,D]=eig(R);
% normalize eigenvalues according to equation (38)
e= diag(D);
E=e/sum(e);
% calculate the synchronization index between 
% the EEG signal and the reference signal according to equation (39)
S=1+(sum(E.*log(E)))/log(7);
end






