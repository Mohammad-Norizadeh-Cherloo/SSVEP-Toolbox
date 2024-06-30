function [W]= myTRCA(eeg_signal,n_components)
% Task-related component analysis (TRCA)
%
% function W = myTRCA(eeg_signal)
%
% Inputs: eeg_signal -- EEG signal data (channels x points x trials) 
%         n_components --number of components
%
% Output: W -- spatial filters(Weight coefficients for electrodes)
 

% by    Mohammad Norizadeh Cherloo,
%       Homa Kashefi Amiri,
%       Amir Mohammad Mijani,
%       Liang Zhan,
%       Mohammad Reza Daliri

% Rerefence: 
% A comprehensive study for template-based frequency detection methods in SSVEP-based BCIs

[n_channels, n_points, n_trials]  = size(eeg_signal);

%% calculate  matrix S(t) according to equation 28
S = zeros(n_channels);
for i = 1:1:n_trials-1
    trial_i=eeg_signal(:,:,i);
    x1 = squeeze(trial_i);
    % normalize
    x1 = bsxfun(@minus, x1, mean(x1,2));
    for j = i+1:1:n_trials
        trial_j= eeg_signal(:,:,j);
        x2 = squeeze(trial_j);
        % normalize
        x2 = bsxfun(@minus, x2, mean(x2,2));
        s=x1*x2' + x2*x1';
        S = S + s;
    end
end 

UX = reshape(eeg_signal, n_channels, n_points*n_trials);
% normalize
UX = bsxfun(@minus, UX, mean(UX,2));
%% calculate the covariance matrix of continuous data according to equation 32
Q = UX*UX';
%% eigen value decomposition
[U,~] = eigs(S, Q);
W= U(:,1:n_components);
end



