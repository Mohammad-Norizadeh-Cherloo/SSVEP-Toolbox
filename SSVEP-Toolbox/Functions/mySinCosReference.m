function [Xref] = mySinCosReference(fstim,duration,Nh,Fs)
%% Constructs sine-cosine reference signal for each stimulus according to equation 2
% Input:  fstim -- stimuli frequencies (1 x number of stimuli) 
%         duration -- data length in sec 
%         Nh-- number of Harmonics
%         Fs-- sampling rate

% Output: Xref -- Pre-constructed reference signal (2*Nh x points x number of stimuli)

% by    Mohammad Norizadeh Cherloo,
%       Homa Kashefi Amiri,
%       Amir Mohammad Mijani,
%       Liang Zhan,
%       Mohammad Reza Daliri

% Rerefence: 
% A comprehensive study for template-based frequency detection methods in SSVEP-based BCIs

Len= duration * Fs;
t= linspace(0,duration,Len);
Xref= zeros(2*Nh,Len,numel(fstim));
for i= 1:numel(fstim)
    Y=[];
    for n=1:Nh
        tp(1,:)= sin(2*pi*(n*fstim(i))*t);
        tp(2,:)= cos(2*pi*(n*fstim(i))*t);
        Y=[Y;tp];
    end
    Xref(:,:,i) = Y;
end
end

