clc;
clear
close all;
%% fusing CCA in SSVEP frequency detection (demo code)
% we used individual temaplate for reference signals instead of sine-cosine waves for FoCCA
% by    Mohammad Norizadeh Cherloo,
%       Homa Kashefi Amiri,
%       Amir Mohammad Mijani,
%       Liang Zhan,
%       Mohammad Reza Daliri

%% define prameters (Fs,data length, channels and Nh,...)
Fs=250;% sampling rate

% Nine channels that are used for analysis
% [O2, Oz, O1, PO6, PO4, POZ, PO3, PO7, and P8]
chn=[52 53 55 56 57 58 61 62 63];
% number of harmonics
Nh=5;
% data lenght in seconds (0.5,1,1.5,2,2.5 and 3 were considered in our study)
duration=1;
time= linspace(0,6,1500);
position= find(time>=0.5 & time<=0.5+duration); % index of EEG signal

% design a band-pass butterworth filter
[b,a]= butter(3,[8 90]/(Fs/2), 'bandpass');

% define weights for correlation coefficients
sigma1=0.6;
nc=[1:numel(chn)]';
w= exp(-nc/ (2*(sigma1^2)));

% load frequency-phase information of stimuli
load('dataset\Freq_Phase.mat')
fstim= freqs;
% build label for each stimulus which will be used for evaluatoin
freqs= repmat(freqs,1,6);
y_true= repmat(1:40,1,6);

%% SSVEP frequency detection using FoCCA
for sbj=1:35
    load(['dataset/S',num2str(sbj),'.mat/','S',num2str(sbj),'.mat'])
    % concatenate all trials to costruct a 3 dimension matrix
    EEGdata= cat(3,data(:,:,:,1),data(:,:,:,2),data(:,:,:,3),data(:,:,:,4),...
        data(:,:,:,5),data(:,:,:,6));
    clear data
    % preprocessing
    for trial_nm=1:240
        X= EEGdata(:,position,trial_nm)';
        % apply designed band-pass filter[8-90Hz]
        X= filtfilt(b,a,X(:,chn));
        SSVEPdata(:,:,trial_nm)= X';
    end
    clear EEGdata
    % seperating each stimuls trials
    for i=1:numel(fstim)
        indx= find(y_true==i);
        SSVEPdata_sep(:,:,:,i)= SSVEPdata(:,:,indx);
    end
    clear SSVEPdata
    % SSVEP frequency recognition
    n_run=6;
    y_pred=[];
    for run=1:6
        idx_traindata=1:n_run;
        idx_traindata(run)=[];
        % Individual Template calculation for each stimulus
        for i=1:numel(fstim)
            data_i=SSVEPdata_sep(:,:,idx_traindata,i);
            ind_temp(:,:,i)= mean(data_i,3);%Individual_Template
        end
        
         % calculate cannonical correlation between the EEG signal(X) and each of the reference signals(Individual Template)
        for j= 1:numel(fstim)
            for k= 1:numel(fstim)
                ref=[ind_temp(:,:,k)];
                [~,~,r(:,k)] = myCCA(SSVEPdata_sep(:,:,run,j),ref);
            end
            % calculates the comination of correlation coefficients according to equation 4
            W= repmat(w,1,size(r,2));
            Rho= sum(W.*(r.^2));
            % determine the the stimulus frequency of EEG signal(X)
            [v,idx(j)]=max(Rho);
        end
        y_pred=[y_pred,idx];
        idx=[];
    end
    %% Performance evaluation
    C= confusionmat(y_true,y_pred); %cunfusion matrix
    Accuracy(sbj)= sum(diag(C)) / sum(C(:)) *100; % accuracy
    disp(['Accuracy(',num2str(sbj),'): ', num2str(Accuracy(sbj)),' %'])
end
tderror= std( Accuracy ) / sqrt( length( Accuracy ));
Ave_Acc_across_sbjs= mean(Accuracy );
disp(['Average accuracy: ',num2str(mean(Accuracy))...
    ,' ',plusminu,' ',num2str(stderror),' %'])


