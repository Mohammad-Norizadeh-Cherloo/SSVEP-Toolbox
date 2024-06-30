clc;
clear
close all;
%% Multilayer Correlation Maximization Model (MCM) in SSVEP frequency detection (demo code)
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
% time samples of EEG signal
% each epoch is 6 secends in which from 0.5 to 5.5 sec is related to SSVEP
position= find(time>=0.5 & time<=0.5+duration);

% design a band-pass butterworth filter
[b,a]= butter(3,[8 90]/(Fs/2), 'bandpass');

% load frequency-phase information of stimuli
load('dataset\Freq_Phase.mat')
fstim= freqs;
% build label for each stimulus which will be used for evaluatoin
freqs= repmat(freqs,1,6);
y_true= repmat(1:40,1,6);
%% Construct sine-cosine reference signal for each stimulus according to equation 2
Xref = mySinCosReference(fstim,duration,Nh,Fs);
%% SSVEP frequency detection using MCM
for sbj=1
    load(['dataset/S',num2str(sbj),'.mat/','S',num2str(sbj),'.mat'])
    EEGdata= cat(3,data(:,:,:,1),data(:,:,:,2),data(:,:,:,3),data(:,:,:,4),...
        data(:,:,:,5),data(:,:,:,6));
    clear data
    % pre-processing
    for i=1:size(EEGdata,3)
        X= EEGdata(chn,position,i)'; % EEG signal
        % apply designed band-pass filter[8-90Hz]
        X= filtfilt(b,a,X);
        SSVEP(:,:,i)= X';
    end
    clear EEGdata

    for i=1:numel(fstim)
        indx= find(y_true==i);
        SSVEPdata_sep(:,:,:,i)= SSVEP(:,:,indx);
    end
    clear SSVEPdata
    %% define MCM parameters
    % number of extracted components for each spatial filter in each layer
    L1=6;
    L2=1;
    L3=1;
    % ********************MCM********************%
    n_run=6;
    y_pred=[];
    for run=1:n_run
        idx_traindata=1:n_run;
        idx_traindata(run)=[];
        %% first layer:
        % extract the stimulus frequency-related information from the EEG samples
        for j= 1:numel(fstim)
            for i=1:numel(idx_traindata)
                Xi=SSVEPdata_sep(:,:,idx_traindata(i),j);
                [Wi,~,~]=myCCA(Xi, Xref(:,:,j));
                Si(:,:,i)= Wi(:,1:L1)'*Xi;
            end
            S(:,:,:,j)= Si;
        end
        %% second layer
        % Reference signals optimization by MsetCCA
        % extract the common features(frequency-related) shared by the spatially filtered data.
        for j= 1:numel(fstim)
            U(:,:,:,j)=myMsetCCA(S(:,:,:,j),L2);
        end
        % Reference signals optimization using extrcted spatial filters
        Zm=zeros((n_run-1)*L2,numel(position),numel(fstim));
        for j= 1:numel(fstim)
            for qq=1:5
                Zm((qq-1)*L2+1:qq*L2,:,j)=U(:,:,qq,j)'*S(:,:,qq,j);
            end
        end
        %% third layer
        % The third layer is to re-optimize the reference signal set
        for m= 1:numel(fstim)
            [Wh,~,~]=myCCA(Zm(:,:,m), Xref(:,:,m));
            Qm(:,:,m)= Wh(:,1:L3)'*Zm(:,:,m);
        end
        %% Recognition (test)
        % calculate cannonical correlation between the EEG signal(X) and each of the reference signals(Qm)
        for j= 1:numel(fstim)
            for k= 1:numel(fstim)
                [wx,wy,r(:,k)]=myCCA(SSVEPdata_sep(:,:,run,j),Qm(:,:,k));
            end
            % calculates the maximum correlation between the EEG signal(X) and each of the reference signals(Xref)
            Rho=max(r);
            % determine the the stimulus frequency of EEG signal(X)
            [v,y_p(j)]=max(Rho);
            r=[];
        end
        y_pred=[y_pred,y_p];
        y_p=[];
    end
    %% Performance evaluation
    C= confusionmat(y_true,y_pred); %cunfusion matrix
    Accuracy(sbj)= sum(diag(C)) / sum(C(:)) *100; % accuracy
    disp(['Accuracy(',num2str(sbj),'): ', num2str(Accuracy(sbj)),' %'])
end
plusminu=char(177);
stderror= std( Accuracy ) / sqrt( length( Accuracy ));
tderror= std( Accuracy ) / sqrt( length( Accuracy ));
Ave_Acc_across_sbjs= mean(Accuracy );
disp(['Average accuracy: ',num2str(mean(Accuracy))...
    ,' ',plusminu,' ',num2str(stderror),' %'])


