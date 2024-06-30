clc;
clear
close all;
%% Multi-way CCA (MCCA) in SSVEP frequency detection (demo code)
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

% load frequency-phase information of stimuli
load('dataset\Freq_Phase.mat')
fstim= freqs;
% build label for each stimulus which will be used for evaluatoin
freqs= repmat(freqs,1,6);
y_true= repmat(1:40,1,6);
%% Construct sine-cosine reference signal for each stimulus according to equation 2
Xref = mySinCosReference(fstim,duration,Nh,Fs);
%% SSVEP frequency detection using Multi-way CCA (MCCA)
for sbj=1:35
    load(['dataset/S',num2str(sbj),'.mat/','S',num2str(sbj),'.mat'])
    EEGdata= cat(3,data(:,:,:,1),data(:,:,:,2),data(:,:,:,3),data(:,:,:,4),...
        data(:,:,:,5),data(:,:,:,6));
    clear data

    % preprocessing
    for i=1:size(EEGdata,3)
        X= EEGdata(:,position,i)'; % EEG signal
        % apply designed band-pass filter[8-90Hz]
        X= filtfilt(b,a,X(:,chn));
        SSVEPdata(:,:,i)= X';
    end
    clear EEGdata
    for i=1:numel(fstim)
        indx= find(y_true==i);
        SSVEPdata_sep(:,:,:,i)= SSVEPdata(:,:,indx);
    end
    clear SSVEPdata

    % SSVEP frequency recognition
    % define Multiway parameters
    max_iter=200;   % the maximal number of iteration for running L1MCCA
    n_comp=1;       % number of projection components for learning the reference signals

    %****************** leave one out validation method ******************%
    % in each iteration one trial is used for testing and the rest (5 trials)
    % are used for training
    n_run=6; % number of runs of EEG data recording
    y_pred=[];
    for run=1:n_run
        idx_traindata=1:n_run;
        idx_traindata(run)=[];

        iniw3=ones(length(idx_traindata),1);
        % run MCCA to learn projections
        for j=1:numel(fstim)
            [w1(:,j),w3(:,j),v1(:,j)]=...
                myMCCA(Xref(:,:,j),SSVEPdata_sep(:,:,idx_traindata,j),max_iter,iniw3,n_comp);
            % Reference signals optimization 
            op_refer=ttm(tensor(SSVEPdata_sep(:,:,idx_traindata,j)),w3(:,j)',3);
            op_refer=tenmat(op_refer,1);
            op_refer=op_refer.data;
            op_refer=w1(:,j)'*op_refer;
            Ref(:,j)=op_refer;
        end
        % frequecny Recognition
        for j= 1:numel(fstim)
            Xtest=SSVEPdata_sep(:,:,run,j);
            % calculate cannonical correlation between the EEG signal(X) and each of the reference signals(Xref)
            for k= 1:numel(fstim)
                Zk=Ref(:,k);
                [wx,wy,r(:,k)]=myCCA(Xtest,Zk');
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


