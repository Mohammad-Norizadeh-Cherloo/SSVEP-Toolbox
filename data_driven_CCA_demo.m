clc;
clear
close all;

%%  Data-Driven CCA (DDCCA) in SSVEP frequency detection (demo code)
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

%% SSVEP frequency detection
for sbj=1:35
    load(['dataset/S',num2str(sbj),'.mat/','S',num2str(sbj),'.mat'])
    EEGdata= cat(3,data(:,:,:,1),data(:,:,:,2),data(:,:,:,3),data(:,:,:,4),...
        data(:,:,:,5),data(:,:,:,6));
    clear data
    % data preprocessing
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
    clear SSVEP

    %% recognition
    n_run=6;
    y_pred=[];
    for run=1:n_run
        idx_traindata=1:n_run;
        idx_traindata(run)=[];
        % Individual Template extraction
        for i=1:numel(fstim)
            data_i=SSVEPdata_sep(:,:,idx_traindata,i);
            ind_temp(:,:,i)= mean(data_i,3);%Individual_Template
        end

        % Recognition
        for j= 1:numel(fstim)
            Xtrain= SSVEPdata_sep(:,:,idx_traindata,j);
            Xtrain= reshape(Xtrain,[size(Xtrain,1) size(Xtrain,2)*size(Xtrain,3)]   );
            Xtest=SSVEPdata_sep(:,:,run,j)';
            for k= 1:numel(fstim)
                % calculate spatial filters according to equation 21 
                S=ind_temp(:,:,k);
                ref=repmat(S,1,numel(idx_traindata));
                Wx=myCCA(Xtrain,ref);
                Wx= Wx(:,1:1);
                % calculate ordinary correlation according to equaltion 22
                cr = corrcoef(Xtest*Wx,S'*Wx);
                r(:,k) = cr(1,2);
            end
            Rho=(r);
            % determine the the stimulus frequency of EEG signal(X)
            [~,y_p(j)]=max(Rho);
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


