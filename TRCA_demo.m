clc;
clear
close all;
%%  Task Related Component Analysis (TRCA) in SSVEP frequency detection (demo code)
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

% load frequency-phase information of stimuli
load('dataset/Freq_Phase.mat')
fstim= freqs;
% build label for each stimulus which will be used for evaluatoin
freqs= repmat(freqs,1,6);
y_true= repmat(1:40,1,6);

% design a band-pass butterworth filter
[b,a]= butter(3,[8 90]/(Fs/2), 'bandpass');

%%  ssvep frequency recognition
for sbj=1:35
    load(['dataset/S',num2str(sbj),'.mat/','S',num2str(sbj),'.mat'])
    EEGdata= cat(3,data(:,:,:,1),data(:,:,:,2),data(:,:,:,3),data(:,:,:,4),...
        data(:,:,:,5),data(:,:,:,6));
    clear data
    % apply filter
    for i=1:size(EEGdata,3)
        X= EEGdata(:,position,i)';
        SSVEP(:,:,i)= filtfilt(b,a,X(:,chn))';
    end
    clear EEGdata
    for i=1:numel(fstim)
        indx= find(y_true==i);
        SSVEPdata_sep(:,:,:,i)= SSVEP(:,:,indx);
    end
    clear SSVEP SSVEP_sb2 SSVEP_sb3 SSVEP_sb4 SSVEP_sb5
    %%  define model parameters
    m=1;    % number of extracted components for each spatial filter
    n_run=6;
    y_pred=[];
    for run=1:n_run
        idx_traindata=1:n_run;
        idx_traindata(run)=[];
        % Individual Templates ad spatial filters calculation
        for i=1:numel(fstim)
            data_sb1=SSVEPdata_sep(:,:,idx_traindata,i);

            % calculate spatial filters
            W_sb1(:,:,i) = myTRCA(data_sb1,m);

            % calculate Individual Templates
            ind_temp_sb1(:,:,i)= mean(data_sb1,3);
        end

        % Recognition
        for j= 1:numel(fstim)
            Xtest_sb1=SSVEPdata_sep(:,:,run,j)';
            for k= 1:numel(fstim)
                traindata_sb1=ind_temp_sb1(:,:,k)';
                w_sb1= W_sb1(:,:,k);
                % apply filters and calculate ordinary correlation
                cr1 = corrcoef(Xtest_sb1*w_sb1, traindata_sb1*w_sb1);
                % combine correlation coefficients according to equation (6)
                r(:,k)= ([cr1(1,2)]);
            end
            Rho=(r);
            % determine the the stimulus frequency of EEG signal(X)
            [v,y_p(j)]=max(Rho);
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


