clc;
clear
close all;
%% Filter Bank Ensemble Task Related Component Analysis (FBETRCA) in SSVEP frequency detection (demo code)
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
load('dataset\Freq_Phase.mat')
fstim= freqs;
% build label for each stimulus which will be used for evaluatoin
freqs= repmat(freqs,1,6);
y_true= repmat(1:40,1,6);

fs_low=min(fstim);
% define frequency bands and their weights
bands=[fs_low*[1:5]; 80*ones(1,5)];

sigma=1;
Nsb=[1:size(bands,2)];
w= exp(-Nsb/ (2*(sigma^2)) );
%%  ssvep frequency recognition
for sbj=1:35
    load(['dataset/S',num2str(sbj),'.mat/','S',num2str(sbj),'.mat'])
    EEGdata= cat(3,data(:,:,:,1),data(:,:,:,2),data(:,:,:,3),data(:,:,:,4),...
        data(:,:,:,5),data(:,:,:,6));
    clear data
    % filter bank
    for i=1:size(EEGdata,3)
        X= EEGdata(:,position,i)';
        % filter data using frequency band[1]
        [b,a]= butter(3,bands(:,1)/(Fs/2), 'bandpass');
        SSVEP_sb1(:,:,i)= filtfilt(b,a,X(:,chn))';
        % filter data using frequency band[2]
        [b,a]= butter(3,bands(:,2)/(Fs/2), 'bandpass');
        SSVEP_sb2(:,:,i)= filtfilt(b,a,X(:,chn))';
        % filter data using frequency band[3]
        [b,a]= butter(3,bands(:,3)/(Fs/2), 'bandpass');
        SSVEP_sb3(:,:,i)= filtfilt(b,a,X(:,chn))';
        % filter data using frequency band[4]
        [b,a]= butter(3,bands(:,4)/(Fs/2), 'bandpass');
        SSVEP_sb4(:,:,i)= filtfilt(b,a,X(:,chn))';
        % filter data using frequency band[5]
        [b,a]= butter(3,bands(:,5)/(Fs/2), 'bandpass');
        SSVEP_sb5(:,:,i)= filtfilt(b,a,X(:,chn))';
    end
    clear EEGdata
    for i=1:numel(fstim)
        indx= find(y_true==i);
        SSVEPdata_sep_sb1(:,:,:,i)= SSVEP_sb1(:,:,indx);
        SSVEPdata_sep_sb2(:,:,:,i)= SSVEP_sb2(:,:,indx);
        SSVEPdata_sep_sb3(:,:,:,i)= SSVEP_sb3(:,:,indx);
        SSVEPdata_sep_sb4(:,:,:,i)= SSVEP_sb4(:,:,indx);
        SSVEPdata_sep_sb5(:,:,:,i)= SSVEP_sb5(:,:,indx);
    end
    clear SSVEP_sb1 SSVEP_sb2 SSVEP_sb3 SSVEP_sb4 SSVEP_sb5
    %%  define model parameters 
    m=1;    % number of extracted components for each spatial filter
    n_run=6;
    y_pred=[];
    for run=1:n_run
        idx_traindata=1:n_run;
        idx_traindata(run)=[];
        % Individual Templates ad spatial filters calculation
        for i=1:numel(fstim)
            data_sb1=SSVEPdata_sep_sb1(:,:,idx_traindata,i);
            data_sb2=SSVEPdata_sep_sb2(:,:,idx_traindata,i);
            data_sb3=SSVEPdata_sep_sb3(:,:,idx_traindata,i);
            data_sb4=SSVEPdata_sep_sb4(:,:,idx_traindata,i);
            data_sb5=SSVEPdata_sep_sb5(:,:,idx_traindata,i);

            % calculate spatial filters
            W_sb1(:,:,i) = myTRCA(data_sb1,m);
            W_sb2(:,:,i) = myTRCA(data_sb2,m);
            W_sb3(:,:,i) = myTRCA(data_sb3,m);
            W_sb4(:,:,i) = myTRCA(data_sb4,m);
            W_sb5(:,:,i) = myTRCA(data_sb5,m);

            % calculate Individual Templates 
            ind_temp_sb1(:,:,i)= mean(data_sb1,3);
            ind_temp_sb2(:,:,i)= mean(data_sb2,3);
            ind_temp_sb3(:,:,i)= mean(data_sb3,3);
            ind_temp_sb4(:,:,i)= mean(data_sb4,3);
            ind_temp_sb5(:,:,i)= mean(data_sb5,3);
        end
        
        % Recognition
        for j= 1:numel(fstim)
            Xtest_sb1=SSVEPdata_sep_sb1(:,:,run,j)';
            Xtest_sb2=SSVEPdata_sep_sb2(:,:,run,j)';
            Xtest_sb3=SSVEPdata_sep_sb3(:,:,run,j)';
            Xtest_sb4=SSVEPdata_sep_sb4(:,:,run,j)';
            Xtest_sb5=SSVEPdata_sep_sb5(:,:,run,j)';
            
            for k= 1:numel(fstim)
                traindata_sb1=ind_temp_sb1(:,:,k)';
                traindata_sb2=ind_temp_sb2(:,:,k)';
                traindata_sb3=ind_temp_sb3(:,:,k)';
                traindata_sb4=ind_temp_sb4(:,:,k)';
                traindata_sb5=ind_temp_sb5(:,:,k)';
                
                w_sb1= squeeze(W_sb1);
                w_sb2= squeeze(W_sb2);
                w_sb3= squeeze(W_sb3);
                w_sb4= squeeze(W_sb4);
                w_sb5= squeeze(W_sb5);

                % apply filters and calculate ordinary correlation 
                cr1 = corrcoef(Xtest_sb1*w_sb1, traindata_sb1*w_sb1);
                cr2 = corrcoef(Xtest_sb2*w_sb2, traindata_sb2*w_sb2);
                cr3 = corrcoef(Xtest_sb3*w_sb3, traindata_sb3*w_sb3);
                cr4 = corrcoef(Xtest_sb4*w_sb4, traindata_sb4*w_sb4);
                cr5 = corrcoef(Xtest_sb5*w_sb5, traindata_sb5*w_sb5);
                % combine correlation coefficients according to equation (6)
                r(:,k)= sum(w.*[cr1(1,2),cr2(1,2),cr3(1,2),cr4(1,2),cr5(1,2)]);
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


