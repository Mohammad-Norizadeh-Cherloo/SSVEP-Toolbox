clc;
clear
close all;
%%  Combination of CCA and ITCCA (combination method) in SSVEP frequency detection (demo code)
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
Xref_sincos = mySinCosReference(fstim,duration,Nh,Fs);
%% SSVEP frequency detection
for sbj=1:35
    load(['dataset/S',num2str(sbj),'.mat/','S',num2str(sbj),'.mat'])
    EEGdata= cat(3,data(:,:,:,1),data(:,:,:,2),data(:,:,:,3),data(:,:,:,4),...
        data(:,:,:,5),data(:,:,:,6));
    clear data
    % frequency recognition
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

        % frequecny Recognition
        for j= 1:numel(fstim)
            Xtest=SSVEPdata_sep(:,:,run,j);
            for k= 1:numel(fstim)
                % extract spatial filters of EEG test signal and pre-constructed
                % sine-cosine reference signal
                % rho1 is the correlationcoefficients between them
                [W1,~,rho1] = myCCA(Xtest,Xref_sincos(:,:,k));
                % extract spatial filters of EEG test signal and Individual Template
                [W2,~,~] = myCCA(Xtest,ind_temp(:,:,k));
                % extract spatial filters of Individual Template and pre-constructed sine-cosine reference signal
                [W3,~,~] = myCCA(ind_temp(:,:,k),Xref_sincos(:,:,k));

                % apply spatial filters are applied on the new test EEG signal X and individual template signal
                tp2_1= W2(:,1)'*Xtest;
                tp2_2= W2(:,1)'*ind_temp(:,:,k);
                % calculate ordinary correlation between two vectors
                rho2= corr(tp2_1',tp2_2');

                % apply spatial filters are applied on the new test EEG signal X and individual template signal
                tp3_1= W1(:,1)'*Xtest;
                tp3_2= W1(:,1)'*ind_temp(:,:,k);
                % calculate ordinary correlation between two vectors
                rho3= corr(tp3_1',tp3_2');

                % apply spatial filters are applied on the new test EEG signal X and individual template signal
                tp4_1= W3(:,1)'*Xtest;
                tp4_2= W3(:,1)'*ind_temp(:,:,k);
                % calculate ordinary correlation between two vectors
                rho4= corr(tp4_1',tp4_2');

                % concatenate all correlation coefficients
                r(:,k)= [max(rho1),rho2,rho3,rho4];
            end
            % combine correlation coefficients according to equation (19)
            Rho=sum(sign(r) .* (r.^2));
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


