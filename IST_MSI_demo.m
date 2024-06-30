clc;
clear
close all;
%%   Individual Signal mixing Template MSI (IST-MSI) in SSVEP frequency detection (demo code)
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
% define models parameters
SA=0.3;
SB=0.7;
sigma1=8;
n1=1:Nh;
w= exp(-n1/ (2*(sigma1^2)) );
%% Construct sine-cosine reference signal for each stimulus according to equation 2
Xref = mySinCosReference(fstim,duration,Nh,Fs);
%% SSVEP frequency detection using IST-MSI
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
    %% SSVEP recognition
    n_run=6;
    y_pred=[];
    for run=1:n_run
        idx_traindata=1:n_run;
        idx_traindata(run)=[];
        % training
        for i=1:numel(fstim)
            % Individual Template extraction
            data_i=SSVEPdata_sep(:,:,idx_traindata,i);
            ind_temp(:,:,i)= mean(data_i,3);%Individual_Template
            % reference optimization
            for n=1:Nh
                Wx = myCCA( ind_temp(:,:,i),Xref(n*2-1:n*2,:,i));
                Wx=Wx(:,1);
                ref=Wx'* ind_temp(:,:,i);
                % concatenating sine-cosine reference signal
                % and optimized Individual Template
                temp(:,:,n)= [ref;Xref(n*2-1:n*2,:,i)];
            end
            Yn(:,:,:,i)=temp;
            temp=[];
        end
        % Recognition
        for j= 1:numel(fstim)
            Xtest=SSVEPdata_sep(:,:,run,j)';
            for k= 1:numel(fstim)
                Sk0 = myMSI(Xtest',Xref(:,:,k));

                Sk(1) = myMSI(Xtest',Yn(:,:,1,k));
                Sk(2) = myMSI(Xtest',Yn(:,:,2,k));
                Sk(3) = myMSI(Xtest',Yn(:,:,3,k));
                Sk(4) = myMSI(Xtest',Yn(:,:,4,k));
                Sk(5) = myMSI(Xtest',Yn(:,:,5,k));
                % combine synchronization indexs according to equation (42)
                S(:,k)=(SA*Sk0 )+ ( SB*(sum(Sk.*w)) );
            end
            Rho=(S);
            % determine the the stimulus frequency of EEG signal(X)
            [~,y_p(j)]=max(Rho);
            S=[];
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


