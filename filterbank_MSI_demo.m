clc;
clear
close all;
%% filter bank MSI (FBMSI) in SSVEP frequency detection (demo code)
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

% define frequency bands and their weights
freq_bands= [(1:9)*8; ones(1,9)*90];
freq_bands(1,1:end)=freq_bands(1,1:end)-2;

sigma=1;
% sigma=0.2;
Nsb=[1:size(freq_bands,2)]';
w= exp(-Nsb/ (2*(sigma^2)) );
%% Construct sine-cosine reference signal for each stimulus according to equation 2
Xref = mySinCosReference(fstim,duration,Nh,Fs);
%% SSVEP frequency detection using filter bank MSI (FBMSI)
for sbj=1:35
    load(['dataset/S',num2str(sbj),'.mat/','S',num2str(sbj),'.mat'])
    EEGdata= cat(3,data(:,:,:,1),data(:,:,:,2),data(:,:,:,3),data(:,:,:,4),...
        data(:,:,:,5),data(:,:,:,6));
    clear data
    % frequency recognition
    y_pred= zeros(1,size(EEGdata,3));
    for i=1:size(EEGdata,3)
        X= EEGdata(:,position,i)';
        % apply designed band-pass filter[8-90Hz]
        X= filtfilt(b,a,X);
        % calculate cannonical correlation between the EEG sub-bands and each of the reference signals(Xref)
        for sb=1:size(freq_bands,2)
            % filter data using one of the frequency bands
            [b2,a2]= butter(3,[freq_bands(:,sb)]/(Fs/2), 'bandpass');
            X_sb= filtfilt(b2,a2,X(:,chn));
            for k= 1:size(Xref,3)
               temp(:,k) = myMSI(X_sb',Xref(:,:,k));
            end
            Rho_sb(sb,:)= (temp);
            temp=[];
        end
        Rho_sb= rescale(Rho_sb,0,1);
        % combine correlation coefficients according to equation(6)
        W= repmat(w,1,size(Rho_sb,2));
        Rho= sum(W.* (Rho_sb.^2));
        % determine the the stimulus frequency of EEG signal(X)
        [~,y_p]= max(Rho);
        y_pred(i)= y_p;
        Rho_sb=[];
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



