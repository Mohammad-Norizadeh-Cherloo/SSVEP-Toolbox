clc;
clear
close all;
%% www.onlinebme.com
% filter bank Spatio-spectral CCA (FBSS-CCA) in SSVEP frequency detection (demo code)
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
tau= 2;%time delay
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

% define frequency bands and their weights
fs_low=min(fstim);
freq_bands=[fs_low*[1:5]; 80*ones(1,5)];

sigma2=0.6;
nc2=[1:size(freq_bands,2)]';
w= exp(-nc2/ (2*(sigma2^2)) );

% define the weight of the correlation coefficients,
sigma1=1;
nc1=[1:2*numel(chn)]';
phi= exp(-nc1/ (2*(sigma1^2)) );
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
        [b,a]= butter(3,freq_bands(:,1)/(Fs/2), 'bandpass');
        SSVEP_sb1(:,:,i)= filtfilt(b,a,X(:,chn))';
        % filter data using frequency band[2]
        [b,a]= butter(3,freq_bands(:,2)/(Fs/2), 'bandpass');
        SSVEP_sb2(:,:,i)= filtfilt(b,a,X(:,chn))';
        % filter data using frequency band[3]
        [b,a]= butter(3,freq_bands(:,3)/(Fs/2), 'bandpass');
        SSVEP_sb3(:,:,i)= filtfilt(b,a,X(:,chn))';
        % filter data using frequency band[4]
        [b,a]= butter(3,freq_bands(:,4)/(Fs/2), 'bandpass');
        SSVEP_sb4(:,:,i)= filtfilt(b,a,X(:,chn))';
        % filter data using frequency band[5]
        [b,a]= butter(3,freq_bands(:,5)/(Fs/2), 'bandpass');
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
    %%  SSVEP recognition
    n_run=6;
    y_pred=[];

    for run=1:n_run
        idx_traindata=1:n_run;
        idx_traindata(run)=[];
        % Individual Template extraction
        % two reference signals are calculated by averaging over the first
        % and last half of EEG training trials of k-th stimulus frequency
        for i=1:numel(fstim)
            data_sb1=SSVEPdata_sep_sb1(:,:,idx_traindata(1:3),i);
            Ref1(:,:,i,1)=mean(data_sb1,3);

            data_sb2=SSVEPdata_sep_sb2(:,:,idx_traindata(1:3),i);
            Ref1(:,:,i,2)=mean(data_sb2,3);

            data_sb3=SSVEPdata_sep_sb3(:,:,idx_traindata(1:3),i);
            Ref1(:,:,i,3)=mean(data_sb3,3);

            data_sb4=SSVEPdata_sep_sb4(:,:,idx_traindata(1:3),i);
            Ref1(:,:,i,4)=mean(data_sb4,3);

            data_sb5=SSVEPdata_sep_sb5(:,:,idx_traindata(1:3),i);
            Ref1(:,:,i,5)=mean(data_sb5,3);
            % **************************************
            data_sb1=SSVEPdata_sep_sb1(:,:,idx_traindata(4:5),i);
            Ref2(:,:,i,1)=mean(data_sb1,3);

            data_sb2=SSVEPdata_sep_sb2(:,:,idx_traindata(4:5),i);
            Ref2(:,:,i,2)=mean(data_sb2,3);

            data_sb3=SSVEPdata_sep_sb3(:,:,idx_traindata(4:5),i);
            Ref2(:,:,i,3)=mean(data_sb3,3);

            data_sb4=SSVEPdata_sep_sb4(:,:,idx_traindata(4:5),i);
            Ref2(:,:,i,4)=mean(data_sb4,3);

            data_sb5=SSVEPdata_sep_sb5(:,:,idx_traindata(4:5),i);
            Ref2(:,:,i,5)=mean(data_sb5,3);

        end
        % Recognition
        for j= 1:numel(fstim)
            Xtest=SSVEPdata_sep_sb1(:,:,run,j)';

            for k= 1:numel(fstim)
                % calcualte correlation bewtween EEG signal and reference
                % signals  (first half)
                [~,r_sb1_1] = myCORCA(Xtest,Ref1(:,:,k,1)');
                [~,r_sb2_1] = myCORCA(Xtest,Ref1(:,:,k,2)');
                [~,r_sb3_1] = myCORCA(Xtest,Ref1(:,:,k,3)');
                [~,r_sb4_1] = myCORCA(Xtest,Ref1(:,:,k,4)');
                [~,r_sb5_1] = myCORCA(Xtest,Ref1(:,:,k,5)');
                %********************************************
                % calcualte correlation bewtween EEG signal and reference
                % signals  (second half)
                [~,r_sb1_2] = myCORCA(Xtest,Ref2(:,:,k,1)');
                [~,r_sb2_2] = myCORCA(Xtest,Ref2(:,:,k,2)');
                [~,r_sb3_2] = myCORCA(Xtest,Ref2(:,:,k,3)');
                [~,r_sb4_2] = myCORCA(Xtest,Ref2(:,:,k,4)');
                [~,r_sb5_2] = myCORCA(Xtest,Ref2(:,:,k,5)');
                
                % concatenate and sort correlation coefficients
                rho1= sort([r_sb1_1;r_sb1_2],'descend');
                rho2= sort([r_sb2_1;r_sb2_2],'descend');
                rho3= sort([r_sb3_1;r_sb3_2],'descend');
                rho4= sort([r_sb4_1;r_sb4_2],'descend');
                rho5= sort([r_sb5_1;r_sb5_2],'descend');
                % combine correlation coefficients according to equation (52)
                rho_sb1(k)= sum((phi.*(rho1)));
                rho_sb2(k)= sum((phi.*(rho2)));
                rho_sb3(k)= sum((phi.*(rho3)));
                rho_sb4(k)= sum((phi.*(rho4)));
                rho_sb5(k)= sum((phi.*(rho5)));
            end
            % combine correlation coefficients of all subbands according to equation (53)
            rho_t=[rho_sb1;rho_sb2;rho_sb3;rho_sb4;rho_sb5];

            Rho=sum(repmat(w,1,size(rho_t,2)).*(rho_t));
            % determine the the stimulus frequency of EEG signal(X)
            [~,y_p(j)]=max(Rho);
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


