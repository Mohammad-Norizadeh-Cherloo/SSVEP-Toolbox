clc;
clear
close all;
%% Two-Stage Correlated Component Analysis (TSCORRCA) in SSVEP frequency detection (demo code)
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
%% SSVEP frequency detection using TSCORRCA
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
    %% define model parameters
    n_run=6;
    y_pred=[];
    for run=1:n_run
        idx_traindata=1:n_run;
        idx_traindata(run)=[];
        % Individual Template extraction
        for i=1:numel(fstim)
            data_i=SSVEPdata_sep(:,:,idx_traindata,i);
            Zi(:,:,i)= mean(data_i,3);
        end
        % spatial filters calculation
        for i= 1:numel(fstim)
            x1=SSVEPdata_sep(:,:,idx_traindata(1),i);
            x2=SSVEPdata_sep(:,:,idx_traindata(2),i);
            x3=SSVEPdata_sep(:,:,idx_traindata(3),i);
            x4=SSVEPdata_sep(:,:,idx_traindata(4),i);
            x5=SSVEPdata_sep(:,:,idx_traindata(5),i);
            % constrcut two types of trial aggregated data matrices
            % according to equation (47)
            X1_i= [(x1+x2)/2,(x1+x3)/2,(x1+x4)/2,(x1+x5)/2,(x2+x3)/2];
            X2_i= [(x2+x4)/2,(x2+x5)/2,(x3+x4)/2,(x3+x5)/2,(x4+x5)/2];
            % calculate spatial filters
            w(:,i)= myCORCA(X1_i',X2_i');
        end
        % Recognition
        for j= 1:numel(fstim)
            Xtest=SSVEPdata_sep(:,:,run,j)';
            for k= 1:numel(fstim)
                 % calculate correlation between test EEG signal, and the reference signal
                [~,b0] = myCORCA(Xtest,Zi(:,:,k)');

                % calculate covariance matrices
                XY= [Xtest,Zi(:,:,k)'];
                Cv= cov(XY);
                p= size(Xtest,2);
                R11= Cv(1:p,1:p);
                R22= Cv(p+1:end,p+1:end);
                R12= Cv(1:p,p+1:end);
                R21= Cv(p+1:end,1:p);
                % calculate  the correlation coefficients between test EEG 
                % signal X and reference signal according to equation (48)
                for i=1:numel(fstim)
                    wi=w(:,i);
                    num= wi'*R12*wi;
                    denum=sqrt(wi'*R11*wi) *sqrt(wi'*R22*wi);
                    b_i(i,:)= num/(denum+eps);
                end
                % construct correlation vector according to equation (49)
                bk=[(b0);b_i];
                % combine correlation coefficients according to equation (50)
                r(k)= sum ( sign(bk).* (bk.^2));
                b_i=[];
            end
            Rho=(r);
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


