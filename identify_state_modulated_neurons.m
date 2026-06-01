% state modified neuron identification for single mice, data was saved
% -append to sxx_postprocess_0.5s.mat
%% load files
tic


period=10;
bin=0.5;% s
minlength=3;% minlength time = (minlength+1)*epochtime
skip_time=0.4*3600;
stop_time=3.375*3600;


s98_NW_z_mean_all=[];
s98_NR_z_mean_all=[];
s98_RW_z_mean_all=[];
s98_WN_z_mean_all=[];


% 1 = REM sleep, 2 = wakefulness, 3 = NREM sleep, 4 = undefined

idx_NW=[];
idx_RW=[];
idx_WN=[];
idx_NR=[];

idx_NW_time=[];
idx_RW_time=[];
idx_WN_time=[];
idx_NR_time=[];


labels_temp=labels; % attention锛? use labels锛? not labels_i
labels_temp(labels_temp==3)=5;
labels_temp(labels_temp==4)=10;
labels_diff=diff(labels_temp);
idx_NW=find(labels_diff==-3);
idx_RW=find(labels_diff==1);
idx_WN=find(labels_diff==3);
idx_NR=find(labels_diff==-4);

idx_NW=idx_NW(idx_NW>skip_time/info.epochtime+minlength & idx_NW<stop_time/info.epochtime-minlength);
idx_RW=idx_RW(idx_RW>skip_time/info.epochtime+minlength & idx_RW<stop_time/info.epochtime-minlength);
idx_WN=idx_WN(idx_WN>skip_time/info.epochtime+minlength & idx_WN<stop_time/info.epochtime-minlength);
idx_NR=idx_NR(idx_NR>skip_time/info.epochtime+minlength & idx_NR<stop_time/info.epochtime-minlength);

idx_NW_time=idx_NW*info.epochtime-start_time;
idx_RW_time=idx_RW*info.epochtime-start_time;
idx_WN_time=idx_WN*info.epochtime-start_time;
idx_NR_time=idx_NR*info.epochtime-start_time;


% exclude transition length less than minlength(L&R)
outNR=[];
for i=1:length(idx_NR_time)
    if sum(labels_diff(idx_NR(i)-minlength:idx_NR(i)+minlength))~= -4
        outNR(length(outNR)+1)=i;
    end
end
idx_NR_time(outNR)=[];
outNW=[];
for i=1:length(idx_NW)
    if sum(labels_diff(idx_NW(i)-minlength:idx_NW(i)+minlength))~= -3
        outNW(length(outNW)+1)=i;
    end
end
idx_NW_time(outNW)=[];
outRW=[];
for i=1:length(idx_RW_time)
    if sum(labels_diff(idx_RW(i)-minlength:idx_RW(i)+minlength))~= 1
        outRW(length(outRW)+1)=i;
    end
end
idx_RW_time(outRW)=[];
outWN=[];
for i=1:length(idx_WN_time)
    if sum(labels_diff(idx_WN(i)-minlength:idx_WN(i)+minlength))~= 3
        outWN(length(outWN)+1)=i;
    end
end
idx_WN_time(outWN)=[];

% spike freq
tempRWds=[];
tempNWds=[];
tempWNds=[];
tempNRds=[];

tempNWds_z=[];
tempNRds_z=[];
tempRWds_z=[];
tempWNds_z=[];


% WAKE TO NREM
for i=1:length(idx_WN_time)
    tempWNds(:,:,i)=s98_spike_freq_good(:,round(idx_WN_time(i)/bin)-period/bin+1:round(idx_WN_time(i)/bin)+period/bin);
    tempWNds_z(:,:,i)=zscore(tempWNds(:,:,i),0,2);
end
s98_WN_z_mean_all=nanmean(tempWNds_z,3);

temp_mean1=mean(tempWNds_z(:,6:15,:),2);
temp_mean2=mean(tempWNds_z(:,26:35,:),2);
temp_T_m=nanmean(temp_mean1,3)-nanmean(temp_mean2,3);
for i = 1:size(tempWNds_z,1)
    [WN_pvalue(i),r(i)] = ranksum(squeeze(temp_mean1(i,:,:)),squeeze(temp_mean2(i,:,:)));
end
WN_FDR = mafdr(WN_pvalue,'BHFDR','true');
temp_T_m_WN_modified=temp_T_m(find(WN_FDR<=0.001));
s98_idx_WN_modified(:,1)=find(WN_FDR<=0.001)';%good idx中NW modified neurons
s98_idx_WN_modified(find(temp_T_m_WN_modified>0),2)=1; %前一个state高的赋1，后一个高的赋0


% NREM TO WAKE
for i=1:length(idx_NW_time)
    tempNWds(:,:,i)=s98_spike_freq_good(:,round(idx_NW_time(i)/bin)-period/bin+1:round(idx_NW_time(i)/bin)+period/bin);
    tempNWds_z(:,:,i)=zscore(tempNWds(:,:,i),0,2);
end
s98_NW_z_mean_all=nanmean(tempNWds_z,3);

temp_mean1=mean(tempNWds_z(:,6:15,:),2);
temp_mean2=mean(tempNWds_z(:,26:35,:),2);
temp_T_m=nanmean(temp_mean1,3)-nanmean(temp_mean2,3);
for i = 1:size(tempNWds_z,1)
    [NW_pvalue(i),r(i)] = ranksum(squeeze(temp_mean1(i,:,:)),squeeze(temp_mean2(i,:,:)));
end
NW_FDR = mafdr(NW_pvalue,'BHFDR','true');
temp_T_m_NW_modified=temp_T_m(find(NW_FDR<=0.001));
s98_idx_NW_modified(:,1)=find(NW_FDR<=0.001)';%good idx中NW modified neurons
s98_idx_NW_modified(find(temp_T_m_NW_modified>0),2)=1; %前一个state高的赋1，后一个高的赋0

% NREM TO REM
if mean(idx_NR)>0
    for i=1:length(idx_NR_time)
        tempNRds(:,:,i)=s98_spike_freq_good(:,round(idx_NR_time(i)/bin)-period/bin+1:round(idx_NR_time(i)/bin)+period/bin);
        tempNRds_z(:,:,i)=zscore(tempNRds(:,:,i),0,2);
    end
    s98_NR_z_mean_all=nanmean(tempNRds_z,3);
    
    temp_mean1=mean(tempNRds_z(:,6:15,:),2);
    temp_mean2=mean(tempNRds_z(:,26:35,:),2);
    temp_T_m=nanmean(temp_mean1,3)-nanmean(temp_mean2,3);
    for i = 1:size(tempNRds_z,1)
        [NR_pvalue(i),r(i)] = ranksum(squeeze(temp_mean1(i,:,:)),squeeze(temp_mean2(i,:,:)));
    end
    NR_FDR = mafdr(NR_pvalue,'BHFDR','true');
    temp_T_m_NR_modified=temp_T_m(find(NR_FDR<=0.001));
    s98_idx_NR_modified(:,1)=find(NR_FDR<=0.001)';%good idx中NW modified neurons
    s98_idx_NR_modified(find(temp_T_m_NR_modified>0),2)=1; %前一个state高的赋1，后一个高的赋0
    
    % REM TO WAKE
    for i=1:length(idx_RW_time)
        tempRWds(:,:,i)=s98_spike_freq_good(:,round(idx_RW_time(i)/bin)-period/bin+1:round(idx_RW_time(i)/bin)+period/bin);
        tempRWds_z(:,:,i)=zscore(tempRWds(:,:,i),0,2);
    end
    s98_RW_z_mean_all=nanmean(tempRWds_z,3);
    
    temp_mean1=mean(tempRWds_z(:,6:15,:),2);
    temp_mean2=mean(tempRWds_z(:,26:35,:),2);
    temp_T_m=nanmean(temp_mean1,3)-nanmean(temp_mean2,3);
    for i = 1:size(tempRWds_z,1)
        [RW_pvalue(i),r(i)] = ranksum(squeeze(temp_mean1(i,:,:)),squeeze(temp_mean2(i,:,:)));
    end
    RW_FDR = mafdr(RW_pvalue,'BHFDR','true');
    temp_T_m_RW_modified=temp_T_m(find(RW_FDR<=0.001));
    s98_idx_RW_modified(:,1)=find(RW_FDR<=0.001)';%good idx中NW modified neurons
    s98_idx_RW_modified(find(temp_T_m_RW_modified>0),2)=1; %前一个state高的赋1，后一个高的赋0
end

toc




