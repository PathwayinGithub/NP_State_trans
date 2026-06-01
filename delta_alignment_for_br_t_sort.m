%trial alignment  by delta power across sessions
%% partI append to sxx_SVM2.mat(WN) or sxx_SVM_wake2.mat(NW)
addpath(genpath('X:\code'));

for m = 1:40
    % subj={'s16','s22','s26','s27','s28','s29','s30','s33','s34','s35','s36','s38','s42',...
    %     's43','s48','s49','s51','s53','s54','s59','s60','s61','s62','s63','s64',...
    %     's66','s67','s68','s70','s72','s74','s75','s76','s77','s78','s80','s81',...
    %     's85','s86','s87','s88','s99'};%WN   
    subj={'s16','s22','s26','s27','s28','s29','s30','s33','s34','s35','s36','s42',...
    's43','s48','s51','s53','s54','s59','s60','s61','s62','s63','s64',...
    's66','s67','s68','s70','s72','s74','s75','s76','s77','s78','s80','s81',...
    's85','s86','s87','s88','s99'};%NW
    tmp_subj=subj{m};
    filepath='Y:\post_process_output\';
    load(strcat(filepath,'start_stop_SVM.mat'));
    load(strcat(filepath,'update_brain_region_for_fig1.mat'),strcat(tmp_subj,'_cluster_channel_br'));
    load(strcat(filepath,'update_brain_region_for_fig1.mat'),strcat(tmp_subj,'_cluster_channel_name'));
    eval(strcat('tmp_cluster_channel_br=',tmp_subj,'_cluster_channel_br;'));
    eval(strcat('tmp_cluster_channel_name=',tmp_subj,'_cluster_channel_name;'));
    
    matfile=strcat(filepath,strcat(tmp_subj,'_postprocess_0.5s.mat'));
    load(matfile, 'start_time'); 
    load(matfile, 'labels');
    load(matfile, 'EEG');
    load(matfile, 'EMG');
    load(matfile, 'info');
    
    
    load(strcat(filepath,'SVM\',tmp_subj,'_SVM_wake.mat'), 'start','stop','temp_spike_freq','fr_t');
    %power
    sr=info.samplerate;
    EEG=EEG(floor(start*sr)+1:floor(stop*sr));
    EMG=EMG(floor(start*sr)+1:floor(stop*sr));
    [~,~,~,eegPSD]=spectrogram(EEG,112,10,0:0.5:30,sr); 
    eeg_delta=sum(eegPSD(2:9,:)); % 0.5-4
    
    
    delta_frt=[];
    parfor i=1:size(temp_spike_freq,2)
        delta_frt(i,1)=eeg_delta(min(ceil((i*fr_t*sr-56)/102)+1,length(eeg_delta)));
    end
    delta_frt_s=movmean(delta_frt,40,1);
    
    load(strcat(filepath,'SVM\',tmp_subj,'_SVM_wake2.mat'), 'trans_time2', 'idx_medium23_good_curve','idx_medium23_notgood', 'table_states_corr','left_win','right_win','B_trans');
    tmp_idx=find(trans_time2<=300 & trans_time2>=100);
    idx_medium23_notgood=idx_medium23_notgood(tmp_idx);
    idx_medium23=[idx_medium23_good_curve;idx_medium23_notgood];

    num_trials=length(idx_medium23(1:end-2));
    delta_trials=zeros(200,num_trials);
    for i = 1:num_trials
        tmp_idx=table_states_corr(idx_medium23(i)+1,4);
        delta_trials(:,i)=delta_frt_s(tmp_idx-100:tmp_idx+100-1);
    end
    
    delta_trial_miu=smoothdata(mean(delta_trials,2),'gaussian',20);
    diff1_delta=diff(delta_trial_miu,1);
    
    %
    a=diff1_delta>0;
    b=diff1_delta<0;
    c=a-b;
    d=diff(c,1);
    trans_idx=find(d==2|d==-2)+1;
    trans_idx=[1;trans_idx;199];
    AUC_diff1=[];
    for i = 1:length(trans_idx)-1
        AUC_diff1(i)=sum(abs(diff1_delta(trans_idx(i):trans_idx(i+1))));
    end
    [~,I]=max(AUC_diff1);
    range_ramp=[trans_idx(I),trans_idx(I+1)];
    delta_ramp=delta_trial_miu(trans_idx(I):trans_idx(I+1));
    delta_max=max(delta_ramp);
    delta_min=min(delta_ramp);
    ramp50=(delta_max+delta_min)/2;
    [~,I2]=min(abs(delta_ramp-ramp50));
    idx50=trans_idx(I)+I2-1;
    
    time_range=-floor((left_win+right_win)/2)+1:1:ceil((left_win+right_win)/2);
    offset_delta_to_midseq=idx50-100;
    time_range_corr=time_range-(idx50-100);
    
    seq_t_corr=time_range_corr(B_trans);
    %save(strcat(filepath,'SVM\',tmp_subj,'_SVM2'),'seq_t_corr','-append');
    save(strcat(filepath,'SVM\',tmp_subj,'_SVM_wake2'),'offset_delta_to_midseq','-append');

    close all
    clear
end
%% statistic
load('br_id_color_fig1.mat');
load('br_name_color_fig1.mat');
% subj={'s16','s22','s26','s27','s28','s29','s30','s33','s34','s35','s36','s38','s42',...
%     's43','s48','s49','s51','s53','s54','s59','s60','s61','s62','s63','s64',...
%     's66','s67','s68','s70','s72','s74','s75','s76','s77','s78','s80','s81',...
%     's85','s86','s87','s88','s99'};%WN 
subj={'s16','s22','s26','s27','s28','s29','s30','s33','s34','s35','s36','s42',...
    's43','s48','s51','s53','s54','s59','s60','s61','s62','s63','s64',...
    's66','s67','s68','s70','s72','s74','s75','s76','s77','s78','s80','s81',...
    's85','s86','s87','s88','s99'};%NW
br_seq_t_corr_all=[];
br_all=[];
for m = 1:length(subj)
    tmp_subj=subj{m};
    filepath='Y:\post_process_output\';
    load(strcat(filepath,'SVM\',tmp_subj,'_SVM_wake2.mat'), 'seq_t_corr', 'trans_tmp_idx', 'I_trans', 'idx_good_presence');
    
    load(strcat(filepath,'update_brain_region_for_fig1.mat'),strcat(tmp_subj,'_cluster_channel_br'));
    eval(strcat('tmp_cluster_channel_br=',tmp_subj,'_cluster_channel_br;'));
    
    tmp_cluster_channel_br=tmp_cluster_channel_br(idx_good_presence,:);
    tmp_cluster_channel_br_seq=tmp_cluster_channel_br(trans_tmp_idx,:);
    tmp_cluster_channel_br_seq=tmp_cluster_channel_br_seq(I_trans,:);
    tmp_cluster_channel_br_seq(:,6)=seq_t_corr;
    tmp_cluster_channel_br_seq(:,7)=str2double(tmp_subj(2:end));
    tmp_cluster_channel_br(:,7)=str2double(tmp_subj(2:end));
    br_seq_t_corr_all=[br_seq_t_corr_all;tmp_cluster_channel_br_seq];
    br_all=[br_all;tmp_cluster_channel_br];
end

statistic_t=zeros(length(br_id_color_fig1),3);%mean median sem
for i = 1:length(br_id_color_fig1(:,1))
    tmp_idx=find(br_seq_t_corr_all(:,5)==br_id_color_fig1(i,1));
    tmp_idx_all=find(br_all(:,5)==br_id_color_fig1(i,1));
    if tmp_idx
        tmp_br_all=br_all(tmp_idx_all,:);
        mice_id=unique(tmp_br_all(:,7));
        tmp_br=br_seq_t_corr_all(tmp_idx,:);
        statistic_t(i,1)=median(tmp_br(:,6));
        statistic_t(i,2)=std(tmp_br(:,6))/sqrt(length(tmp_idx));
        statistic_t(i,3)=length(tmp_idx);
        a=[];
        for j = 1:length(mice_id)
            a(j)=length(find(tmp_br_all(:,7)==mice_id(j)));
        end
        statistic_t(i,4)=sum(a>=10);
    end
end

exclude_idx=find(sum(statistic_t,2)==0 | statistic_t(:,4)<5); 
statistic_t(exclude_idx,:)=[];
br_id_color_fig1(exclude_idx,:)=[];
br_id_name(exclude_idx)=[];

fr_t=0.025;
[~,I]=sort(statistic_t(:,1),'ascend');
statistic_t=statistic_t(I,:);
statistic_t(:,1)=statistic_t(:,1)*fr_t;
br_id_color_fig1=br_id_color_fig1(I,:);
br_id_name=br_id_name(I);


% 

color_ap=[76,88,147;18,85,130;55,126,184;139,184,214;243,143,60;212,36,42;238,144,148;150,145,193;247,191,108;250,233,59]./255;%77,160,222;
st_color_seq=[];
for i= 1:length(br_id_color_fig1) 
    st_color_seq(i,:)=color_ap(br_id_color_fig1(i,2),:);
end
st_color_seq=reshape(st_color_seq,length(br_id_color_fig1),1,3);

patch_x=zeros(4,length(br_id_color_fig1));
patch_x([1 2],:)=repmat([0],2,length(br_id_color_fig1));
patch_x([3 4],:)=repmat([1],2,length(br_id_color_fig1));
patch_y=zeros(4,length(br_id_color_fig1));
patch_y([1 4],:)=repmat(0:length(br_id_color_fig1)-1,2,1);
patch_y([2 3],:)=repmat(1:length(br_id_color_fig1),2,1);


figure('Color','w');
h2=subplot('Position',[0.6 0.1 0.05 0.8]);patch(patch_x,patch_y,st_color_seq,'EdgeColor','none');ylim([0 length(br_id_color_fig1)]);
%set(gca,'YDir','reverse');title('br seq','FontSize',18);
box off


%% statistic of specific brain region (dorpm)  
load('Y:\SGL_DATA\br.mat');
br_seq_t_corr_dorpm = br_seq_t_corr_all(find(br_seq_t_corr_all(:,5)==856),:);
for i = 1:length(br_seq_t_corr_dorpm)
    idx=find(br_level(:,1)==br_seq_t_corr_dorpm(i,3));
    if idx
        br_seq_t_corr_dorpm(i,8)=br_level(idx,6);
    end
end

br_id_unique_dorpm=unique(br_seq_t_corr_dorpm(:,8));

name={};
for i = 1:length(br_id_unique_dorpm)
    idx=find(br_level(:,1)==br_id_unique_dorpm(i));
    name(i)=br_name(idx,2);
end


br_all_dorpm = br_all(find(br_all(:,5)==856),:);
for i = 1:length(br_all_dorpm)
    idx=find(br_level(:,1)==br_all_dorpm(i,3));
    if idx
        br_all_dorpm(i,8)=br_level(idx,6);
    end
end



statistic_t=zeros(length(br_id_unique_dorpm),3);%mean median sem
for i = 1:length(br_id_unique_dorpm)
    tmp_idx=find(br_seq_t_corr_dorpm(:,8)==br_id_unique_dorpm(i));
    tmp_idx_all=find(br_all_dorpm(:,8)==br_id_unique_dorpm(i));
    if tmp_idx
        tmp_br_all=br_all_dorpm(tmp_idx_all,:);
        mice_id=unique(tmp_br_all(:,7));
        tmp_br=br_seq_t_corr_dorpm(tmp_idx,:);
        statistic_t(i,1)=median(tmp_br(:,6));
        statistic_t(i,2)=std(tmp_br(:,6))/sqrt(length(tmp_idx));
        statistic_t(i,3)=length(tmp_idx);
        a=[];
        for j = 1:length(mice_id)
            a(j)=length(find(tmp_br_all(:,7)==mice_id(j)));
        end
        statistic_t(i,4)=sum(a>=5);%
    end
end

exclude_idx=find(sum(statistic_t,2)==0 | statistic_t(:,4)<3);
statistic_t(exclude_idx,:)=[];
br_id_unique_dorpm(exclude_idx,:)=[];
name(exclude_idx)=[];

[~,I]=sort(statistic_t(:,1),'ascend');
statistic_t=statistic_t(I,:);
fr_t=0.025;
statistic_t(:,1)=statistic_t(:,1)*fr_t;
br_id_unique_dorpm=br_id_unique_dorpm(I,:);
name=name(I);