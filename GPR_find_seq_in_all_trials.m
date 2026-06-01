% code using GPR model trained by good curves trials to find seq in other
% save to sxx_SVM2.mat(WN)/sxx_SVM_wake2.mat(NW);
% trials
addpath(genpath('X:\code'));
subj='s74';
filepath='~\post_process_output\SVM\';
%prepare data for GPRmodel trans process predict
load(strcat(filepath,subj,'_SVM.mat'),'temp_spike_freq','trans_tmp_idx','table_states_corr','I_trans','idx_medium23','idx_medium23_good_curve','left_win','right_win','idx_good_presence');
load('~\post_process_output\update_brain_region_for_fig1.mat', strcat(subj,'_cluster_channel_br'),strcat(subj,'_cluster_channel_name'));
eval(strcat('tmp_cluster_channel_br=',subj,'_cluster_channel_br;'));
eval(strcat('tmp_cluster_channel_name=',subj,'_cluster_channel_name;'));



table_states_corr(idx_medium23_good_curve+1,4)=table_states_corr(idx_medium23_good_curve+1,4)-floor((left_win+right_win)/2)+right_win;
table_states_corr(idx_medium23_good_curve,5)=table_states_corr(idx_medium23_good_curve+1,4)-1;


%training GPRmodel by good trials
temp_spike_freq_slid=movmean(temp_spike_freq,25,2);
num_neurons=size(temp_spike_freq_slid,1);
trial_num=length(idx_medium23_good_curve);
neuron_trans_freq=zeros(num_neurons,left_win+right_win,trial_num);
for t = 1:length(idx_medium23_good_curve)
    tmp_idx=table_states_corr(idx_medium23_good_curve(t)+1,4);
    neuron_trans_freq(:,:,t)=temp_spike_freq_slid(:,tmp_idx-floor((left_win+right_win)/2):tmp_idx+ceil((left_win+right_win)/2)-1);
end



t_offsets_label=1:(left_win+right_win);
t_offsets=repmat(t_offsets_label,1,trial_num);
tmp_data=zscore(neuron_trans_freq(:,:,:),0,2);
tmp_data(tmp_data<=0.8)=0; % binary
tmp_data(tmp_data~=0)=1;
tmp_data=reshape(tmp_data(:,:,:),num_neurons,[]);
data=[tmp_data;t_offsets];
[trainedModel, ~] = trainRegressionModel(data);


% find trans_time and correct in trials notgood
a=[];
for i = 1:length(idx_medium23)
    if table_states_corr(idx_medium23(i)+1,4)<=300 | table_states_corr(idx_medium23(i)+1,4)>=(size(temp_spike_freq,2)-300)
        a=[a;i];
    end
end
idx_medium23(a)=[];

idx_medium23_notgood=idx_medium23;
idx_medium23_notgood(ismember(idx_medium23,idx_medium23_good_curve))=[];

fr_t=0.025;
num_neurons=size(temp_spike_freq_slid,1);
trial_num=length(idx_medium23_notgood);
trial_length=left_win+right_win;

t_offsets_label=1:trial_length;
RMSE=zeros(400,trial_num);
R2=zeros(400,trial_num);
parfor i = 1:trial_num
    idx=table_states_corr(idx_medium23_notgood(i)+1,4);
    tmp_freq=temp_spike_freq_slid(:,idx-200-floor(trial_length/2):idx+200+ceil(trial_length/2)-1); 
    tmp_RMSE=[];
    tmp_R2=[];
    for j = 1:400
        A=zscore(tmp_freq(:,j:(j+trial_length-1)),0,2);
        A(A<=0.3)=0;
        A(A~=0)=1;
        yfit = trainedModel.predictFcn(A);
        tmp_RMSE(j)=sqrt(sum((yfit-t_offsets_label').^2)/length(t_offsets_label));
        tmp_R2(j)=1 - (sum((yfit- t_offsets_label').^2) / sum((t_offsets_label' - mean(t_offsets_label')).^2));
    end
    RMSE(:,i)=tmp_RMSE;
    R2(:,i)=tmp_R2;
end


[M,trans_time2]=max(R2,[],1);
offset2=trans_time2-200;
table_states_corr(idx_medium23_notgood+1,4)=table_states_corr(idx_medium23_notgood+1,4)+offset2';
table_states_corr(idx_medium23_notgood,5)=table_states_corr(idx_medium23_notgood+1,4)-1;


% find sequence
% by peak entropy 

averaged_freq=[];
cluster_channel_br=[];
cluster_channel_br_all=[];
cluster_channel_name={};
cluster_channel_name_all={};
PE_p_value_all=[];
pos_re_trans=[];


% PE calculate
neuron_trans_freq=zeros(size(temp_spike_freq_slid,1),left_win+right_win,length(idx_medium23));
for t = 1:length(idx_medium23)
    tmp_idx=table_states_corr(idx_medium23(t)+1,4);
    neuron_trans_freq(:,:,t)=temp_spike_freq_slid(:,tmp_idx-floor((left_win+right_win)/2):tmp_idx+ceil((left_win+right_win)/2)-1);
end

% find fixed peak pattern
fluc_trial=squeeze(mean(abs(zscore(neuron_trans_freq(:,:,:),0,2)),2));
presence_ratio_t=0.5;
trial_num_t=floor(presence_ratio_t*length(idx_medium23));
high_presence_neuron=find(sum(fluc_trial==0,2)<=trial_num_t);

tmp_data=neuron_trans_freq(:,:,:); 
[~,pos_max] = max(tmp_data(:,:,:),[],2);
pos_max=squeeze(pos_max);

peak_m=zeros(size(tmp_data,1),size(tmp_data,3),size(tmp_data,2));
for k = 1:size(tmp_data,3)
    for j = 1:size(tmp_data,1)
        peak_m(j,k,pos_max(j,k))=1;
    end
end
PE= SeqIndexDB(peak_m,10);

% Bootstrap for PE distribution of each neuron
num_trials=length(idx_medium23);
PE_bootstrap=[];
parfor sample_num=1:10000
    randomOrder=randperm(size(temp_spike_freq_slid,2)-(left_win+right_win),num_trials)+ceil((left_win+right_win)/2); % 排除边界问题
    pseudo_trial_freq=zeros(size(temp_spike_freq_slid,1),left_win+right_win,num_trials);
    for m = 1:num_trials
        pseudo_trial_freq(:,:,m)=temp_spike_freq_slid(:,randomOrder(m)-floor((left_win+right_win)/2):randomOrder(m)+ceil((left_win+right_win)/2)-1);
    end
    
    tmp_data=pseudo_trial_freq(:,:,:); 
    [~,pos_max] = max(tmp_data(:,:,:),[],2);
    pos_max=squeeze(pos_max);
    
    peak_m=zeros(size(tmp_data,1),size(tmp_data,3),size(tmp_data,2));
    for k = 1:size(tmp_data,3)
        for j = 1:size(tmp_data,1)
            peak_m(j,k,pos_max(j,k))=1;
        end
    end
    PE_bootstrap(:,sample_num)= SeqIndexDB(peak_m,10);
end

PE_threshold=[];
PE_p_value=[];
for j = 1:size(PE_bootstrap,1)
    [B_PE,~]=sort(PE_bootstrap(j,:),'descend');
    PE_threshold(1,j)=B_PE(3000);%在分布前30%
    PE_p_value(j)=sum(PE_bootstrap(j,:)>=PE(j))/10000;
end

stable_peak_neuron=find((PE-PE_threshold)>=0);
neuron_hp_sp=intersect(high_presence_neuron,stable_peak_neuron);

tmp_idx=find(tmp_cluster_channel_br(:,5)~=0);
tmp_idx=intersect(tmp_idx,neuron_hp_sp);

trans_tmp_idx=tmp_idx;
averaged_freq=mean(neuron_trans_freq(tmp_idx,:,:),3);
cluster_channel_br=tmp_cluster_channel_br(tmp_idx,:);
cluster_channel_br_all=tmp_cluster_channel_br;
cluster_channel_name=tmp_cluster_channel_name(:,tmp_idx);
cluster_channel_name_all=tmp_cluster_channel_name;
PE_p_value_all=PE_p_value;


% sorted according to averaged trial max peak 
%averaged_freq_z=zscore(mean(neruon_trans_freq_trans(a,:,:),3),0,2);
averaged_freq_z=zscore(averaged_freq,0,2);
[~,pos_max]=max((averaged_freq_z),[],2);
[B,I]=sort(pos_max,'ascend');
averaged_freq_sorted=averaged_freq_z(I,:);
I_trans=I;                                 % ...
B_trans=B;
neuron_trans_freq_trans=neuron_trans_freq; % ...
figure('Color','w');
imagesc(averaged_freq_sorted);
%colormap(slanCM('bwr'));
colormap(flipud(othercolor('RdBu9')));
clim([-2 2]);
%set(gca,'YDir','normal');
xlabel('Time bins','FontSize',18);
ylabel('Neurons','FontSize',18);




