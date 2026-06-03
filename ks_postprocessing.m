%% kilosort_postprocessing
% This code used before other analysis: prepare data and single unit
% quality control. save as sxx_postprocess_0.5s.mat

addpath(genpath('~/code'));
%parameters
path="~/s74_g0/";
subj="s74";
gates="_g0";
eegemg='s74_eegemg/';
probes=["imec0","imec1","imec2","imec3"];
drug="iso";
anesth_time=33210;
onset_idx=41;
bin_time=0.5;
fr_t=0.02;
isiviol_t=0.5;
ampcutoff_t=0.1;
snr_t=2.5;
sr=30000;
tic
%% brain states
load(strcat(path,eegemg,'EEG.mat'))
load(strcat(path,eegemg,'EMG.mat'))
load(strcat(path,eegemg,'info.mat'))
load(strcat(path,eegemg,'epocs.mat'))
load(strcat(path,eegemg,'labels.mat'))
start_time=data5.epocs.PC2_.onset(onset_idx);
%start_time=0;


%%
eval(strcat(subj,"_spike_freq_good","=[];"))
% eval(strcat(subj,"_spike_freq_good_z","=[];"))
% eval(strcat(subj,"_spike_freq_good_norm","=[];"))
eval(strcat(subj,"_good_idx","={};"))
eval(strcat(subj,"_cluster_spike_good","=[];"))
eval(strcat(subj,"_all_neuron_metrics","=[];")) % subj,clusterid(orig),probeid,channelid,brid,fr,SNR,isiv,amplicutoff


for p=1:length(probes)
    path_probe=strcat(path,subj,"_out\catgt_",subj,gates,"\",subj,gates,"_",probes(p),"\");
    path_py=strcat(path,subj,"_out\catgt_",subj,gates,"\",subj,gates,"_",probes(p),"\",probes(p),"_ks2\");
    list=dir(path_py);
    if p==1
    metafile=importdata(strcat(path_probe,subj,gates,'_tcat.',probes(p),'.ap.meta'));
    metafile=metafile{12};
    time_end=str2double(metafile(14:end));
    
    end
    for i=1:length(list)-2
        temp=list(i+2).name;
        if contains(temp,'.npy')
            eval(strcat(temp(1:end-4),"=readNPY('",path_probe,probes(p),"_ks2\",temp(1:end-4),".npy');"));
        elseif contains(temp,'.csv')
            eval(strcat("[",temp(1:end-4),",~,~]=xlsread('",path_probe,probes(p),"_ks2\",temp(1:end-4),".csv');"))
        end
    end
    clusters.channels=readNPY(strcat(path_probe,'ibl_output\','clusters.channels.npy'));
    eval(strcat("waveform_metrics_imec",string(p-1),"=waveform_metrics_1;"));
    eval(strcat("spike_clusters_",string(p-1),"=spike_clusters;"));
    eval(strcat("spike_times_",string(p-1),"=spike_times;"));
%     load(strcat(path_py,"rez2.mat"));
    
    %
    fr_whole=zeros(max(spike_clusters)+1,1);
    for i = 0:max(spike_clusters)
        num_spike=length(find(spike_times(spike_clusters==i)<=(anesth_time-start_time)*sr));
        fr_whole(i+1)=num_spike/(anesth_time-start_time); % whole fr before anesth
    end
    % fr>=0.02 isiviol<=0.5 ampcutoff<=0.1 snr>=2.5
    temp_idx=fr_whole>=fr_t & metrics(:,4)<=isiviol_t & metrics(:,6)<=ampcutoff_t;
    good_idx=intersect(find(temp_idx),waveform_metrics_1(waveform_metrics_1(:,4)>=2.5,1)+1); %
    eval(strcat("fr_whole_good_imec",string(p-1),"=fr_whole(good_idx);"));
    valid_idx=find(ismember(metrics(:,1),waveform_metrics_1(:,1)));
    
    spike_freq_good=zeros(length(good_idx),floor(time_end/bin_time));
    cluster_spike_good=sparse(zeros(length(good_idx),ceil(time_end*1000)+10000));% +10s
    for i=1:length(good_idx)%max(spike_clusters)+1
        temp_i=find(spike_clusters==good_idx(i)-1);
        temp_t=double(spike_times(temp_i))/30000; % time: s
        [bin_n,bin_n1]=ismember((1:size(spike_freq_good,2)),ceil(temp_t/bin_time));
        [~,bin_n2]=ismember((1:size(spike_freq_good,2)),ceil(temp_t/bin_time),'legacy');
        spike_freq_good(i,bin_n)=(bin_n2(bin_n2>0)-bin_n1(bin_n1>0)+1);        
        [bin_ms,~]=ismember((1:size(cluster_spike_good,2)),ceil(temp_t.*1000));
%         [~,binms_n2]=ismember((1:size(cluster_spike_good,2)),ceil(temp_t.*1000),'legacy');
        cluster_spike_good(i,bin_ms)=1;
%         temp_t_bin=double(spike_times(temp_i))/30000/bin_time;
%         for j=1:floor(time_end/bin_time)
%             temp_f=length(find(ceil(temp_t_bin)==j))/bin_time;
%             spike_freq_good(i,j)=temp_f;
%         end
    end
   
    brainregion_d=importdata(strcat(path_probe,'ibl_output\','channel_locations.json'));
    channel_br=[];
    channel_br_name={};
    for i=1:length(brainregion_d)/9-1
        a=brainregion_d{i*9-7};
        a1=brainregion_d{i*9};
        a2=brainregion_d{i*9-1};
        a_i=isstrprop(a,'digit');
        a2_i=isstrprop(a2,'digit');
        channel_br(i,1)=str2double(a(a_i));
        channel_br(i,2)=str2double(a2(a2_i)); %brain region id
        channel_br_name{i}=a1(22:end-1);%brain region
        
        %     if i==1
        %     s16_imec3_channel_br(i,3)=1;
        %     elseif i>1 && s16_imec3_channel_br(i,2)==s16_imec3_channel_br(i-1,2)
        %         s16_imec3_channel_br(i,3)=s16_imec3_channel_br(i-1,3);
        %     else
        %         s16_imec3_channel_br(i,3)=s16_imec3_channel_br(i-1,3)+1;
        %     end
    end


   %
    probe_neuron_metrics=zeros(length(valid_idx),9);
    neuron_br_name={};
    s=char(subj);
    probe_neuron_metrics(:,1)=str2double(s(2:3));%subj
    probe_neuron_metrics(:,2)=waveform_metrics_1(:,1);%cluster id
    probe_neuron_metrics(:,3)=p; %probe

    cluster_channels_id=clusters.channels(valid_idx); %id+1=idx
    cluster_channel_idx=[];
    for i = 1:length(cluster_channels_id)
        cluster_channel_idx(i,1)=find(channel_br(:,1)==cluster_channels_id(i));
    end
    probe_neuron_metrics(:,4)=cluster_channels_id; %channel id
    probe_neuron_metrics(:,5)=channel_br(cluster_channel_idx,2); %br_id
    neuron_br_name(:,1)=channel_br_name(cluster_channel_idx); %br_name
    probe_neuron_metrics(:,6)=fr_whole(valid_idx);
    probe_neuron_metrics(:,7)=waveform_metrics_1(:,4);%SNR
    probe_neuron_metrics(:,8)=metrics(valid_idx,4);%ISIviolation
    probe_neuron_metrics(:,9)=metrics(valid_idx,6);%Amplitude cutoff

    
%     spike_freq_good_z=zscore(spike_freq_good,0,2);
%     spike_freq_good_norm=(spike_freq_good./nanmean(spike_freq_good,2));
    
    save(strcat(path_probe,subj,'_',probes(p),'.mat'));
    eval(strcat(subj,"_good_idx{p,1}=good_idx;"));
    eval(strcat(subj,"_good_idx{p,2}=clusters.channels;"));    
    eval(strcat(subj,"_good_idx{p,3}=channel_br;"));
    eval(strcat(subj,"_good_idx{p,4}=channel_br_name;"));
    eval(strcat(subj,"_spike_freq_good","=[",subj,"_spike_freq_good;spike_freq_good];"));
    eval(strcat(subj,"_all_neuron_metrics","=[",subj,"_all_neuron_metrics;probe_neuron_metrics];"));
%     eval(strcat(subj,"_spike_freq_good_z","=[",subj,"_spike_freq_good_z;spike_freq_good_z];"));
%     eval(strcat(subj,"_spike_freq_good_norm","=[",subj,"_spike_freq_good_norm;spike_freq_good_norm];"));
    eval(strcat(subj,"_cluster_spike_good","=[",subj,"_cluster_spike_good;cluster_spike_good];"));
end
    
 

%%
% EEG_i=EEG(round(start_time*info.samplerate)+1:end);
% EMG_i=EMG(round(start_time*info.samplerate)+1:end);
labels_i=labels(round(start_time/info.epochtime)+1:floor(time_end/2.5));
labels_i(round((anesth_time-start_time)/2.5)+1:end)=4;
labels_rem=find(labels_i==1);
labels_wake=find(labels_i==2);
labels_nrem=find(labels_i==3);
labels_anesth=find(labels_i==4);
idx_rem=find(ismember(ceil(spike_times/30000/2.5),labels_rem));
idx_wake=find(ismember(ceil(spike_times/30000/2.5),labels_wake));
idx_nrem=find(ismember(ceil(spike_times/30000/2.5),labels_nrem));
idx_anesth=find(ismember(ceil(spike_times/30000/2.5),labels_anesth));


