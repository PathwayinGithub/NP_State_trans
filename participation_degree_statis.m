%%  participation degree statistics.
% subj={'s16','s22','s26','s27','s28','s29','s30','s33','s34','s35','s36','s38','s42',...
%     's43','s48','s49','s51','s53','s54','s59','s60','s61','s62','s63','s64',...
%     's66','s67','s68','s70','s72','s74','s75','s76','s77','s78','s80','s81',...
%     's85','s86','s87','s88','s99'};%WN
subj={'s16','s22','s26','s27','s28','s29','s30','s33','s34','s35','s36','s42',...
    's43','s48','s51','s53','s54','s59','s60','s61','s62','s63','s64',...
    's66','s67','s68','s70','s72','s74','s75','s76','s77','s78','s80','s81',...
    's85','s86','s87','s88','s99'};%NW

path='Y:\post_process_output\';
load(strcat(path,'br_id_color_fig1.mat'));
br_ratio_seq=cell(size(br_id_color_fig1,1),2);
for i = 1:length(subj)
    tmp_subj=subj{i};
    load(strcat(path,'SVM\',tmp_subj,'_SVM_wake2.mat'),'idx_good_presence','trans_time2','trans_tmp_idx');

    load(strcat(path,'update_brain_region_for_fig1.mat'),strcat(tmp_subj,'_cluster_channel_br'),strcat(tmp_subj,'_cluster_channel_name'));
    eval(strcat('tmp_cluster_channel_br=',tmp_subj,'_cluster_channel_br;'));
    eval(strcat('tmp_cluster_channel_name=',tmp_subj,'_cluster_channel_name;'));

    tmp_cluster_channel_br=tmp_cluster_channel_br(idx_good_presence,:);
    tmp_cluster_channel_name=tmp_cluster_channel_name(:,idx_good_presence);
    
    tmp_cluster_channel_br_seq=tmp_cluster_channel_br(trans_tmp_idx,:);
    tmp_cluster_channel_name_seq=tmp_cluster_channel_name(:,trans_tmp_idx);


    random_ratio=length(trans_tmp_idx)/length(idx_good_presence);
    for b=1:length(br_id_color_fig1)
        num_in_seq=length(find(tmp_cluster_channel_br_seq(:,5)==br_id_color_fig1(b,1)));
        num_all=length(find(tmp_cluster_channel_br(:,5)==br_id_color_fig1(b,1)));
        if num_all>=10 %excluded <10 neuron
            tmp_ratio=num_in_seq/num_all;
            br_ratio_seq{b,1}=[br_ratio_seq{b,1};tmp_ratio];
            br_ratio_seq{b,2}=[br_ratio_seq{b,2};random_ratio];
        end
    end
end


diff_ratio=cell(length(br_ratio_seq),1);
for i = 1:size(br_ratio_seq,1)
    diff_ratio{i,1}=br_ratio_seq{i,1}-br_ratio_seq{i,2};
end


size_diff=[];
for i = 1:length(diff_ratio)
    size_diff(i)=length(diff_ratio{i, 1});
end
exclude_idx=find(size_diff<5);
diff_ratio(exclude_idx)=[];
load(strcat(path,'br_name_color_fig1.mat'));
br_id_color_fig1(exclude_idx,:)=[];
br_id_name(exclude_idx)=[];
size_diff(exclude_idx)=[];

a=[];
for i = 1:size(diff_ratio,1)
    a(i)=mean(diff_ratio{i, 1});
end

[~,I]=sort(a,'descend');
tmp_diff_ratio=diff_ratio(I,:);
tmp_name=br_id_name(I);
size_diff=size_diff(I);
tmp_name=tmp_name'; 


diff_table=zeros(length(tmp_diff_ratio),max(size_diff));
for i = 1:length(tmp_diff_ratio)
    if size_diff(i)~=0
        diff_table(i,1:size_diff(i))=tmp_diff_ratio{i, 1}';
    end
end
diff_table(diff_table==0)=NaN;
diff_table=diff_table'; 





%% specific brain region （dorpm)  
subj={'s16','s22','s26','s27','s28','s29','s30','s33','s34','s35','s36','s38','s42',...
    's43','s48','s49','s51','s53','s54','s59','s60','s61','s62','s63','s64',...
    's66','s67','s68','s70','s72','s74','s75','s76','s77','s78','s80','s81',...
    's85','s86','s87','s88','s99'};%WN



path='Y:\post_process_output\';
load(strcat(path,'br_id_color_fig1.mat'));
dorpm=[];
dorpm_seq=[];
random_ratio=zeros(length(subj),2);
for i =1:length(subj)
    tmp_subj=subj{i};
    load(strcat(path,'SVM\',tmp_subj,'_SVM2.mat'),'idx_good_presence','trans_time2','trans_tmp_idx');

    load(strcat(path,'update_brain_region_for_fig1.mat'),strcat(tmp_subj,'_cluster_channel_br'),strcat(tmp_subj,'_cluster_channel_name'));
    eval(strcat('tmp_cluster_channel_br=',tmp_subj,'_cluster_channel_br;'));
    eval(strcat('tmp_cluster_channel_name=',tmp_subj,'_cluster_channel_name;'));

    tmp_cluster_channel_br=tmp_cluster_channel_br(idx_good_presence,:);
    tmp_cluster_channel_name=tmp_cluster_channel_name(:,idx_good_presence);
    
    tmp_cluster_channel_br_seq=tmp_cluster_channel_br(trans_tmp_idx,:);
    tmp_cluster_channel_name_seq=tmp_cluster_channel_name(:,trans_tmp_idx);

    tmp_idx=find(tmp_cluster_channel_br(:,5)==856);
    tmp_br=tmp_cluster_channel_br(tmp_idx,:);
    tmp_br(:,6)=str2double(tmp_subj(2:end));
    dorpm=[dorpm;tmp_br];

    tmp_idx=find(tmp_cluster_channel_br_seq(:,5)==856);
    tmp_br=tmp_cluster_channel_br_seq(tmp_idx,:);
    tmp_br(:,6)=str2double(tmp_subj(2:end));
    dorpm_seq=[dorpm_seq;tmp_br];
    random_ratio(i,1)=str2double(tmp_subj(2:end));
    random_ratio(i,2)=length(trans_tmp_idx)/length(idx_good_presence);

end

load('Y:\SGL_DATA\br.mat');
br_unique=unique(dorpm_seq(:,3));
for i =1:length(br_unique)
    idx1=find(br_level(:,1)==br_unique(i));
    idx=find(dorpm_seq(:,3)==br_unique(i));
    dorpm_seq(idx,7)=br_level(idx1,6);
    idx=find(dorpm(:,3)==br_unique(i));
    dorpm(idx,7)=br_level(idx1,6);
end


br_unique=unique(dorpm_seq(:,7));
mice_unique=unique(dorpm_seq(:,6));
br_num=cell(length(br_unique),1);
for i = 1:length(mice_unique)
    m_id=mice_unique(i);
    tmp_idx_all=find(dorpm(:,6)==m_id);
    tmp_dorpm=dorpm(tmp_idx_all,:);

    tmp_idx_seq=find(dorpm_seq(:,6)==m_id);
    tmp_dorpm_seq=dorpm_seq(tmp_idx_seq,:);
    for j=1:length(br_unique)
        br_id=br_unique(j);
        num_all=length(find(tmp_dorpm(:,7)==br_id));
        num_seq=length(find(tmp_dorpm_seq(:,7)==br_id));
        if num_all>=5  % excluded <5 neuron
            ratio=num_seq/num_all;
            diff_ratio=ratio-random_ratio(find(random_ratio(:,1)==m_id),2);
            br_num{j,1}=[br_num{j,1};diff_ratio];
        end
    end
end


name={};
for i = 1:length(br_unique)
    idx=find(br_level(:,1)==br_unique(i));
    name(i)=br_name(idx,6);
end

a=[];
for i = 1:size(br_num,1)
    a(i)=mean(br_num{i, 1});
end

[~,I]=sort(a,'descend');

br_num=br_num(I,:);
name=name(I);




size_diff=[];
for i = 1:length(br_num)
    size_diff(i)=length(br_num{i, 1});
end


diff_table=zeros(length(br_num),max(size_diff));
for i = 1:length(br_num)
    if size_diff(i)~=0
        diff_table(i,1:size_diff(i))=br_num{i, 1}';
    end
end
diff_table(diff_table==0)=NaN;
diff_table=diff_table';

diff_table(:,find(size_diff<3))=[]; 
name(find(size_diff<3))=[];
name=name'; 