% Focality of sequence neurons' distribution code
addpath(genpath('X:\code'));
% subj={'s16','s22','s26','s27','s28','s29','s30','s33','s34','s35','s36','s38','s42',...
%     's43','s48','s49','s51','s53','s54','s59','s60','s61','s62','s63','s64',...
%     's66','s67','s68','s70','s72','s74','s75','s76','s77','s78','s80','s81',...
%     's85','s86','s87','s88','s99'};%WN 
subj={'s16','s22','s26','s27','s28','s29','s30','s33','s34','s35','s36','s42',...
    's43','s48','s51','s53','s54','s59','s60','s61','s62','s63','s64',...
    's66','s67','s68','s70','s72','s74','s75','s76','s77','s78','s80','s81',...
    's85','s86','s87','s88','s99'};%NW
cluster_channel_br=[];
cluster_channel_name={};
cluster_channel_br_seq=[];
cluster_channel_name_seq={};
num_seq_neurons=[];
for m=1:length(subj) 
    tmp_subj=subj{m};
    filepath1='Y:\post_process_output\SVM\';
    filepath2='Y:\post_process_output\';
    load(strcat(filepath1,tmp_subj,'_SVM_wake2.mat'),'trans_tmp_idx','idx_good_presence');
    num_seq_neurons(m)=length(trans_tmp_idx);
    load(strcat(filepath2,'update_brain_region_for_fig1.mat'),strcat(tmp_subj,'_cluster_channel_br'));
    eval(strcat('tmp_cluster_channel_br=',tmp_subj,'_cluster_channel_br(idx_good_presence,:);'));
    tmp_cluster_channel_br(:,6)=str2double(tmp_subj(2:end));
    cluster_channel_br=[cluster_channel_br;tmp_cluster_channel_br];
    tmp_cluster_channel_br_seq=tmp_cluster_channel_br(trans_tmp_idx,:);
    cluster_channel_br_seq=[cluster_channel_br_seq;tmp_cluster_channel_br_seq];

    load(strcat(filepath2,'update_brain_region_for_fig1.mat'),strcat(tmp_subj,'_cluster_channel_name'));
    eval(strcat('tmp_cluster_channel_name=',tmp_subj,'_cluster_channel_name(:,idx_good_presence);'));
    cluster_channel_name=cat(2,cluster_channel_name,tmp_cluster_channel_name);
    tmp_cluster_channel_name_seq=tmp_cluster_channel_name(:,trans_tmp_idx);
    cluster_channel_name_seq=cat(2,cluster_channel_name_seq,tmp_cluster_channel_name_seq);
end

br_id_unique=unique(cluster_channel_br(:,5));
for i=1:length(br_id_unique)
    br_id_unique(i,2)=length(find(cluster_channel_br(:,5)==br_id_unique(i)));
    br_id_unique(i,3)=length(find(cluster_channel_br_seq(:,5)==br_id_unique(i)));
end
br_id_unique(find(br_id_unique(:,2)<10),:)=[];
Pa=br_id_unique(:,3)./br_id_unique(:,2);
F_real=sum(Pa.^2)/(sum(Pa).^2);

br_name_unique={};
for i = 1:length(br_id_unique)
    idx=find(cluster_channel_br(:,5)==br_id_unique(i,1));
    br_name_unique(i)=cluster_channel_name(2,idx(1));
end



bootstrp_num=10000;
br_num_random=zeros(size(br_id_unique,1),bootstrp_num);
for b=1:bootstrp_num
    cluster_channel_br_seq=[];
    for m=1:length(subj)
        tmp_subj=subj{m};
        idx=find(cluster_channel_br(:,6)==str2double(tmp_subj(2:end)));
        tmp_cluster_channel_br=cluster_channel_br(idx,:);
        p=randperm(size(tmp_cluster_channel_br,1),num_seq_neurons(m));
        tmp_cluster_channel_br_seq=tmp_cluster_channel_br(p,:);
        cluster_channel_br_seq=[cluster_channel_br_seq;tmp_cluster_channel_br_seq];
    end
    for i=1:length(br_id_unique)
        br_num_random(i,b)=length(find(cluster_channel_br_seq(:,5)==br_id_unique(i)));
    end
end
Pa_bs=br_num_random(:,:)./repmat(br_id_unique(:,2),[1 bootstrp_num]);

F_pesudo=[];
for b=1:bootstrp_num
    tmp_Pa=Pa_bs(:,b);
    F_pesudo(b)=sum(tmp_Pa.^2)/(sum(tmp_Pa).^2);
end



[B,~]=sort(F_pesudo,'descend');
B(500)

figure('Color','w');
histogram(F_pesudo,25);
title('sequence neurons brain region focality index')
hold on
plot([F_real F_real],[0 700],'--r')
hold on
plot([B(500) B(500)],[0 700],'--b')
%legend(' ','Seq Focality','P=0.05');
axis off;



[B2,I2]=sort(Pa,'descend');
a=br_name_unique;
a=a(I2);
