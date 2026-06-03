%s74 parallel analysis
load('~/NW_s74_PLS.mat', 'seq_freq_traces');
[cof sco lat]=pca(seq_freq_traces');

shuffle_num=1000;
lat_shuffle=zeros(size(seq_freq_traces,1),shuffle_num);
for n = 1:shuffle_num
    tmp_data=seq_freq_traces;
    for i = 1:size(seq_freq_traces,1)
        randorder=randperm(size(seq_freq_traces,2));
        tmp_data(i,:)=tmp_data(i,randorder);
    end
    [~,~,tmp_lat]=pca(tmp_data');
    lat_shuffle(:,n)=tmp_lat;
end

threshold_lat=zeros(size(seq_freq_traces,1),1);
for i = 1:size(seq_freq_traces,1)
    [B,I]=sort(lat_shuffle(i,:),'ascend');
    threshold_lat(i)=B(ceil(shuffle_num*0.95));
end

upper_bound_dim_num=length(find((lat-threshold_lat)>=0));

