% PLSregression of wake to nrem sequence, same for NW
% results are saved as WN_sxx_PLS.mat or NW_sxx_PLS.mat
% X_proj_trials are saved as WN_sxx_proj_x.mat/ NW_sxx_proj_x.mat
subj='s74';
filepath_main='~\mice_example_data';
load(strcat(filepath_main,'update_brain_region_for_fig1.mat'), strcat(subj,'_cluster_channel_br'),strcat(subj,'_cluster_channel_name'));
load(strcat(filepath_main,subj,'_SVM2.mat'));
eval(strcat('tmp_cluster_channel_br=',subj,'_cluster_channel_br(idx_good_presence,:);'));
tmp_cluster_channel_br=tmp_cluster_channel_br(trans_tmp_idx,:);
tmp_cluster_channel_br=tmp_cluster_channel_br(I_trans,:);
eval(strcat('tmp_cluster_channel_name=',subj,'_cluster_channel_name(:,idx_good_presence);'));
tmp_cluster_channel_name=tmp_cluster_channel_name(:,trans_tmp_idx);
tmp_cluster_channel_name=tmp_cluster_channel_name(:,I_trans);


temp_spike_freq_slid_3=movmean(temp_spike_freq,25,2);  %......................
neuron_trans_freq_fixedpoints=zeros(size(temp_spike_freq_slid_3,1),300,length(idx_medium23));
for t = 1:length(idx_medium23)
    tmp_idx=table_states_corr(idx_medium23(t)+1,4);
    neuron_trans_freq_fixedpoints(:,:,t)=temp_spike_freq_slid_3(:,tmp_idx-150:tmp_idx+150-1);
end


% prepare X
seq_freq_fixedpoints=neuron_trans_freq_fixedpoints(trans_tmp_idx,:,:);
seq_freq_fixedpoints=zscore(seq_freq_fixedpoints(I_trans,:,:),0,2);
seq_freq=seq_freq_fixedpoints(:,151-(right_win+left_win)/2:151+(right_win+left_win)/2-1,:);

X_neuron_range=1:length(trans_tmp_idx);
X_neuron_num=length(X_neuron_range);
X_time_range=1:(left_win+right_win);
X_time_num=length(X_time_range);
trial_num=size(seq_freq,3);

seq_freq_traces=reshape(seq_freq(X_neuron_range,X_time_range,:),X_neuron_num,[]);
figure;
imagesc(zscore(mean(seq_freq,3),0,2));
colormap(flipud(othercolor('RdBu9')));
clim([-2.5 2.5]);

%% prepare supervised Y label
% use repeated averaged data peak
Y_labels=zeros(X_neuron_num,X_time_num,trial_num);
half_win=25;
for i = 1:length(B)
    x=linspace(-5,5,2*half_win+1);
    y=normpdf(x,0,1)*5;
    right_i=min(X_time_num-B(i),half_win);
    left_i=min(B(i)-1,half_win);
    Y_labels(i,max(B(i)-half_win,1):min(B(i)+half_win,X_time_num),:)=repmat(y(half_win+1-left_i:half_win+1+right_i),[1,1,trial_num]);
end

figure;
imagesc(mean(Y_labels,3));
colormap(flipud(othercolor('RdBu9')));
clim([-3 3]);

Y_labels=Y_labels(X_neuron_range,X_time_range,:);
Y_labels=reshape(Y_labels,X_neuron_num,[]);

%% fit regression model
% test which component is most vital
ncomp=370;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(seq_freq_traces',Y_labels',ncomp,'cv',5);
drop_out=20;
global_sim_rmv=[];
for i = 1:drop_out
    tmp_XL=XL;
    tmp_XL(:,i)=0;
    X_hat=(XS*tmp_XL')';
    X_hat_trials=reshape(X_hat,X_neuron_num,X_time_num,[]);
    A=zscore(mean(X_hat_trials,3),0,2);
    ref=zscore(mean(seq_freq(X_neuron_range,X_time_range,:),3),0,2);
    [global_sim_rmv(i) ~] = ssim(double(A>=1),double(ref>=1),'Exponents',[0.1 1 0.1]);
end
figure('Color','white');
dissim=1-global_sim_rmv;
plot(dissim,'k','LineWidth',2);
hold on
scatter(1:5,dissim(1:5),50,'k','filled');
ylabel('Dissimilarity','FontSize',24);
xlabel('PLS dimensions');
ylim([0 0.55]);
set(gca,'FontSize',18);
hold on
yyaxis right
plot(PCTVAR(1,1:20).*100);
ylabel('Variance explained','FontSize',24);

ncomp=4;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(seq_freq_traces',Y_labels',ncomp,'cv',5);
%XL(:,[1])=0;
X_hat=(XS*XL')';
X_hat_trials=reshape(X_hat,X_neuron_num,X_time_num,[]);
A=zscore(mean(X_hat_trials,3),0,2);
ref=zscore(mean(seq_freq(X_neuron_range,X_time_range,:),3),0,2);
[global_sim_rmv ~] = ssim(double(A>=1),double(ref>=1),'Exponents',[0.1 1 0.1]);


%% prepare data for flow field
ncomp=4;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(seq_freq_traces',Y_labels',ncomp,'cv',5);
tmp_XL=XL;
%tmp_XL(:,1)=0;
X_hat=(XS*tmp_XL')';
X_hat_trials=reshape(X_hat,X_neuron_num,X_time_num,[]);
A=zscore(mean(X_hat_trials,3),0,2);
ref=zscore(mean(seq_freq(X_neuron_range,X_time_range,:),3),0,2);
[global_sim_rmv local_sim] = ssim(double(A>=1),double(ref>=1),'Exponents',[0.1 1 0.1]);

%P=XL'*((XL*XL')^(-1))*XL;
P=XL*((XL'*XL)^(-1))*XL';%
seq_freq_traces_fixedpoints=reshape(seq_freq_fixedpoints(:,:,:),X_neuron_num,[]);
X_proj=P*seq_freq_traces_fixedpoints;

X_proj_trials=reshape(X_proj,X_neuron_num,300,[]); %save as WN_sxx_proj_x.mat/ NW_sxx_proj_x.mat







