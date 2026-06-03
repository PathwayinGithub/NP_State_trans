%% GPR time decoding

subj='s74';
addpath(genpath('~/code'));
filepath='~/';
load(strcat(filepath,subj,'_SVM_wake2.mat'),'temp_spike_freq','trans_tmp_idx','I_trans','left_win','right_win','idx_good_presence','averaged_freq_sorted','neuron_trans_freq_trans');
load(strcat(filepath,'update_brain_region_for_fig1.mat'), strcat(subj,'_cluster_channel_br'),strcat(subj,'_cluster_channel_name'));
load(strcat(filepath,'br_id_color_fig1.mat'));


%prepare data for GPRmodel trans process predict
neuron_trans_freq=neuron_trans_freq_trans(trans_tmp_idx,:,:);
neuron_trans_freq=neuron_trans_freq(I_trans,:,:);
num_neurons=size(neuron_trans_freq,1);
len_win=size(neuron_trans_freq,2);
test_ratio=0.2; % 20 percent as test data
all_trial_num=size(neuron_trans_freq,3);
test_trial_num=round(all_trial_num*test_ratio);
train_trial_num=all_trial_num-test_trial_num;

t_offsets_label=1:len_win;
t_offsets=repmat(t_offsets_label,1,train_trial_num);


tmp_data=zscore(neuron_trans_freq(:,:,1:train_trial_num),0,2);
tmp_data(tmp_data<=0.8)=0; % binary
tmp_data(tmp_data~=0)=1;
tmp_data=reshape(tmp_data(:,:,:),num_neurons,[]);
data=[tmp_data;t_offsets];
[trainedModel, ~] = trainRegressionModel(data);

test_trial=(train_trial_num+1):all_trial_num;
yfit=[];
test_RMSE=[];
test_R2_i=[];
for i = 1:test_trial_num
    test=neuron_trans_freq(:,:,test_trial(i));
    test=zscore(test,0,2);
    test(test<=0.8)=0;
    test(test~=0)=1;
    yfit(i,:) = trainedModel.predictFcn(test);
    test_RMSE(i)=sqrt(sum((yfit(i,:)-t_offsets_label).^2)/length(t_offsets_label));
    test_R2_i(i)=1 - (sum((yfit(i,:)- t_offsets_label).^2) / sum((t_offsets_label - mean(t_offsets_label)).^2));
end

fr_t=0.025;
%pixel_t=0.25;%s WN
pixel_t=0.2;%s NW
pixel_grid_num=pixel_t/fr_t;
grid_num=len_win/pixel_grid_num;
y_fit_0d1=zeros(size(yfit,1),grid_num);
for i = 1:test_trial_num
    for j =1:pixel_grid_num:len_win
        y_fit_0d1(i,floor(j/pixel_grid_num)+1)=floor(mean(yfit(i,j:j+pixel_grid_num-1)/pixel_grid_num))+1;
    end
end

yfit_m=zeros(grid_num,grid_num);
for i = 1:grid_num
    for j = 1:test_trial_num
        yfit_m(i,y_fit_0d1(j,i))=yfit_m(i,y_fit_0d1(j,i))+1;
    end
end
figure('Color','w');
t=tiledlayout(1,1);
ax2=axes(t);
imagesc(ax2,yfit_m');
clim([0 test_trial_num])
set(gca,'YDir','normal');
colormap(slanCM('Blues'));
%colorbar;
ax2.XAxisLocation='top';
ax2.YAxisLocation='right';
set(gca,'XTick',[]);
set(gca,'YTick',[]);
ax1=axes(t);
%plot(ax1,(1:10:len_win)+pixel_grid_num/2,yfit(1,1:10:len_win),'Color','k','LineWidth',3); % example WN
plot(ax1,(1:10:len_win)+pixel_grid_num/2,yfit(2,1:10:len_win),'Color','k','LineWidth',3); % example NW
ax1.Color='none';
ax1.Box='off';
xlabel('Time labels (25ms/bin)','FontSize',16);
ylabel('Decoded time labels (25ms/bin)','FontSize',16);
ylim([0 len_win]);



% random data 
rng(0)
temp_spike_freq=temp_spike_freq(trans_tmp_idx,:);
temp_spike_freq_slid_3=movmean(temp_spike_freq,25,2);  
randomOrder=randperm(size(temp_spike_freq_slid_3,2));
neuron_trans_freq_r=zeros(num_neurons,len_win,all_trial_num);
for t = 1:all_trial_num
    tmp_idx=randomOrder(t);
    neuron_trans_freq_r(:,:,t)=temp_spike_freq_slid_3(:,tmp_idx-left_win:tmp_idx+right_win-1);
end

tmp_data=zscore(neuron_trans_freq_r(:,:,1:train_trial_num),0,2);
tmp_data(tmp_data<=0.8)=0; % binary
tmp_data(tmp_data~=0)=1;
tmp_data=reshape(tmp_data(:,:,:),num_neurons,[]);
data=[tmp_data;t_offsets];
[trainedModel, ~] = trainRegressionModel(data);

test_trial=(train_trial_num+1):all_trial_num;
yfit_r=[];
test_RMSE_r=[];
test_R2_i_r=[];
for i = 1:test_trial_num
    test=neuron_trans_freq_r(:,:,test_trial(i));   
    test=zscore(test,0,2);
    test(test<=0.8)=0;
    test(test~=0)=1;
    yfit_r(i,:) = trainedModel.predictFcn(test);
    test_RMSE_r(i)=sqrt(sum((yfit_r(i,:)-t_offsets_label).^2)/length(t_offsets_label));
    test_R2_i_r(i)=1 - (sum((yfit_r(i,:)- t_offsets_label).^2) / sum((t_offsets_label - mean(t_offsets_label)).^2));
end

%pixel_t=0.25;%s
pixel_t=0.2;%s
pixel_grid_num=pixel_t/fr_t;
grid_num=len_win/pixel_grid_num;
y_fit_0d1=zeros(size(yfit_r,1),grid_num);
for i = 1:test_trial_num
    for j =1:pixel_grid_num:len_win
        y_fit_0d1(i,floor(j/pixel_grid_num)+1)=floor(mean(yfit_r(i,j:j+pixel_grid_num-1)/pixel_grid_num))+1;
    end
end

yfit_m=zeros(grid_num,grid_num);
for i = 1:grid_num
    for j = 1:test_trial_num
        yfit_m(i,y_fit_0d1(j,i))=yfit_m(i,y_fit_0d1(j,i))+1;
    end
end
figure('Color','w');
t=tiledlayout(1,1);
ax2=axes(t);
imagesc(ax2,yfit_m');
clim([0 test_trial_num])
set(gca,'YDir','normal');
colormap(slanCM('Blues'));
%colorbar;
ax2.XAxisLocation='top';
ax2.YAxisLocation='right';
set(gca,'XTick',[]);
set(gca,'YTick',[]);
ax1=axes(t);
%plot(ax1,(1:10:len_win)+pixel_grid_num/2,yfit_r(2,1:10:len_win),'Color','k','LineWidth',3); % example
plot(ax1,(1:10:len_win)+pixel_grid_num/2,yfit_r(4,1:10:len_win),'Color','k','LineWidth',3); % example
ax1.Color='none';
ax1.Box='off';
xlabel('Time labels (25ms/bin)','FontSize',16);
ylabel('Decoded time labels (25ms/bin)','FontSize',16);
ylim([0 len_win]);

