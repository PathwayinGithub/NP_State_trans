% analyze attractors at nrem or wake states
clc;
clear;
addpath(genpath('X:\code'));
subj='s38';

filepath='Y:\post_process_output\';
load(strcat(filepath,'update_brain_region_for_fig1.mat'),strcat(subj,'_cluster_channel_br'));
load(strcat(filepath,'update_brain_region_for_fig1.mat'),strcat(subj,'_cluster_channel_name'));
eval(strcat('tmp_cluster_channel_br=',subj,'_cluster_channel_br;'));
eval(strcat('tmp_cluster_channel_name=',subj,'_cluster_channel_name;'));

matfile=strcat(filepath,strcat(subj,'_postprocess_0.5s.mat'));
load(matfile, strcat(subj,'_cluster_spike_good'));
load(matfile, 'start_time'); 
load(matfile, 'labels');
load(matfile, 'EEG');
load(matfile, 'EMG');
load(matfile, 'info');

load(strcat(filepath,'start_stop_SVM.mat'));
start=start_stop_SVM(find(start_stop_SVM(:,1)==str2double(subj(2:end))),2)*3600; % EEG start hour *
stop=start_stop_SVM(find(start_stop_SVM(:,1)==str2double(subj(2:end))),3)*3600; % stop


%frequency
eval(strcat('temp_spike=',subj,'_cluster_spike_good;'));
fr_t=0.1;%freq_time s
temp_spike_freq=temp_spike(:,round(start-start_time)*1000+1:round(stop-start_time)*1000);
temp_spike_freq=temp_spike_freq(:,1:round(size(temp_spike_freq,2)/(fr_t/0.001))*(fr_t/0.001))';
temp_spike_freq=full(temp_spike_freq);
temp_spike_freq=reshape(temp_spike_freq,(fr_t/0.001),[],size(temp_spike_freq,2));
temp_spike_freq=squeeze(sum(temp_spike_freq,1))';

%label
label_bin=2.5;
labels_i=labels(round(start/label_bin)+1:round(stop/label_bin)+1);
temp_labels=repmat(labels_i,[1 label_bin/fr_t]);
temp_labels=reshape(temp_labels',[],1);
temp_labels=temp_labels(1:size(temp_spike_freq,2));
table_states=stable_states(temp_labels,fr_t);

%power
sr=info.samplerate;
EEG=EEG(floor(start*sr)+1:floor(stop*sr));
EMG=EMG(floor(start*sr)+1:floor(stop*sr));
[~,~,~,eegPSD]=spectrogram(EEG,112,10,0:0.5:30,sr); 
eeg_delta=sum(eegPSD(2:9,:)); % 0.5-4

delta_frt=[];
for i=1:size(temp_spike_freq,2)
    delta_frt(i,1)=eeg_delta(min(ceil((i*fr_t*sr-56)/102)+1,length(eeg_delta)));
end
delta_frt_s=movmean(delta_frt,5,1);

%find long WN(2to3)/NW(3to2) trial
long_period=10;%s
tmp_dif=diff(table_states(:,1));
idx_23=find(tmp_dif==3); %3 WN
idx_long23=[];
for  i = 1:length(idx_23)
    tmp_idx=idx_23(i);
    if table_states(tmp_idx,2)>=long_period/fr_t && table_states(tmp_idx+1,2)>=long_period/fr_t
        idx_long23=[idx_long23;tmp_idx];
    end
end

load(strcat(filepath,'SVM\',subj,'_SVM2.mat'), 'idx_good_presence');
temp_spike_freq=temp_spike_freq(idx_good_presence,:);

% for shuffled data
% temp_spike_freq_shuffle=zeros(size(temp_spike_freq,1),size(temp_spike_freq,2));
% for i = 1:size(temp_spike_freq,1)
%     randorder=randperm(size(temp_spike_freq,2));
%     temp_spike_freq_shuffle(i,:)=temp_spike_freq(i,randorder);
% end
% temp_spike_freq=temp_spike_freq_shuffle;

% PCA
num_neurons=size(temp_spike_freq,1);
trial_num=length(idx_long23);
half_win_t=7.5;%s
time_num=half_win_t/fr_t*2;
spike_freq_2to3_1=zeros(num_neurons,time_num,trial_num);
delta_2to3=zeros(time_num,trial_num);
for i = 1:trial_num
    tmp_idx=idx_long23(i)+1;
    half_win3=half_win_t/fr_t;
    tmp_spike_freq=temp_spike_freq(:,table_states(tmp_idx,4)-half_win3:table_states(tmp_idx,4)+half_win3-1);
    tmp_labels=temp_labels(table_states(tmp_idx,4)-half_win3:table_states(tmp_idx,4)+half_win3-1);
    tmp_delta=delta_frt_s(table_states(tmp_idx,4)-half_win3:table_states(tmp_idx,4)+half_win3-1);
    spike_freq_2to3_1(:,:,i)=tmp_spike_freq;
    labels_2to3_1=tmp_labels;
    delta_2to3(:,i)=tmp_delta;
end

%zscored trial by trial
spike_freq_2to3_z=zscore(spike_freq_2to3_1,0,2);
delta_2to3_z=zscore(delta_2to3,0,1);
data=reshape(spike_freq_2to3_z,num_neurons,[])';

[cof sco lat]=pca(data);
sco=sco';
sco_trial=reshape(sco,[],half_win_t*2/fr_t,trial_num);



% plot figures
%% corr % 2D PLOT(FIG2 AB)
pc_trial_corr=[];
for i = 1:num_neurons
    for j = 1:size(sco_trial,3)
        pc_trial_corr(i,j)=corr(sco_trial(i,:,j)',delta_2to3_z(:,j));
    end
end
[B,I]=sort(abs(mean(pc_trial_corr,2)),'descend');
d1=find(I==1);
d2=find(I==2);
d3=find(I==3);
a=pc_trial_corr([d1,d2,d3],:);
figure('Color','w');
temp_one=zscore(smoothdata(delta_2to3,'gaussian',30),0,1)';
shadedErrorBar(1:150,temp_one,{@nanmean,@(x) nanstd(x)./sqrt(size(x,1))},'lineprops',{'color',[243 143 60]./255,'LineWidth',2},'patchSaturation',0.5);
hold on
temp_one=zscore(smoothdata(reshape(sco(d1,:),[],trial_num),'gaussian',30),0,1)';
shadedErrorBar(1:150,temp_one,{@nanmean,@(x) nanstd(x)./sqrt(size(x,1))},'lineprops',{'color',[179 38 48]./255,'LineWidth',2},'patchSaturation',0.5);
hold on
temp_one=zscore(smoothdata(reshape(sco(d2,:),[],trial_num),'gaussian',30),0,1)';
shadedErrorBar(1:150,temp_one,{@nanmean,@(x) nanstd(x)./sqrt(size(x,1))},'lineprops',{'color',[37 115 166]./255,'LineWidth',2},'patchSaturation',0.5);
hold on
temp_one=zscore(smoothdata(reshape(sco(d3,:),[],trial_num),'gaussian',30),0,1)';
shadedErrorBar(1:150,temp_one,{@nanmean,@(x) nanstd(x)./sqrt(size(x,1))},'lineprops',{'color',[97,102,105]./255,'LineWidth',2},'patchSaturation',0.5);
hold on
plot([75 75],[-1.5 1.5],'Color',[226 30 37]./255,'LineStyle','--','LineWidth',3)
%legend('Delta power','PC1','PC2');
box off
ax=gca
ax.YAxis.Visible='off';
ylabel('(a.u.)');
ax.XAxis.TickDirection = 'out'; 
ax.XAxis.FontName = 'Arial';  
ax.XAxis.FontSize = 14;       
ax.LineWidth = 2;
ax.TickLength = [0.02, 0.02]; 

%% flow field
d1=d1;
d2=d2;

d1_range=[min(sco(d1,:)),max(sco(d1,:))];
d2_range=[min(sco(d2,:)),max(sco(d2,:))];

num_grid=15;
x_bottom=floor(d1_range(1));
x_upper=ceil(d1_range(2));
y_bottom=floor(d2_range(1));
y_upper=ceil(d2_range(2));
x_range_one_grid=(x_upper-x_bottom)/num_grid;
y_range_one_grid=(y_upper-y_bottom)/num_grid;
x=linspace(x_bottom,x_upper,num_grid+1);
y=linspace(y_bottom,y_upper,num_grid+1);
[xx, yy] = meshgrid(x, y);
xx=(xx(1:end-1,1:end-1)-x_bottom+x_range_one_grid/2)/x_range_one_grid;
yy=(yy(1:end-1,1:end-1)-y_bottom+y_range_one_grid/2)/y_range_one_grid;

%num cal
num_dots=zeros(num_grid,num_grid);
for i = 1:num_grid
    x_grid_range=[x(i),x(i+1)];
    for j = 1:num_grid
        y_grid_range=[y(j),y(j+1)];
        idx_in_grid=find(sco(d1,:)>=x_grid_range(1)&sco(d1,:)<x_grid_range(2)&sco(d2,:)>=y_grid_range(1)&sco(d2,:)<y_grid_range(2));
        if idx_in_grid
            num_dots(i,j)=length(idx_in_grid);
        end
    end
end


vecf=zeros(num_grid,num_grid,2);
for i=1:size(sco,2)-1
       if mod(i,time_num)~=0
           i0=floor((sco(d1,i)-x_bottom)/x_range_one_grid)+1;
           j0=floor((sco(d2,i)-y_bottom)/y_range_one_grid)+1;
           %if i0<=num_grid&j0<=num_grid
               vecf(i0,j0,1)=vecf(i0,j0,1)+(sco(d1,i+1)-sco(d1,i));
               vecf(i0,j0,2)=vecf(i0,j0,2)+(sco(d2,i+1)-sco(d2,i));
           %end
       end
end

min_dots=max(size(data,2)/num_grid.^2 , 3);
for i=1:num_grid
    for j=1:num_grid
        if num_dots(i,j)>=min_dots
            vecf(i,j,:)=vecf(i,j,:)/num_dots(i,j);
        else
            vecf(i,j,:)=0;
        end
    end
end
vecfn=vecf/max(abs(vecf(:))); %normalize


color=[27 149 212;237 177 32]./255;
color_label=[repmat(color(1,:),[time_num/2 1]);repmat(color(2,:),[time_num/2 1])];
figure('Color','w');
for i = 1:time_num
    tmp_color=color_label(i,:);
    scatter1=scatter((squeeze(sco_trial(d1,i,:))-x_bottom)/x_range_one_grid,(squeeze(sco_trial(d2,i,:))-y_bottom)/y_range_one_grid,50,repmat(tmp_color,[trial_num 1]),"filled");
    scatter1.MarkerFaceAlpha = .5;
    scatter1.MarkerEdgeAlpha = .5;
    hold on
end
ylim([-5 15]);
xlim([-1 13]);
xlabel('PC1','FontSize',18);
ylabel('PC2','FontSize',18);
ax=gca
ax.XAxis.TickDirection = 'out';  
ax.YAxis.TickDirection = 'out';
ax.XAxis.FontName = 'Arial';  
ax.XAxis.FontSize = 14;       
ax.YAxis.FontSize = 14;   
ax.LineWidth = 1.5;
ax.TickLength = [0, 0]; 
box on

% colorbar
x_range=[0.28*x_range_one_grid+x_bottom,12.2*x_range_one_grid+x_bottom];

num_grid=20;
grid_x=linspace(x_range(1),x_range(2),num_grid);
labels_all=repmat(labels_2to3_1,trial_num,1);

num_ratio=zeros(length(grid_x)-1,2);
for i = 1:length(grid_x)-1
    idx=find(sco(d1,:)>=grid_x(i)&sco(d1,:)<grid_x(i+1));
    label_tmp=labels_all(idx);
    num_ratio(i,1)=length(find(label_tmp==2));
    num_ratio(i,2)=length(find(label_tmp==3));
end

color_bar_wake=slanCM('Blues',max(num_ratio(:,1))+100); % WN
color_map_wake=color_bar_wake(num_ratio(:,1)+1,:);
figure;
colormap(color_map_wake);
colorbar;

figure;
colormap(color_bar_wake(1:end-100,:));
colorbar;

color_bar_nrem=zeros(max(num_ratio(:,2))+1,3);
color_bar_nrem(:,1)=linspace(226,247,max(num_ratio(:,2))+1)./255;
color_bar_nrem(:,2)=linspace(151,239,max(num_ratio(:,2))+1)./255;
color_bar_nrem(:,3)=linspace(0,223,max(num_ratio(:,2))+1)./255;
color_bar_nrem=flipud(color_bar_nrem);
color_map_nrem=color_bar_nrem(num_ratio(:,2)+1,:);
figure;
colormap(color_map_nrem);
colorbar;
figure;
colormap(color_bar_nrem);
colorbar;



% flow field
color=[27 149 212;237 177 32]./255;
color_label=[repmat(color(1,:),[time_num/2 1]);repmat(color(2,:),[time_num/2 1])];
figure('Color','w');
for i = 1:time_num
    tmp_color=color_label(i,:);
    scatter1=scatter((squeeze(sco_trial(d1,i,:))-x_bottom)/x_range_one_grid,(squeeze(sco_trial(d2,i,:))-y_bottom)/y_range_one_grid,50,repmat(tmp_color,[trial_num 1]),"filled");
    scatter1.MarkerFaceAlpha = .1;
    scatter1.MarkerEdgeAlpha = .1;
    hold on
end
plot((mean(sco_trial(d1,:,:),3)-x_bottom)/x_range_one_grid,(mean(sco_trial(d2,:,:),3)-y_bottom)/y_range_one_grid,'Color',[212 36 42]./255,'LineWidth',2);
hold on
quiver(xx',yy',vecfn(:,:,1),vecfn(:,:,2),1,'LineWidth',2,'Color','k');

ylim([-5 15]);
xlim([-1 13]);
xlabel('PC1','FontSize',18);
ylabel('PC2','FontSize',18);
ax=gca
ax.XAxis.TickDirection = 'out';  
ax.YAxis.TickDirection = 'out';
ax.XAxis.FontName = 'Arial';  
ax.XAxis.FontSize = 14;      
ax.YAxis.FontSize = 14;   
ax.LineWidth = 1.5;
ax.TickLength = [0, 0]; 
box on




