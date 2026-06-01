addpath(genpath('/Volumes/psw/code'));
%% prepare data
subj={'s133','s134','s137','s147','s148','s153'};
session_num1=[1,1,1,1,1,1];
session_num2=[5,5,5,5,5,5];


label_path='~\labels\';

epoch=10; % s
laser_time=120;%s
laser_bin_num=laser_time/epoch;
win=240;%s
win_bin_num=win/epoch;

%
label_trials_all=[];
trial_num_all=zeros(length(subj),1);
for i = 1:length(subj)
    tmp_session_num1=session_num1(i);
    tmp_session_num2=session_num2(i);
    for j = tmp_session_num1:tmp_session_num2
        load(strcat(label_path,subj{i},'\label',string(j)));
        load(strcat(label_path,subj{i},'\epocs',string(j)));
        %
        
        laser_onset_t=data5.epocs.Pu1_.onset(1:end)-1;% recording deviation
        laser_onset_t=laser_onset_t(1:end-1); % avoid not entire trials
        laser_onset_bin=round(laser_onset_t./epoch);
        
        trial_num=length(laser_onset_t);
        label_trials=[];%zeros(trial_num,2*win_bin_num);
        for m =1:trial_num
            label_trials(m,:)=labels(laser_onset_bin(m)-win_bin_num:laser_onset_bin(m)+laser_bin_num+win_bin_num);
        end
        trial_num_all(i)=trial_num_all(i)+trial_num;
        
        label_trials_all=[label_trials_all;label_trials];
    end
end
%% plot trial
tmp_label_trials_all=label_trials_all(:,1:end-1);
trial_num=size(tmp_label_trials_all,1);
figure('Color','white');
imagesc(tmp_label_trials_all);
xlabel('bins (10s/bin)');
ylabel('Trials','FontSize',24);

set(gca,'FontSize',20)
cmap=[[12 172 228]./255;[140 145 146]./255;[255 188 0]./255];
colormap(cmap);
colorbar;
hold on
plot([25 25],[0 trial_num],'b--','LineWidth',2);
hold on
plot([36 36],[0 trial_num],'b--','LineWidth',2);


%% plot states
wake_p=zeros(length(subj),2*win_bin_num+laser_bin_num);
nrem_p=zeros(length(subj),2*win_bin_num+laser_bin_num);
rem_p=zeros(length(subj),2*win_bin_num+laser_bin_num);
for i = 1:length(subj)
    tmp_label_trials_all=label_trials_all(sum(trial_num_all(1:i))-trial_num_all(i)+1:sum(trial_num_all(1:i)),1:end-1);
    trial_num=size(tmp_label_trials_all,1);
    wake_p(i,:)=sum(tmp_label_trials_all==2,1)./trial_num;
    nrem_p(i,:)=sum(tmp_label_trials_all==3,1)./trial_num;
    rem_p(i,:)=sum(tmp_label_trials_all==1,1)./trial_num;
end

figure;
shadedErrorBar(1:60,wake_p*100,{@nanmean,@(x) nanstd(x)./sqrt(size(x,1))},'lineprops',{'color',[140 145 146]./255,'LineWidth',2},'patchSaturation',0.5);
hold on
shadedErrorBar(1:60,nrem_p*100,{@nanmean,@(x) nanstd(x)./sqrt(size(x,1))},'lineprops',{'color',[255 188 0]./255,'LineWidth',2},'patchSaturation',0.5);
hold on
shadedErrorBar(1:60,rem_p*100,{@nanmean,@(x) nanstd(x)./sqrt(size(x,1))},'lineprops',{'color',[12 172 228]./255,'LineWidth',2},'patchSaturation',0.5);
hold on
plot([25 25],[0 100],'b--','LineWidth',2);
hold on
plot([36 36],[0 100],'b--','LineWidth',2);
ylim([0 100]);


figure;
plot(1:60,wake_p*100,'b');
hold on
plot(1:60,nrem_p*100,'r');
hold on
plot(1:60,rem_p*100,'g');
hold on
plot([25 25],[0 100],'b--','LineWidth',2);
hold on
plot([36 36],[0 100],'b--','LineWidth',2);
ylim([0 100]);
%% transition probability
trial_num=size(label_trials_all,1);
time_bin_num=size(label_trials_all,2)-1;

state1=3;
state2=2;


p_bin=zeros(1,time_bin_num);
for i = 1:time_bin_num
    idx_state1=find(label_trials_all(:,i)==state1);
    if idx_state1
        tmp_trials_state1=label_trials_all(idx_state1,:);
        idx_to_state2=find(tmp_trials_state1(:,i+1)==state2);
        p_bin(i)=length(idx_to_state2)/length(idx_state1);
    end
end

p_bin_deflat=reshape(p_bin,[6 10]);
p_bin_average=mean(p_bin_deflat,1);
figure;
bar(p_bin_average)
hold on
plot([4.5 4.5],[0 max(p_bin_average)],'b--','LineWidth',2);
hold on
plot([6.5 6.5],[0 max(p_bin_average)],'b--','LineWidth',2);
hold on
plot([0 11],[mean(p_bin_average(1:4)),mean(p_bin_average(1:4))],'k--','LineWidth',2);
ylabel('trans probability');
xlabel('60s/bin');
title(strcat(string(state1),'to',string(state2)));
%ylim([0 0.25])


