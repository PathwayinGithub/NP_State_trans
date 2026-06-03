% a=presence(temp_spike_freq,temp_labels);

function presence_ratio=presence(spikes,labels)
% spikes:num_neurons*spikes(t) labels: length(t)*1
presence_ratio=zeros(size(spikes,1),3);
for i = [1,2,3]
    idx_state=find(labels==i);
    temp_spike_state=spikes(:,idx_state);
    if size(temp_spike_state,1)~=0
        bins=round(linspace(1,size(temp_spike_state,2),100));
        
        tmp_spike=zeros(size(spikes,1),99);
        for j = 1:length(bins)-1
            tmp_spike(:,j)=sum(temp_spike_state(:,bins(j):bins(j+1)-1),2);
        end
        
        presense_c=sum(tmp_spike>=1,2);
        presence_ratio(:,i)=presense_c./100;
    end
end

end


% figure;
% cdfplot(presence_ratio(:,1));
% histogram(presence_ratio);
% 
% sum(presence_ratio>=0.7,2);
% sum(ans>0)