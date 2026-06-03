%table_states_eegPSD=stable_states(labels_eegPSD,2.5);
% return table_states(n,5): col1:states(5==NREM) col2:state length col3:bin length
% col4:start index col5:end index
function table_states=stable_states(labels,bin)
table_states=[];
labels(labels==3)=5;
labels_diff=diff(labels);
idx_boundary=find(labels_diff~=0);
state_length=diff(idx_boundary);
for i = 1:length(idx_boundary)-1
    table_states(i,4)=idx_boundary(i)+1;
    table_states(i,5)=idx_boundary(i+1);
    table_states(i,1)=labels(idx_boundary(i)+1);
end
table_states(:,2)=state_length;
table_states(:,3)=bin;

tmp_table=[];
tmp_table(:,1)=labels(idx_boundary(1)-1);
tmp_table(:,2)=idx_boundary(1);
tmp_table(:,3)=bin;
tmp_table(:,4)=1;
tmp_table(:,5)=idx_boundary(1);
table_states=[tmp_table;table_states];

tmp_table=[];
tmp_table(:,1)=labels(idx_boundary(end)+1);
tmp_table(:,2)=length(labels)-idx_boundary(end);
tmp_table(:,3)=bin;
tmp_table(:,4)=idx_boundary(end)+1;
tmp_table(:,5)=length(labels);
table_states=[table_states;tmp_table];
% table_states(table_states(:,1)==5,1)=3;
end