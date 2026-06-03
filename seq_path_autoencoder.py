# autoencoder modelling of WN sequence path by top PLS dimensions same for NW
# prepare data

import h5py
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


SUBJ = 's74'
DATA_FILE = f'~/WN_{SUBJ}_proj_4.mat'
MODEL_FILE = f'~/{SUBJ}_WN_bridge_path_AEmodel.pth'
TRAJECTORY_POINTS_FILE = '~/WN_all_trajectory_points.npy'


SEED=1234
import random
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) # for GPU training
torch.cuda.manual_seed_all(SEED) # for multi-GPU training

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

a=np.array([1,2,3,4,5,6,7,21,26,30,32,35,36,40,60,65,71])-1 # for better visualization, but same conclusion with other choice
b = np.arange(0, 86)
result = b[~np.isin(b, a)]

print(result)


def resolve_path(file_path):
    return str(Path(file_path).expanduser())


def load_data(file_path):
    with h5py.File(resolve_path(file_path), 'r') as file:
        traces_trials = file['X_proj_trials'][:,:,:]
        all_trials = traces_trials
        train_trials = traces_trials[result,:,:]
        test_trials = traces_trials[a,:,:]
    data_dic = {
        'all_trials': all_trials,
        'train_trials': train_trials,
        'test_trials': test_trials,
    }
    return data_dic

data_dic = load_data(DATA_FILE)
len_trials = np.shape(data_dic['train_trials'])[1]
num_trials= np.shape(data_dic['train_trials'])[0]
print(data_dic['train_trials'].shape)
cell_nums = data_dic['train_trials'].shape[-1]
print(f'number of cells {cell_nums}')

class DataTraces(Dataset):
    """Neural traces dataset."""

    def __init__(self, traces_trials_t,traces_trials_tp):
        """
        Args:
            data (Data): Object that has two attributes, X and y,
                         where Xt is a matrix of neural activity and
                         Xtp1 is a matrix of next time-point neural activity
        """
        self.Xt = torch.tensor(traces_trials_t, dtype=torch.float32)
        self.Xtp1 = torch.tensor(traces_trials_tp, dtype=torch.float32)
    def __len__(self):
        return len(self.Xt)
    def __getitem__(self, idx):
        return self.Xt[idx], self.Xtp1[idx]

# training trials
a=data_dic['train_trials']
num_train_trials=a.shape[0]
a=a.reshape([-1,cell_nums])
scaler = StandardScaler()
a=scaler.fit_transform(a)
a=a.reshape([num_train_trials,len_trials,cell_nums])
tmp_traces_trials_t = np.zeros([num_train_trials,a.shape[1] - 1,a.shape[2]])
tmp_traces_trials_tp = np.zeros([num_train_trials,a.shape[1] - 1,a.shape[2]])
for i in range(num_train_trials):
    scaler = StandardScaler()
    tmp_data=a[i,:,:]
    tmp_data = scaler.fit_transform(tmp_data)
    tmp_traces_trials_t[i,:,:] = tmp_data[0:-1, :]
    tmp_traces_trials_tp[i,:,:] = tmp_data[1:, :]


traces_trials_t=tmp_traces_trials_t.reshape([-1,cell_nums])
traces_trials_tp=tmp_traces_trials_tp.reshape([-1,cell_nums])
dataset = DataTraces(traces_trials_t,traces_trials_tp)


# test trials
a=data_dic['test_trials']
num_test_trials = a.shape[0]
a=a.reshape([-1,cell_nums])
scaler = StandardScaler()
a=scaler.fit_transform(a)
a=a.reshape([num_test_trials,len_trials,cell_nums])
tmp_traces_trials_t = np.zeros([ num_test_trials,a.shape[1] - 1, a.shape[2]])
tmp_traces_trials_tp = np.zeros([ num_test_trials,a.shape[1] - 1, a.shape[2]])
for i in range(num_test_trials):
    scaler = StandardScaler()
    tmp_data = a[i, :, :]
    tmp_data = scaler.fit_transform(tmp_data)
    tmp_traces_trials_t[i, :, :] = tmp_data[0:-1, :]
    tmp_traces_trials_tp[i, :, :] = tmp_data[1:, :]

test_trials_t = tmp_traces_trials_t.reshape([-1, cell_nums])
test_trials_tp = tmp_traces_trials_tp.reshape([-1, cell_nums])


scaler = StandardScaler()
averaged_z=scaler.fit_transform(np.mean(data_dic['all_trials'],0))



# model training
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) # for GPU training
torch.cuda.manual_seed_all(SEED) # for multi-GPU training
class autoencoder(nn.Module):
    def __init__(self, num_cells, latent_dim):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_cells, 192),
            nn.Tanh(),
            nn.Linear(192, 96),
            nn.Tanh(),
            nn.Linear(96, 48),
            nn.Tanh(),
            nn.Linear(48, latent_dim),
            nn.Tanh(),
            )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 96),
            nn.Tanh(),
            nn.Linear(96, 192),
            nn.Tanh(),
            nn.Linear(192, num_cells),
            )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def worker_init_fn(worker_id):
    random.seed(SEED + worker_id)

#seq
learning_rate = 5e-5
latent_dim = 2
model = autoencoder(cell_nums, latent_dim=latent_dim).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-6)
batch_size = 300
g = torch.Generator()
g.manual_seed(SEED)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,worker_init_fn=worker_init_fn,generator = g)

# load model
# create a same structure before load
model = autoencoder(cell_nums, latent_dim=latent_dim).cuda()
model.load_state_dict(torch.load(resolve_path(MODEL_FILE)))
model.eval()

num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for batch_id, (xt, xtp1) in enumerate(dataloader):
        if torch.cuda.is_available():
            xt   = xt.cuda()
            xtp1 = xtp1.cuda()
        # print(f'Xt : {xt[:2,:2]}')
        # print(f'Xt+1 : {xtp1[:2,:2]}')
        # ===================forward=====================
        output = model(xt)
        loss = criterion(output, xtp1)
        total_loss += loss.item()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.7f}'
          .format(epoch + 1, num_epochs, total_loss/len(dataloader.dataset)))

#torch.save(model.state_dict(), resolve_path(MODEL_FILE))

output = model(torch.from_numpy(traces_trials_t).to(dtype=torch.float).cuda()) # gt
output_latent = model.encoder(torch.from_numpy(traces_trials_t).to(dtype=torch.float).cuda())
output_latent=output_latent.cpu().detach().numpy()
output_pred=model(torch.from_numpy(test_trials_t).to(dtype=torch.float).cuda()) # test
output_pred=output_pred.cpu().detach().numpy()
output_latent_tp1=model.encoder(torch.from_numpy(test_trials_t).to(dtype=torch.float).cuda()) # test
output_latent_tp1=output_latent_tp1.cpu().detach().numpy()
output_latent_tp1_trials=output_latent_tp1[:,:].reshape([-1,len_trials-1,latent_dim])
output_latent_tp1_average=np.mean(output_latent_tp1_trials,0)


# plot data predicted
import matplotlib
import matplotlib.pyplot as plt
row = 10
col = 5
fig, axs = plt.subplots(row, col, figsize=(40, 4))
cell_idxs = range(273)

for r in range(row):
    for c in range(col):
        cellidx = cell_idxs[r*col + c]
        axs[r,c].plot(output_pred[1:400, cellidx], label='predicted')
        axs[r,c].plot(test_trials_t[1:400, cellidx], label='orig')
        axs[r,c].legend()

        axs[r,c].set_title(f'cell {cellidx}')
plt.show(block=True)

# test
a = np.reshape(output_pred[:,:],(-1,len_trials-1,cell_nums))
a = np.mean(a,0)
plt.figure()
plt.imshow(a.T,aspect = "auto",cmap = 'jet',vmin=-0.5,vmax=0.5,interpolation = 'none')
plt.title('test pred',fontsize= 20)

b = np.reshape(test_trials_tp[:,:],(-1,len_trials-1,cell_nums))
b = np.mean(b,0)
plt.figure()
plt.imshow(b.T,aspect = "auto",cmap = 'jet',vmin=-0.5,vmax=0.5,interpolation = 'none')
plt.title('test gt',fontsize= 20)
#plt.show(block=True)
# train
c = np.reshape(output[:,:].cpu().detach().numpy(),(-1,len_trials-1,cell_nums))
c = np.mean(c,0)
plt.figure()
plt.imshow(c.T,aspect = "auto",cmap = 'jet',vmin=-0.5,vmax=0.5,interpolation = 'none')
plt.title('train pred',fontsize= 20)

d = np.reshape(traces_trials_t[:,:],(-1,len_trials-1,cell_nums))
d = np.mean(d,0)
plt.figure()
plt.imshow(d.T,aspect = "auto",cmap = 'jet',vmin=-0.5,vmax=0.5,interpolation = 'none')
plt.title('train gt',fontsize= 20)
plt.show(block=True)



#########################t0 evolution
num_epochs = 5 #140
for epoch in range(num_epochs):
    total_loss = 0
    for batch_id, (xt, xtp1) in enumerate(dataloader):
        if torch.cuda.is_available():
            xt   = xt.cuda()
            xtp1 = xtp1.cuda()
        # print(f'Xt : {xt[:2,:2]}')
        # print(f'Xt+1 : {xtp1[:2,:2]}')
        # ===================forward=====================
        output = model(xt)
        loss = criterion(output, xtp1)
        total_loss += loss.item()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.7f}'
          .format(epoch + 1, num_epochs, total_loss/len(dataloader.dataset)))

output = model(torch.from_numpy(traces_trials_t).to(dtype=torch.float).cuda()) # gt
output_latent = model.encoder(torch.from_numpy(traces_trials_t).to(dtype=torch.float).cuda())
output_latent=output_latent.cpu().detach().numpy()
output_pred=model(torch.from_numpy(test_trials_t).to(dtype=torch.float).cuda()) # test
output_pred=output_pred.cpu().detach().numpy()
output_latent_tp1=model.encoder(torch.from_numpy(test_trials_t).to(dtype=torch.float).cuda()) # test
output_latent_tp1=output_latent_tp1.cpu().detach().numpy()
output_latent_tp1_trials=output_latent_tp1[:,:].reshape([-1,len_trials-1,latent_dim])

import matplotlib
import matplotlib.pyplot as plt
iter_start=91
test_trials_t_trial=test_trials_t.reshape(num_test_trials,len_trials-1,cell_nums)
output_latent_iter=np.zeros([num_test_trials,len_trials-iter_start,latent_dim])
output_freq_iter=np.zeros([num_test_trials,len_trials-iter_start,cell_nums])
for i in range(num_test_trials):
    tmp_freq_t = test_trials_t_trial[i, iter_start, :]
    for j in range(len_trials-iter_start):
        tmp_latent_tp = model.encoder(torch.from_numpy(tmp_freq_t).to(dtype=torch.float).cuda())
        tmp_latent_tp=tmp_latent_tp.cpu().detach().numpy()
        output_latent_iter[i,j,:]=tmp_latent_tp
        tmp_freq_tp = model.decoder(torch.from_numpy(tmp_latent_tp).to(dtype=torch.float).cuda())
        tmp_freq_tp = tmp_freq_tp.cpu().detach().numpy()
        output_freq_iter[i,j,:]=tmp_freq_tp
        tmp_freq_t = tmp_freq_tp



plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.mean(output_freq_iter[:,:,:],0).T,aspect = "auto",cmap = 'jet',vmin=-0.5,vmax=0.5,interpolation = 'none')
#plt.imshow(output_freq_iter[2,:,:].T,aspect = "auto",cmap = 'jet',vmin=-0.5,vmax=0.5,interpolation = 'none')
plt.title('test iter',fontsize= 20)
plt.subplot(1,2,2)
plt.imshow(np.mean(test_trials_t_trial,0).T,aspect = "auto",cmap = 'jet',vmin=-0.5,vmax=0.5,interpolation = 'none')
plt.title('test gt',fontsize= 20)
plt.show(block=True)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(output_freq_iter[4,:,:].T,aspect = "auto",cmap = 'jet',vmin=-0.5,vmax=0.5,interpolation = 'none')
plt.title('test iter',fontsize= 20)
plt.subplot(1,2,2)
plt.imshow(np.mean(test_trials_t_trial,0).T,aspect = "auto",cmap = 'jet',vmin=-0.5,vmax=0.5,interpolation = 'none')
plt.title('test gt',fontsize= 20)
plt.show(block=True)
############################PLOT latent dimensions
import matplotlib
import matplotlib.pyplot as plt
output_latent_trial=output_latent[:,:].reshape(-1, len_trials-1,latent_dim)
latent_averaged=model.encoder(torch.from_numpy(averaged_z).to(dtype=torch.float).cuda())
latent_averaged=latent_averaged.cpu().detach().numpy()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# time
cmap = plt.cm.jet
colors = [cmap(i) for i in np.linspace(0, 1, len_trials)]
fig, ax = plt.subplots()
ax.scatter(latent_averaged[:,0],latent_averaged[:,1], c='r', marker='o')
for i in range(0,len_trials-1):
    ax.scatter(output_latent_trial[:, i, 0], output_latent_trial[:, i, 1], c=colors[i], marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show(block=True)

############################flow field
num_pixel=30
x_bottom=-1
x_upper=1
y_bottom=-1
y_upper=1.1
x_range_one_grid=(x_upper-x_bottom)/num_pixel
y_range_one_grid=(y_upper-y_bottom)/num_pixel
x=np.linspace(x_bottom, x_upper, num=num_pixel, endpoint=True)
y=np.linspace(y_bottom, y_upper, num=num_pixel, endpoint=True)
xx, yy = np.meshgrid(x, y)
xx_target=np.zeros(np.shape(xx))
yy_target=np.zeros(np.shape(yy))
for i in range(num_pixel):
    for j in range(num_pixel):
            tmp_latent=np.zeros([1,2])
            tmp_latent[0, 0] = xx[i, j]
            tmp_latent[0, 1] = yy[i, j]
            tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
            tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
            tmp_latent_tp= model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
            tmp_latent_tp= tmp_latent_tp.cpu().detach().numpy()
            xx_target[i, j]=tmp_latent_tp[0, 0]-tmp_latent[0,0]
            yy_target[i, j] = tmp_latent_tp[0, 1]-tmp_latent[0,1]



fig, ax = plt.subplots()
ax.quiver(xx,yy,xx_target,yy_target,color=[21/255,109/255,183/255],scale=4)
from scipy.ndimage import gaussian_filter
latent_averaged_s = gaussian_filter(latent_averaged,sigma=0.5)
cmap = plt.cm.jet
colors_labels = [cmap(i) for i in np.linspace(0, 1, 2)]
colors = [colors_labels[i] for i in np.tile([0],150)]+[colors_labels[i] for i in np.tile([1],150)]
for i in range(0,len_trials-1):
    plt.plot(latent_averaged_s[i:i+2, 0], latent_averaged_s[i:i+2, 1], c=colors[i],lw=4)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
plt.show(block=True)





# test trials
for t in range(num_test_trials): # t=9
    start_point=output_latent_tp1_trials[t,0,:]
    trajectory=np.zeros([len_trials-1,2])
    trajectory[0,:]=start_point
    for i in range(len_trials-2):
        tmp_latent = trajectory[i, :]
        tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
        tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
        tmp_latent_tp = model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
        trajectory[i+1,:] = tmp_latent_tp.cpu().detach().numpy()

    start_point=output_latent_tp1_trials[t,90,:]
    trajectory2=np.zeros([len_trials,2])
    trajectory2[0,:]=start_point
    for i in range(len_trials-1):
        tmp_latent = trajectory2[i, :]
        tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
        tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
        tmp_latent_tp = model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
        trajectory2[i+1,:] = tmp_latent_tp.cpu().detach().numpy()


    fig, ax = plt.subplots()
    ax.quiver(xx,yy,xx_target,yy_target,scale=4)
    plt.plot(output_latent_tp1_trials[t,:,0], output_latent_tp1_trials[t,:,1], c='r')
    cmap = plt.cm.RdYlBu
    colors = [cmap(i) for i in np.linspace(0, 1, 300)]
    for i in range(0,len_trials-1):
        ax.scatter(trajectory[i, 0], trajectory[i, 1], c=colors[i], marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
    #plt.show(block=True)

    colors = [cmap(i) for i in np.linspace(0, 1, len_trials)]
    for i in range(0,len_trials-1):
        ax.scatter(trajectory2[i, 0], trajectory2[i, 1], c=colors[i], marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

    plt.xlim((-0.8,0.8))
    plt.ylim((-1, 1))
    plt.show(block=True)


# example plot
t=np.array([8,11,12])  #8 11 12

fig, ax = plt.subplots()
ax.quiver(xx, yy, xx_target, yy_target, scale=4)
plt.plot(output_latent_tp1_trials[8, :, 0], output_latent_tp1_trials[8, :, 1], c='r')
plt.plot(output_latent_tp1_trials[11, :, 0], output_latent_tp1_trials[11, :, 1], c='r')
plt.plot(output_latent_tp1_trials[12, :, 0], output_latent_tp1_trials[12, :, 1], c='r')
plt.xlim((-0.8, 1))
plt.ylim((-1, 1.1))
plt.show(block=True)


# example plot with evolve
t=8 #1 7 8 11 12 16

evolve_num=15
start_point = output_latent_tp1_trials[t, 0, :]
trajectory = np.zeros([evolve_num, 2])
trajectory[0, :] = start_point
for i in range(evolve_num-1):
    tmp_latent = trajectory[i, :]
    tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
    tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
    tmp_latent_tp = model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
    trajectory[i + 1, :] = tmp_latent_tp.cpu().detach().numpy()

evolve_num2=100
start_point = output_latent_tp1_trials[t, 90, :]
trajectory2 = np.zeros([evolve_num2, 2])
trajectory2[0, :] = start_point
for i in range(evolve_num2-1):
    tmp_latent = trajectory2[i, :]
    tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
    tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
    tmp_latent_tp = model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
    trajectory2[i + 1, :] = tmp_latent_tp.cpu().detach().numpy()

fig, ax = plt.subplots()
ax.quiver(xx, yy, xx_target, yy_target, scale=4)
plt.plot(output_latent_tp1_trials[t, :, 0], output_latent_tp1_trials[t, :, 1], c='r')
cmap = plt.cm.RdYlBu
colors = [cmap(i) for i in np.linspace(0, 1, evolve_num2)]
for i in range(0, evolve_num2-1):
    ax.scatter(trajectory2[i, 0], trajectory2[i, 1], c=colors[i], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')


colors = [cmap(i) for i in np.linspace(0, 1, evolve_num)]
for i in range(0, evolve_num-1):
    ax.scatter(trajectory[i, 0], trajectory[i, 1], c=colors[i], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

plt.xlim((-0.8, 1))
plt.ylim((-1, 1.1))
plt.show(block=True)


#### random start point and evolve 10 steps
def MSE(a, b):
    return np.mean(np.sum((a - b) ** 2,1))

import numpy as np
cosine_sim_evolve_all=np.zeros([output_latent_tp1_trials.shape[0],1])
cosine_sim_average_all=np.zeros([output_latent_tp1_trials.shape[0],1])
dist_evolve_all=np.zeros([output_latent_tp1_trials.shape[0],1])
dist_average_all=np.zeros([output_latent_tp1_trials.shape[0],1])
for k in range(output_latent_tp1_trials.shape[0]):
    output_latent_tp1_average_seq=output_latent_tp1_average[90:210,:] # average path during the sequence period
    tmp_trial_trajectory=output_latent_tp1_trials[k,90:210,:] # real data from individual trials

    np.random.seed(42)
    random_num=30
    local_evolve_num=10 #cycle num
    indices = np.random.choice(tmp_trial_trajectory.shape[0]-local_evolve_num, random_num, replace=False)  # generate non-repeating indices

    tmp_trial_trajectory_r=np.zeros([random_num,local_evolve_num,2])
    output_latent_tp1_average_seq_r=np.zeros([random_num,local_evolve_num,2])
    for i in range(random_num):
        tmp_trial_trajectory_r[i,:,:]=tmp_trial_trajectory[indices[i]+1:indices[i]+1+local_evolve_num,:] # sampled real data
        output_latent_tp1_average_seq_r[i,:,:]=output_latent_tp1_average_seq[indices[i]+1:indices[i]+1+local_evolve_num,:] # sampled average path


    start_point = tmp_trial_trajectory[indices]
    evolve_local_trajectory = np.zeros([random_num,local_evolve_num+1,2])
    evolve_local_trajectory[:,0,:]=start_point
    for i in range(start_point.shape[0]):
        for j in range(local_evolve_num):
            tmp_latent = evolve_local_trajectory[i,j,:]
            tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
            tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
            tmp_latent_tp = model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
            evolve_local_trajectory[i, j+1, :] = tmp_latent_tp.cpu().detach().numpy() # evolved data from sampled points
    evolve_local_trajectory=evolve_local_trajectory[:,1:,:]

    #cosine similarity
    tangent_real=tmp_trial_trajectory_r[:,-1,:]-tmp_trial_trajectory_r[:,0,:]
    tangent_average=output_latent_tp1_average_seq_r[:,-1,:]-output_latent_tp1_average_seq_r[:,0,:]
    tangent_evolve=evolve_local_trajectory[:,-1,:]-evolve_local_trajectory[:,0,:]

    dist_average=np.zeros([random_num,1])
    dist_evolve=np.zeros([random_num,1])
    cosine_sim_average=np.zeros([random_num,1])
    cosine_sim_evolve=np.zeros([random_num,1])
    for i in range(random_num):
        tmp_real_trajectory=tmp_trial_trajectory_r[i,:,:]
        tmp_average_trajectory=output_latent_tp1_average_seq_r[i,:,:]
        tmp_evolve_trajectory=evolve_local_trajectory[i,:,:]
        dist_average[i]=MSE(tmp_real_trajectory,tmp_average_trajectory)
        dist_evolve[i]=MSE(tmp_real_trajectory,tmp_evolve_trajectory)

        tmp_real=tangent_real[i,:]
        tmp_average=tangent_average[i,:]
        tmp_evolve=tangent_evolve[i,:]
        cosine_sim_average[i] = np.dot(tmp_real, tmp_average) / (np.linalg.norm(tmp_real) * np.linalg.norm(tmp_average))
        cosine_sim_evolve[i] = np.dot(tmp_real, tmp_evolve) / (np.linalg.norm(tmp_real) * np.linalg.norm(tmp_evolve))

    cosine_sim_evolve_all[k]=np.mean(cosine_sim_evolve)
    cosine_sim_average_all[k]= np.mean(cosine_sim_average)
    dist_evolve_all[k]=np.mean(dist_evolve)
    dist_average_all[k]= np.mean(dist_average)




fig, ax = plt.subplots()
plt.plot(tmp_trial_trajectory_r[12,:,0], tmp_trial_trajectory_r[12,:,1], c='r')
plt.plot(output_latent_tp1_average_seq_r[12,:,0], output_latent_tp1_average_seq_r[12,:,1], c='b')
plt.plot(evolve_local_trajectory[12,:,0], evolve_local_trajectory[12,:,1], c='g')
plt.show(block=True)

fig, ax = plt.subplots()
plt.scatter(dist_average_all,dist_evolve_all,c='b')
#plt.scatter(cosine_sim_average_all,cosine_sim_evolve_all,c='r')
plt.plot([0,0.4],[0,0.4],c='b')
# plt.xlim((0, 1))
# plt.ylim((0, 1))
plt.xlabel('averaged')
plt.ylabel('evolve')
plt.show(block=True)

import pandas as pd
df=pd.DataFrame(dist_evolve_all)
df.to_clipboard(index=False,header=False)

import matplotlib.pyplot as plt
categories = ['average', 'evolve']
# values = [np.mean(cosine_sim_average_all),np.mean(cosine_sim_evolve_all)]
values = [np.mean(dist_average_all),np.mean(dist_evolve_all)]
plt.bar(categories, values, color='skyblue', width=0.8)
# plt.ylim((0, 1))
plt.title('Simple Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show(block=True)
####################################landscape
from scipy.ndimage import gaussian_filter
scalars = np.sqrt(xx_target**2 + yy_target**2)
scalars=gaussian_filter(scalars,sigma=1.5)
scalars=np.log2(scalars)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx, yy, scalars, cmap='RdYlBu')
fig.colorbar(surf)
ax.set_title("3D Surface Plot of Flow Field")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_zlabel("Flow field magnitude (Z)")
#ax.set_axis_off()
plt.show(block=True)




#prob. landscape
#all_trajectory_points=np.load(resolve_path(TRAJECTORY_POINTS_FILE))
num_pixel=100
x_bottom=-1.1
x_upper=1.2
y_bottom=-1.1
y_upper=1.2
x_range_one_grid=(x_upper-x_bottom)/num_pixel
y_range_one_grid=(y_upper-y_bottom)/num_pixel
x=np.linspace(x_bottom, x_upper, num=num_pixel, endpoint=True)
y=np.linspace(y_bottom, y_upper, num=num_pixel, endpoint=True)



import random
random.seed(42)
random_num=10000
random_x = [random.uniform(x_bottom, x_upper) for _ in range(random_num)]
random_y = [random.uniform(y_bottom, y_upper) for _ in range(random_num)]
evolve_num=100
all_trajectory_points=np.zeros([random_num,evolve_num,2])

for i in range(0,random_num):
    start = np.zeros([1, 2])
    start[0, 0] = random_x[i]
    start[0, 1] = random_y[i]
    tmp_trajectory = np.zeros([100, 2])
    tmp_trajectory[0, :] = start
    for j in range(0,evolve_num-1):
        tmp_latent = tmp_trajectory[j, :]
        tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
        tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
        tmp_latent_tp = model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
        tmp_trajectory[j + 1, :] = tmp_latent_tp.cpu().detach().numpy()
    all_trajectory_points[i,:,:]=tmp_trajectory
#np.save(resolve_path(TRAJECTORY_POINTS_FILE), all_trajectory_points)


all_flat=all_trajectory_points.reshape(-1,2)
prob_scale=np.zeros([num_pixel-1,num_pixel-1])
for i in range(0,num_pixel-1):
    tmp_x = x[i]
    tmp_x1= x[i+1]
    for j in range(0,num_pixel-1):
        tmp_y=y[j]
        tmp_y1=y[j+1]
        tmp_num=len(np.where((all_flat[:,0]>=tmp_x) & (all_flat[:,0]<tmp_x1) & (all_flat[:,1]>=tmp_y) &(all_flat[:,1]<tmp_y1))[0])
        prob_scale[i,j]=tmp_num/(random_num*evolve_num)


x_prob=x[0:num_pixel-1]+x_range_one_grid/2
y_prob=x[0:num_pixel-1]+y_range_one_grid/2
xx_prob, yy_prob = np.meshgrid(x_prob, y_prob)

def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap

from scipy.ndimage import gaussian_filter
prob_scale1=gaussian_filter(prob_scale,sigma=4)
prob_scale1=np.log(prob_scale1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(xx_prob, yy_prob, prob_scale1, cmap=truncate_colormap('RdBu_r',minval=0.1, maxval=0.9),alpha=0.8)
fig.colorbar(surf)
ax.set_title("3D Surface Plot of Flow Field")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_zlabel("Flow field magnitude (Z)")
ax.set_axis_off()
plt.show(block=True)


