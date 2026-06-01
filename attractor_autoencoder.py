# autoencoder modelling for attractor manifolds by most correlated PCs with delta power curve

# prepare data

import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


SEED=1234
import random
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


def load_data(file_path):
    with h5py.File(file_path, 'r') as file:
        traces_trials = file['sxx_pc12_spike_trials'][:,:,:]
        all_trials = traces_trials
        train_trials = traces_trials[0:18,:,:] # according to trial num
        test_trials = traces_trials[18:,:,:]
    data_dic = {
        'all_trials': all_trials,
        'train_trials': train_trials,
        'test_trials': test_trials,
    }
    return data_dic

data_dic = load_data('~/sxx_pc12_spike_trials_WN.mat')
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
torch.cuda.manual_seed(SEED) # 适用于显卡训练
torch.cuda.manual_seed_all(SEED) # 适用于多显卡训练
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


#attractor
learning_rate = 1e-4
latent_dim = 2
model = autoencoder(cell_nums, latent_dim=latent_dim).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-6)
batch_size = 60
g = torch.Generator()
g.manual_seed(SEED)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,worker_init_fn=worker_init_fn,generator = g)

# load model
# create a same model structure before load
model = autoencoder(cell_nums, latent_dim=latent_dim).cuda()
model.load_state_dict(torch.load('~/sxx_WN_attractor_AEmodel.pth'))
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

#save model
#torch.save(model.state_dict(), '~/sxx_WN_attractor_AEmodel.pth')


output = model(torch.from_numpy(traces_trials_t).to(dtype=torch.float).cuda()) # gt
output_latent = model.encoder(torch.from_numpy(traces_trials_t).to(dtype=torch.float).cuda())
output_latent=output_latent.cpu().detach().numpy()
output_pred=model(torch.from_numpy(test_trials_t).to(dtype=torch.float).cuda()) # test
output_pred=output_pred.cpu().detach().numpy()
output_latent_tp1=model.encoder(torch.from_numpy(test_trials_t).to(dtype=torch.float).cuda()) # test
output_latent_tp1=output_latent_tp1.cpu().detach().numpy()
output_latent_tp1_trials=output_latent_tp1[:,:].reshape([-1,len_trials-1,latent_dim])
#np.save('~/WN_attractor_output_pred_shuffle.npy', output_pred)



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



############################PLOT latent dimensions
import matplotlib
import matplotlib.pyplot as plt
output_latent_trial=output_latent[:,:].reshape(-1, len_trials-1,latent_dim)
latent_averaged=model.encoder(torch.from_numpy(averaged_z).to(dtype=torch.float).cuda())
latent_averaged=latent_averaged.cpu().detach().numpy()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

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



num_pixel=30
x_bottom=-0.8
x_upper=0.05  #0.4
y_bottom=-0.5 #-0.7
y_upper=0.8
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
ax.quiver(xx,yy,xx_target,yy_target,color=[21/255,109/255,183/255],scale=8)
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


#random start points evolved trajectories
start_point=np.zeros([6,2])
start_point[0,:]=[-0.7,-0.4]
start_point[1,:]=[-0.5,-0.45]
start_point[2,:]=[-0.25,-0.3]
start_point[3,:]=[-0.65,0.45]
start_point[4,:]=[-0.4,0.7]
start_point[5,:]=[-0.1,0.4]

trajectory=np.zeros([6,len_trials,2])
for j in range(6):
    trajectory[j,0,:]=start_point[j,:]
    for i in range(len_trials-1):
        tmp_latent = trajectory[j,i, :]
        tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
        tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
        tmp_latent_tp = model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
        trajectory[j,i+1,:] = tmp_latent_tp.cpu().detach().numpy()

fig, ax = plt.subplots()
ax.quiver(xx,yy,xx_target,yy_target,color='black',scale=5)
#plt.plot(latent_averaged[:, 0], latent_averaged[:, 1], c='r')
color1 = np.array([27, 149, 212]) / 255
color2 = np.array([237, 177, 32]) / 255
colors = np.zeros((150, 3))
colors[:75] = color1
colors[75:] = color2
for i in range(0,len_trials-1):
    ax.scatter(output_latent_trial[:, i, 0], output_latent_trial[:, i, 1], c=colors[i], marker='o',alpha=0.2)
plt.ylim(y_bottom, y_upper)
plt.xlim(x_bottom, x_upper)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
for j in range(6):
    cmap = plt.cm.jet
    colors = [cmap(i) for i in np.linspace(0, 1, 300)]
    for i in range(0,300):
        plt.plot(trajectory[j,i:i+2, 0], trajectory[j,i:i+2, 1], c=colors[i],marker='o',lw=4)
plt.show(block=True)
#plt.savefig("test.svg", dpi=300, format="svg")



#landscape
from scipy.ndimage import gaussian_filter
scalars = np.sqrt(xx_target**2 + yy_target**2)
scalars=gaussian_filter(scalars,sigma=2)
scalars=np.log(scalars)
def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx, yy, scalars, cmap=truncate_colormap('RdBu_r',minval=0.1, maxval=0.9),alpha=0.8)
fig.colorbar(surf)
ax.set_title("3D Surface Plot of Flow Field")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_zlabel("Flow field magnitude (Z)")
ax.set_axis_off()
plt.show(block=True)
#plt.savefig("test.svg", dpi=300, format="svg")


#################### local 多步迭代的dist和cosine similarity比较
def MSE(a, b):
    return np.mean(np.sum((a - b) ** 2,1))

import numpy as np
cosine_sim_evolve_all=np.zeros([output_latent_tp1_trials.shape[0],1])
dist_evolve_all=np.zeros([output_latent_tp1_trials.shape[0],1])
for k in range(output_latent_tp1_trials.shape[0]):
    tmp_trial_trajectory=output_latent_tp1_trials[k,:,:] # individual trials true data

    np.random.seed(42)
    random_num=100
    local_evolve_num=10 #cycle num for each point
    indices = np.random.choice(tmp_trial_trajectory.shape[0]-local_evolve_num, random_num, replace=False)  # 生成不重复的索引

    tmp_trial_trajectory_r=np.zeros([random_num,local_evolve_num,2])
    for i in range(random_num):
        tmp_trial_trajectory_r[i,:,:]=tmp_trial_trajectory[indices[i]+1:indices[i]+1+local_evolve_num,:] #真实数据随机取样后


    start_point = tmp_trial_trajectory[indices]
    evolve_local_trajectory = np.zeros([random_num,local_evolve_num+1,2])
    evolve_local_trajectory[:,0,:]=start_point
    for i in range(start_point.shape[0]):
        for j in range(local_evolve_num):
            tmp_latent = evolve_local_trajectory[i,j,:]
            tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
            tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
            tmp_latent_tp = model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
            evolve_local_trajectory[i, j+1, :] = tmp_latent_tp.cpu().detach().numpy()
    evolve_local_trajectory=evolve_local_trajectory[:,1:,:]

    #计算路径欧氏距离
    dist_evolve = np.zeros([random_num, 1])
    for i in range(random_num):
        tmp_real_trajectory = tmp_trial_trajectory_r[i, :, :]
        tmp_evolve_trajectory = evolve_local_trajectory[i, :, :]
        dist_evolve[i] = MSE(tmp_real_trajectory, tmp_evolve_trajectory)
    dist_evolve_all[k] = np.median(dist_evolve)


    #cosine similarity
    tangent_real=tmp_trial_trajectory_r[:,-1,:]-tmp_trial_trajectory_r[:,0,:]
    tangent_evolve=evolve_local_trajectory[:,-1,:]-evolve_local_trajectory[:,0,:]
    cosine_sim_evolve=np.zeros([random_num,1])
    for i in range(random_num):
        tmp_real=tangent_real[i,:]
        tmp_evolve=tangent_evolve[i,:]
        cosine_sim_evolve[i] = np.dot(tmp_real, tmp_evolve) / (np.linalg.norm(tmp_real) * np.linalg.norm(tmp_evolve))
    cosine_sim_evolve_all[k]=np.median(cosine_sim_evolve)

import pandas as pd
df=pd.DataFrame(dist_evolve_all)
df.to_clipboard(index=False,header=False)

#plot example
k=1
example=74
tmp_trial_trajectory_r = np.zeros([random_num, local_evolve_num+1, 2])
for i in range(random_num):
    tmp_trial_trajectory_r[i, :, :] = tmp_trial_trajectory[indices[i]:indices[i] + 1 + local_evolve_num,:]

start_point = tmp_trial_trajectory[indices]
evolve_local_trajectory = np.zeros([random_num, local_evolve_num + 1, 2])
evolve_local_trajectory[:, 0, :] = start_point
for i in range(start_point.shape[0]):
    for j in range(local_evolve_num):
        tmp_latent = evolve_local_trajectory[i, j, :]
        tmp_output_tp = model.decoder(torch.from_numpy(tmp_latent).to(dtype=torch.float).cuda())
        tmp_output_tp = tmp_output_tp.cpu().detach().numpy()
        tmp_latent_tp = model.encoder(torch.from_numpy(tmp_output_tp).to(dtype=torch.float).cuda())
        evolve_local_trajectory[i, j + 1, :] = tmp_latent_tp.cpu().detach().numpy()

num_pixel=30
x_bottom=-0.8
x_upper=0.05  #0.4
y_bottom=-0.5 #-0.7
y_upper=0.8
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
ax.quiver(xx,yy,xx_target,yy_target,color=[21/255,109/255,183/255],scale=5)
plt.plot(tmp_trial_trajectory_r[example,:,0],tmp_trial_trajectory_r[example,:,1],c='r')
plt.plot(evolve_local_trajectory[example,:,0],evolve_local_trajectory[example,:,1],c='b')
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
plt.show(block=True)

