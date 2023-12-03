import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from  evaluefunction import *
from  evalue import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from utils import load_agent_list_by_file

# args = parse_args()
# G = Generator(args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len, args.dropout).to(
#     args.device)
# G.load_state_dict(torch.load(args.G_dict))
# real=getAgentTrajList("eth")
# real=np.concatenate(real,axis=0)
# fake = generateTrajByDataset("eth",G,args)
# fake=np.concatenate(fake,axis=0)


data=load_agent_list_by_file('univ_sfm.csv')
data=np.concatenate(data,axis=0)


def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent



# Generate some test data


img, extent = myplot(data[:, 0], data[:, 1], 32)
plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet)

plt.show()
