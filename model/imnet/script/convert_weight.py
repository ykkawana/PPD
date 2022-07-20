#%%
import torch
import os
import dotenv
dotenv.load_dotenv()
os.chdir(os.getenv('PROJECT_ROOT'))
from model.imnet import imnet
#%%
weight_path = 'artifacts/imnet/checkpoint/all_vox256_img_ae_64/IM_AE.model64-399.pth'

model = imnet.IMNetAutoEncoder()
weight = torch.load(weight_path)
model.load_state_dict(weight)
