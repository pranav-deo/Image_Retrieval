import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import sys
import os

# Adding models directory to systems path
cwd = os.getcwd()
sys.path.append(cwd + '/../model')
sys.path.append(cwd + '/../utils')

from ae import AE
import json
with open("./hyperparams_ae.json", "r") as read_file:
    hyperparams = json.load(read_file)

batch_size = hyperparams["batch_size"]
val_data_folder = hyperparams["val_data_folder"]
saved_model_name = hyperparams["saved_model_name"]
save_folder_name = hyperparams["save_folder_name"]
K = hyperparams["K"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_transform = transforms.Compose([
    transforms.ToTensor(),
])


testset = torchvision.datasets.ImageFolder(val_data_folder, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=20)


model = AE(K=K).to(device)
model = nn.DataParallel(model, device_ids=[0])
model.load_state_dict(torch.load(saved_model_name, map_location={'cuda:1': 'cuda:0'}))

if not os.path.exists(save_folder_name):
    os.makedirs(save_folder_name)

with tqdm(total=len(test_loader), desc="Batches") as pbar:
    for i, (data) in enumerate(test_loader):
        model.eval()
        img, labels = data
        encoded, out, hashed = model(img)
        torch.save(out, save_folder_name + "/out/out_{}.pt".format(i))
        torch.save(labels, save_folder_name + "/lab/lab_{}.pt".format(i))
        torch.save(hashed, save_folder_name + "/hash/hash_{}.pt".format(i))
        pbar.update(1)
