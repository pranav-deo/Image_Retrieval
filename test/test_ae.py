import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import sys
import os

# Adding models directory to systems path
cwd = os.getcwd()
sys.path.append(cwd + '../model')
sys.path.append(cwd + '../utils')


from ae import AE
import json
with open("./hyperparams_colab_ae.json", "r") as read_file:
    hyperparams = json.load(read_file)

batch_size = hyperparams["batch_size"]
val_data_folder = hyperparams["val_data_folder"]
saved_model_name = hyperparams["saved_model_name"]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_transform = transforms.Compose([
    transforms.ToTensor(),
])


testset = torchvision.datasets.ImageFolder(val_data_folder, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=20)


model = AE().to(device)
model = nn.DataParallel(model, device_ids=[0])
model.load_state_dict(torch.load(saved_model_name))

if not os.path.exists('./results'):
    os.makedirs('./results')

for i, (data) in enumerate(test_loader):
    model.eval()
    print(i)
    img, _ = data
    encoded, out = model(img)
    torch.save(encoded, "/content/results/enc_{}.pt".format(i))
