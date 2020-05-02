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


from resnet50_ae_hash import ResNet_AE
import json
with open("./hyperparams_resae.json", "r") as read_file:
    hyperparams = json.load(read_file)

K = hyperparams["K"]
q_lambda = hyperparams["q_lambda"]
batch_size = hyperparams["batch_size"]
val_data_folder = hyperparams["val_data_folder"]
saved_model_name = hyperparams["saved_model_name"]

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_transform = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.ImageFolder(val_data_folder, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=20)


model = ResNet_AE(K=K).to(device)
# model = nn.DataParallel(model, device_ids=[1, 0])
model = nn.DataParallel(model, device_ids=[0])
model.load_state_dict(torch.load(saved_model_name, map_location={'cuda:1': 'cuda:0'}))

# from losses import cauchy_loss, comp_loss

# criterion1 = comp_loss()
# criterion2 = cauchy_loss(K=K, q_lambda=q_lambda)

for f in ['h', 'l', 'in']:
    if not os.path.isdir(os.getcwd() + '/../results/resnet_ae/out_tensors/{}'.format(f)):
        os.makedirs(os.getcwd() + '/../results/resnet_ae/out_tensors/{}'.format(f))

with tqdm(total=len(test_loader), desc="Batches") as pbar:
    for i, (data) in enumerate(test_loader):
        model.eval()
        img, label = data
        out, hashed_layer = model(img)
        # save_image(torch.cat((img.cpu(), output.cpu())), folder_images+'/'+str(i)+'.jpg', nrow=batch_size)
        # imp = [output[1].cpu()[:,ii,:,:] for ii in range(32)]
        # imp = [t[:,None,:,:] for t in imp]
        # print(imp[0].size())
        # save_image(torch.cat(imp), folder_images+'/'+str(i)+'_imp.jpg', nrow=batch_size)
        # print("="*30)
        # torch.set_printoptions(profile="full")
        # print("Image {}".format(i),hashed_layer)
        # torch.set_printoptions(profile="default")
        # loss, msssim = criterion1(output.cpu().float(), img.cpu().float())
        # loss_hash = criterion2(hashed_layer)
        # print("Loss occured: {}, MSSIM: {}, Cauchy quantization loss: {}".format(loss,msssim,loss_hash))
        # print("\n")
        # labels.append(label)
        # hashed_layers.append(hashed_layer)
        # torch.save(out, "./test_results/img_out.pt")
        torch.save(hashed_layer, "../results/resnet_ae/out_tensors/h/h_{}.pt".format(i))
        torch.save(label, "../results/resnet_ae/out_tensors/l/l_{}.pt".format(i))
        torch.save(img, "../results/resnet_ae/out_tensors/in/in_{}.pt".format(i))
        pbar.update(1)

# torch.save(labels, "./test_results/labels.pt")
# torch.save(hashed_layers, "./test_results/hashed_layers.pt")
