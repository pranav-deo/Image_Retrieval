import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pytorch_msssim

import sys
import os
cwd = os.getcwd()
sys.path.append(cwd+'/../model')

from model_pretrained import AE

K = 16 	# K- bit hashing 
q_lambda = 1 #Weight to cauchy 
batch_size = 16
test_data_folder = "test"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

saved_model_name = "compression_32_d5_hash_multitask_best_weights_stage2"
folder_images = "test_results"

test_transform = transforms.Compose([
	transforms.ToTensor(),
])

dataset = torchvision.datasets.ImageFolder(test_data_folder,transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=20)

model = AE(K=K).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(saved_model_name + '.pt'))

# from train_compression_32_d5_hashed_multitask import comp_loss

# class comp_loss(nn.Module):
# 	def __init__(self):
# 		super(comp_loss,self).__init__()

# 	def forward(self,reconst, original):
# 		MSSSIM = pytorch_msssim.ms_ssim(original, reconst, nonnegative_ssim=True).to(device)
# 		MSE = (torch.nn.functional.mse_loss(original, reconst)).to(device)
# 		loss = MSE - MSSSIM + 1
# 		return loss, MSSSIM

# from cauchy_loss import cauchy_loss

# criterion1 = comp_loss()
# criterion2 = cauchy_loss(K=K, q_lambda=q_lambda)

for i,(data) in enumerate(test_loader):
	model.eval()
	img, _ = data
	img = img.to(device)
	hashed_layer = model(img)
	# save_image(torch.cat((img.cpu(), output[0].cpu())), folder_images+'/'+str(i)+'.jpg', nrow=batch_size)
	# imp = [output[1].cpu()[:,ii,:,:] for ii in range(32)]
	# imp = [t[:,None,:,:] for t in imp]
	# print(imp[0].size())
	# save_image(torch.cat(imp), folder_images+'/'+str(i)+'_imp.jpg', nrow=batch_size)
	print("="*30)
	torch.set_printoptions(profile="full")
	print("Image {}".format(i),hashed_layer)
	torch.set_printoptions(profile="default")
	torch.save(hashed_layer,"./test_results/hashed_layer_pretrained_resnet50_{}.pt".format(i))
	# loss, msssim = criterion1(output[0].cpu().float(), img.cpu().float())
	# loss_hash = criterion2(hashed_layer, None)
	# print("Loss occured: {}, MSSIM: {}, Cauchy quantization loss: {}".format(loss,msssim,loss_hash))
	print("\n")