import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import pytorch_msssim

import sys
import os
cwd = os.getcwd()
sys.path.append(cwd+'/../model')

from model.model_compression_32_d5_hashed_multitask import AE
from cauchy_loss import cauchy_loss 

""" Hyprtparameters """
K = 16 	# K- bit hashing 
q_lambda = 0.01 #Weight to cauchy 
train_stage = 1
lr_net_1 = 0.0005
lr_net_2 = 0.0001
lr_cauchy = lr_net_2*10

""" strings """
saved_model_name = "compression_32_d5_hash_multitask_best_weights.pt"
train_data_folder = ""
val_data_folder = ""

if len(train_data_folder)*len(val_data_folder) == 0:
	print("Set data folder paths")
	assert(0==1) 

writer = SummaryWriter('runs/hashed_multitask')
batch_size = 256
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

epsilon = torch.empty(1).cuda()
nn.init.constant_(epsilon, 1e-8)

train_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),	
	transforms.ToTensor(),
])

val_transform = transforms.Compose([
	transforms.ToTensor(),
])

trainset = torchvision.datasets.ImageFolder(train_data_folder,transform=train_transform)
train_loader = torch.utils.cpu().item().DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

valset = torchvision.datasets.ImageFolder(val_data_folder,transform=val_transform)
val_loader = torch.utils.cpu().item().DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

class comp_loss(nn.Module):
	def __init__(self):
		super(comp_loss,self).__init__()
		# self.sigma_1 = torch.nn.Parameter(torch.randn(1).to(device))
		# self.sigma_1.requires_grad = True
		# self.sigma_2 = torch.nn.Parameter(torch.randn(1).to(device))
		# self.sigma_2.requires_grad = True

	def forward(self,reconst, original):
		MSSSIM = pytorch_msssim.msssim(original, reconst).cuda()
		MSE = (torch.nn.functional.mse_loss(original, reconst)).cuda()
		loss = MSE - MSSSIM + 1
		return loss, MSSSIM


model = AE(K=K).cuda()
model = nn.DataParallel(model, device_ids=[0,1,2])

if train_stage == 1:
	params = list(model.parameters())
	criterion1 = comp_loss()
	params.extend(list(criterion1.parameters()))
	optimizer = torch.optim.Adam(params, lr=lr_net_1, weight_decay=1e-4)

elif train_stage == 2:
	model.load_state_dict(torch.load(saved_model_name))
	hashed_layer = ['hashed_layer.0.weight', 'hashed_layer.0.bias']
	params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in hashed_layer, model.named_parameters()))))
	base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in hashed_layer, model.named_parameters()))))
	criterion1 = comp_loss()
	criterion2 = cauchy_loss(K=K, q_lambda=q_lambda)
	params.extend(list(criterion1.parameters()))
	params.extend(list(criterion2.parameters()))
	base_params.extend(list(criterion1.parameters()))
	base_params.extend(list(criterion2.parameters()))
	optimizer = torch.optim.Adam([{'params': base_params}, {'params': params, 'lr': str(lr_cauchy)}], lr=lr_net_2, weight_decay=1e-4)

else:
	print("Set proper train stage")
	assert(1==0)

train_iter_count = 0
val_iter_count = 0
val_iter100_count = 0
best_loss = 100000.0
for epoch in range(100):
	train_loss = 0.0
	train_msssim = 0.0
	for i, (data) in enumerate(train_loader):
		model.train()
		if train_stage==1:
			img,_ = data
		else:
			img, label = data
			label = label.cuda()
		img = img.cuda()
		output, hashed_layer = model(img)
		loss, msssim = criterion1(output, img)
		if train_stage == 2:
			loss_hash = criterion2(hashed_layer, label)
			train_loss += loss_hash.cpu().item()
		train_loss += loss.cpu().item()
		train_msssim += msssim.cpu().item()
		
		writer.add_scalar('train_iter_loss', loss.cpu().item(), train_iter_count)
		writer.add_scalar('train_iter_msssim', msssim.cpu().item(), train_iter_count)
		
		train_iter_count += 1

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i % 50 == 0:
			val_loss = 0.0
			val_msssim = 0.0
			with torch.no_grad():
				for j, (data) in enumerate(val_loader):
					if train_stage==1:
						img, _ = data
					else:
						img, label = data
						label = label.cuda()
					img = img.cuda()
					output, hashed_layer = model(img)
					if j % 50 == 0:
						save_image(torch.cat((img, output.clamp(0,1))), 'hashed_multitask_results/'+str(epoch)+'_'+str(i)+'_'+str(j)+'.jpg', nrow=batch_size)
					loss, msssim = criterion1(output, img)
					if train_stage==2:
						loss_hash = criterion2(hashed_layer, label)
						val_loss += loss_hash.cpu().item()
					val_loss += loss.cpu().item()
					val_msssim += msssim.cpu().item()
					
					writer.add_scalar('validation_iter_loss', loss.cpu().item(), val_iter_count)
					writer.add_scalar('validation_iter_msssim', msssim.cpu().item(), val_iter_count)
					val_iter_count += 1

				writer.add_scalar('validation_100_loss', val_loss*batch_size/len(valset), val_iter100_count)
				writer.add_scalar('validation_100_msssim', val_msssim*batch_size/len(valset), val_iter100_count)
				val_iter100_count += 1
				if val_loss/len(valset) < best_loss:
					torch.save(model.state_dict(), saved_model_name)
					best_loss = val_loss/len(valset)
					print('Least Val loss at '+str(epoch)+'_'+str(i)+' = '+str(batch_size*val_loss/len(valset)))
	
	writer.add_scalar('train_epoch_loss', train_loss*batch_size/len(trainset), epoch)
	writer.add_scalar('train_epoch_msssim', train_msssim*batch_size/len(trainset), epoch)
