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

from model_compression_32_d5_hashed_multitask import AE
from cauchy_loss import cauchy_loss 

""" Hyprtparameters """
K = 16 	# K- bit hashing 
q_lambda = 0.01 #Weight to cauchy 
train_stage = 2
lr_net_1 = 0.0005
lr_net_2 = 0.0001
lr_cauchy = lr_net_2*10

""" strings """
saved_model_name = "compression_32_d5_hash_multitask_best_weights"
train_data_folder = "/home/Drive3/pranav/NCT-CRC-HE-100K"
val_data_folder = ""

if len(train_data_folder) == 0 and len(val_data_folder) == 0:
	print("Set data folder paths")
	assert(0==1) 

tensorboard_folder = 'runs/hashed_multitask'
if train_stage==2:
	tensorboard_folder+='_stage2'
writer = SummaryWriter(tensorboard_folder)
batch_size = 80
val_split = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epsilon = torch.empty(1).to(device)
nn.init.constant_(epsilon, 1e-8)

train_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),	
	transforms.ToTensor(),
])

# val_transform = transforms.Compose([
# 	transforms.ToTensor(),
# ])

dataset = torchvision.datasets.ImageFolder(train_data_folder,transform=train_transform)
trainset, valset = torch.utils.data.random_split(dataset, [int((1-val_split)*len(dataset)), int(val_split*len(dataset))])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=20)

class comp_loss(nn.Module):
	def __init__(self):
		super(comp_loss,self).__init__()
		# self.sigma_1 = torch.nn.Parameter(torch.randn(1).to(device))
		# self.sigma_1.requires_grad = True
		# self.sigma_2 = torch.nn.Parameter(torch.randn(1).to(device))
		# self.sigma_2.requires_grad = True

	def forward(self,reconst, original):
		MSSSIM = pytorch_msssim.ms_ssim(original, reconst, nonnegative_ssim=True).to(device)
		# print(original.shape)
		# print(reconst.shape)
		# if torch.isnan(MSSSIM).any():
		# 	print("msssim is not working")
		MSE = (torch.nn.functional.mse_loss(original, reconst)).to(device)
		loss = MSE - MSSSIM + 1
		return loss, MSSSIM

def make_one_hot(labels, C=9):
	labels = labels.unsqueeze(1);
	'''
	Converts an integer label torch.autograd.Variable to a one-hot Variable.

	Parameters
	----------
	labels : torch.autograd.Variable of torch.cuda.LongTensor
	    N x 1 , where N is batch size. 
	    Each value is an integer representing correct classification.
	C : integer. 
	    number of classes in labels.

	Returns
	-------
	target : torch.autograd.Variable of torch.cuda.FloatTensor
	    N x C, where C is class number. One-hot encoded.
	'''
	# print(labels.size())
	one_hot = torch.cuda.FloatTensor(labels.size(0), C).zero_()
	target = one_hot.scatter_(1, labels.data, 1)
	return target

# def criterion1(reconst, original):
# 	MSSSIM = pytorch_msssim.ms_ssim(original, reconst, nonnegative_ssim=True).to(device)
# 	# print(original.shape)
# 	# print(reconst.shape)
# 	# if torch.isnan(MSSSIM).any():
# 	# 	print("msssim is not working")
# 	MSE = (torch.nn.functional.mse_loss(original, reconst)).to(device)
# 	# if torch.isnan(MSE).any():
# 	# 	print("mse is not working")
# 	loss = MSE - MSSSIM + 1
# 	return loss, MSSSIM

model = AE(K=K)
model = nn.DataParallel(model).to(device)

if train_stage == 1:
	params = list(model.parameters())
	criterion1 = comp_loss()
	params.extend(list(criterion1.parameters()))
	optimizer = torch.optim.Adam(params, lr=lr_net_1, weight_decay=1e-4)

elif train_stage == 2:
	model.load_state_dict(torch.load(saved_model_name + '.pt'))
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
	saved_model_name += '_stage2'

else:
	print("Set proper train stage")
	assert(1==0)

train_iter_count = 0
val_iter_count = 0
val_iter100_count = 0
best_loss = 100000.0
for epoch in range(100):
	print("Epoch {} of 100:  ".format(epoch+1),end = " ")
	train_loss = 0.0
	train_msssim = 0.0
	train_hash = 0.0
	for i, (data) in enumerate(train_loader):
		model.train()
		if train_stage==1:
			img,_ = data
		else:
			img, label = data
			label = label.to(device)
			label = make_one_hot(label)
		img = img.to(device)
		output, hashed_layer = model(img)
		loss, msssim = criterion1(output[0], img)
		if train_stage == 2:
			loss_hash = criterion2(hashed_layer, label)
			train_loss += loss_hash.cpu().item()
			train_hash += loss_hash.cpu().item()
			loss += loss_hash.cpu().item()
		train_loss += loss.cpu().item()
		train_msssim += msssim.cpu().item()
		
		writer.add_scalar('train_iter_loss', loss.cpu().item(), train_iter_count)
		writer.add_scalar('train_iter_msssim', msssim.cpu().item(), train_iter_count)
		if train_stage==2:
			writer.add_scalar('train_iter_hash', loss_hash.cpu().item(), train_iter_count)
		
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
						label = label.to(device)
						label = make_one_hot(label)
					img = img.to(device)
					output, hashed_layer = model(img)
					if j % 50 == 0:
						folder_images = 'hashed_multitask_results'
						if train_stage==2:
							folder_images+='_stage2'
						save_image(torch.cat((img, output[0])), folder_images+'/'+str(epoch)+'_'+str(i)+'_'+str(j)+'.jpg', nrow=batch_size)
					loss, msssim = criterion1(output[0], img)
					if train_stage==2:
						loss_hash = criterion2(hashed_layer.to(device), label.to(device))
						val_loss += loss_hash.cpu().item()
						loss += loss_hash.cpu().item()
					val_loss += loss.cpu().item()
					val_msssim += msssim.cpu().item()
					
					writer.add_scalar('validation_iter_loss', loss.cpu().item(), val_iter_count)
					writer.add_scalar('validation_iter_msssim', msssim.cpu().item(), val_iter_count)
					if train_stage==2:
						writer.add_scalar('validation_iter_hash', loss_hash.cpu().item(), val_iter_count)

					val_iter_count += 1

				writer.add_scalar('validation_100_loss', val_loss*batch_size/len(valset), val_iter100_count)
				writer.add_scalar('validation_100_msssim', val_msssim*batch_size/len(valset), val_iter100_count)
				val_iter100_count += 1
				if val_loss/len(valset) < best_loss:
					torch.save(model.state_dict(), saved_model_name+'.pt')
					best_loss = val_loss/len(valset)
					print('Least Val loss at '+str(epoch)+'_'+str(i)+' = '+str(batch_size*val_loss/len(valset)))
		
	print("Train loss: {}. MSSSIM loss: {}".format(train_loss,train_msssim))
	writer.add_scalar('train_epoch_loss', train_loss*batch_size/len(trainset), epoch)
	writer.add_scalar('train_epoch_msssim', train_msssim*batch_size/len(trainset), epoch)
	if train_stage==2:
		writer.add_scalar('train_epoch_hash', train_hash*batch_size/len(trainset), epoch)
