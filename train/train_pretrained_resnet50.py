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
sys.path.append(cwd + '/../model')

from model_pretrained import AE
from cauchy_loss import cauchy_loss

""" Hyperparameters """
K = 16 	# K- bit hashing
q_lambda = 1  # Weight to cauchy
train_stages = [2]
num_epoch = [10, 10]
#lr_net_1 = 0.0005
lr_net_2 = 0.00005
lr_cauchy = lr_net_2 * 10
batch_size = 192
val_split = 0.1
cauchy_loss_weight = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epsilon = torch.empty(1).to(device)
nn.init.constant_(epsilon, 1e-8)

""" strings """
saved_model_name = "compression_32_d5_hash_multitask_best_weights"
train_data_folder = "/home/Drive3/pranav/NCT-CRC-HE-100K"
val_data_folder = ""

# class comp_loss(nn.Module):
#	def __init__(self):
#		super(comp_loss,self).__init__()
#
#	def forward(self,reconst, original):
#		MSSSIM = pytorch_msssim.ms_ssim(original, reconst, nonnegative_ssim=True).to(device)
#		MSE = (torch.nn.functional.mse_loss(original, reconst)).to(device)
#		loss = MSE - MSSSIM + 1
#		return loss, MSSSIM


def make_one_hot(labels, C=9):
    labels = labels.unsqueeze(1)
    '''
	Converts an integer label to a one-hot Variable.

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


if len(train_data_folder) == 0 and len(val_data_folder) == 0:
    print("Set data folder paths")
    assert(0 == 1)

train_transform = transforms.Compose([
    transforms.Resize(224, 224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

for train_stage in train_stages:
    print("Started with train stage-{}".format(train_stage))
    tensorboard_folder = 'runs/hashed_multitask'
    if train_stage == 2:
        tensorboard_folder += '_stage2'
    writer = SummaryWriter(tensorboard_folder)

    dataset = torchvision.datasets.ImageFolder(train_data_folder, transform=train_transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int((1 - val_split) * len(dataset)), int(val_split * len(dataset))])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=20)

    model = AE(K=K)
    # assert 1==0
    model = nn.DataParallel(model).to(device)
    # for name, _ in model.named_parameters():
    #	print(name)

    if train_stage == 1:
        params = list(model.parameters())
        criterion1 = comp_loss()
        params.extend(list(criterion1.parameters()))
        optimizer = torch.optim.Adam(params, lr=lr_net_1, weight_decay=1e-4)
        # sched1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,4], gamma=0.5)

        def lmbda(epoch): return 0.96 ** epoch
        sched2 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    elif train_stage == 2:
        #model.load_state_dict(torch.load(saved_model_name + '.pt'))
        hashed_layer = ['module.pretrained_net.fc.2.weight', 'module.pretrained_net.fc.6.weight']
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in hashed_layer, model.named_parameters()))))
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in hashed_layer, model.named_parameters()))))
        #params = list(filter(lambda kv: kv[0] in hashed_layer, model.named_parameters()))
        #base_params = list(filter(lambda kv: kv[0] not in hashed_layer, model.named_parameters()))
        #criterion1 = comp_loss()
        criterion2 = cauchy_loss(K=K, q_lambda=q_lambda)
        assert len(params) + len(base_params) == len(list(model.named_parameters()))
        if len(params) == 0 or len(base_params) == 0:
            print("Length base params {}".format(len(base_params)))
            print("Length params {}".format(len(params)))
            print("Different Learning rate not applied")
            assert 1 == 0
        # params.extend(list(criterion1.parameters()))
        params.extend(list(criterion2.parameters()))
        # base_params.extend(list(criterion1.parameters()))
        base_params.extend(list(criterion2.parameters()))
        optimizer = torch.optim.Adam([{'params': base_params}, {'params': params, 'lr': lr_cauchy}], lr=lr_net_2, weight_decay=1e-4)
        saved_model_name += '_stage2'
        # sched1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,4], gamma=0.5)
        #lmbda = lambda epoch: 0.95 ** epoch
        sched2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96, last_epoch=-1)

    else:
        print("Set proper train stage")
        assert(1 == 0)

    train_iter_count = 0
    val_iter_count = 0
    val_iter100_count = 0
    best_loss = 100000.0
    for epoch in range(num_epoch[train_stage - 1]):
        print("Epoch {} of {}:".format(epoch + 1, num_epoch[train_stage - 1]))
        if train_stage == 1:
            for param_group in optimizer.param_groups:
                print("LR used:{}".format(param_group['lr']))
        elif train_stage == 2:
            l = []
            for param_group in optimizer.param_groups:
                l.append(param_group['lr'])
            print("LR for net: {}, For cauchy layer: {}".format(l[0], l[1]))
        #train_mse = 0.0
        #train_msssim = 0.0
        train_hash = 0.0
        # sched2.step()
        # print("Stepped")
        # assert 1==0
        with tqdm(total=len(train_loader), desc="Batches") as pbar:
            for i, (data) in enumerate(train_loader):
                model.train()
                if train_stage == 1:
                    img, _ = data
                else:
                    img, label = data
                    label = label.to(device)
                    label = make_one_hot(label)
                img = img.to(device)
                hashed_layer = model(img)
                #loss, msssim = criterion1(output[0], img)
                if train_stage == 2:
                    loss_hash = criterion2(hashed_layer, label)
                    # train_loss += loss_hash.cpu().item()
                    train_hash += loss_hash.cpu().item()
                #train_mse += loss.cpu().item()
                #train_msssim += msssim.cpu().item()

                #writer.add_scalar('train_iter_loss', loss.cpu().item(), train_iter_count)
                #writer.add_scalar('train_iter_msssim', msssim.cpu().item(), train_iter_count)
                if train_stage == 2:
                    writer.add_scalar('train_iter_hash', loss_hash.cpu().item(), train_iter_count)

                train_iter_count += 1
                if train_stage == 2:
                    total_loss = loss_hash
                else:
                    total_loss = loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if i == len(train_loader) - 1:
                    val_loss = 0.0
                    #val_msssim = 0.0
                    with torch.no_grad():
                        with tqdm(total=len(val_loader), desc="Val Batches at {}th batch".format(i)) as vbar:
                            for j, (data) in enumerate(val_loader):
                                if train_stage == 1:
                                    img, _ = data
                                else:
                                    img, label = data
                                    label = label.to(device)
                                    label = make_one_hot(label)
                                img = img.to(device)
                                hashed_layer = model(img)
                                # if j % 50 == 0:
                                #	folder_images = 'hashed_multitask_results'
                                #	if train_stage==2:
                                #		folder_images+='_stage2'
                                #	save_image(torch.cat((img, output[0])), folder_images+'/'+str(epoch)+'_'+str(i)+'_'+str(j)+'.jpg', nrow=batch_size)
                                #loss, msssim = criterion1(output[0], img)
                                if train_stage == 2:
                                    loss_hash = criterion2(hashed_layer, label)
                                    val_loss += loss_hash.cpu().item()
                                    # loss += loss_hash
                                #val_loss += loss.cpu().item()
                                #val_msssim += msssim.cpu().item()

                                #writer.add_scalar('validation_iter_loss', loss.cpu().item(), val_iter_count)
                                #writer.add_scalar('validation_iter_msssim', msssim.cpu().item(), val_iter_count)
                                if train_stage == 2:
                                    writer.add_scalar('validation_iter_hash', loss_hash.cpu().item(), val_iter_count)

                                val_iter_count += 1
                                vbar.update(1)
                        #writer.add_scalar('validation_100_loss', val_loss*batch_size/len(valset), val_iter100_count)
                        #writer.add_scalar('validation_100_msssim', val_msssim*batch_size/len(valset), val_iter100_count)
                        writer.add_scalar('validation_100_hash', val_loss * batch_size / len(valset), val_iter100_count)
                        val_iter100_count += 1
                        if val_loss / len(valset) < best_loss:
                            torch.save(model.state_dict(), saved_model_name + '.pt')
                            best_loss = val_loss / len(valset)
                            # print('Least Val loss at '+str(epoch)+'_'+str(i)+' = '+str(batch_size*val_loss/len(valset)))
                pbar.update(1)

        # sched1.step()
        sched2.step()
        print(" Train hash: {}".format(train_hash))
        #writer.add_scalar('train_epoch_mse', train_mse*batch_size/len(trainset), epoch)
        #writer.add_scalar('train_epoch_msssim', train_msssim*batch_size/len(trainset), epoch)
        if train_stage == 2:
            writer.add_scalar('train_epoch_hash', train_hash * batch_size / len(trainset), epoch)
