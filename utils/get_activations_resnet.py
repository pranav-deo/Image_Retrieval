import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torchvision.utils import save_image
# import pytorch_msssim

from model_pretrained import AE

K = 16
batch_size = 96
test_data_folder = "test"

test_transform = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	transforms.ToTensor(),
])

testset = torchvision.datasets.ImageFolder(test_data_folder,transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=20)

model = AE(K=K).cuda()
model.eval()
	
# Visualize feature maps
activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook
layers = ["layer3","layer4"]
instances = [0,50,90]
for layer in layers:
	for instance in instances:
		model.pretrained_net.layer3.register_forward_hook(get_activation(layer))
		data, _ = testset[instance]
		data.unsqueeze_(0)
		data = data.cuda()
		out = model(data)
		out = out.cpu()
		for ii in activation:
			activation[ii] = activation[ii].cpu() 
		act = activation[layer].squeeze()

		print(act.size())

		import matplotlib.pyplot as plt
		fig, axarr = plt.subplots(16,32,figsize=(200,100),dpi=100)
		for row in range(16):
			for col in range(32):
				axarr[row,col].imshow(act[row*32+col])
				axarr[row,col].axis('off')

		fig.savefig('{}_{}.png'.format(layer,str(instance)),bbox_inches='tight')
		plt.close(fig)
