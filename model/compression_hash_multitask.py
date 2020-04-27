import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Num_channels = 32
epsilon = 1e-10
# nn.init.constant_(epsilon, 1e-10)
inner_channels = 128
d = 5

class AE(nn.Module):
	def __init__(self, K):
		super(AE, self).__init__()

		# ENCODER
		self.e_conv_1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding = 2),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)

		self.e_conv_2 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=inner_channels, kernel_size=(5, 5), stride=(2, 2), padding = 2),
			nn.BatchNorm2d(inner_channels),
			nn.ReLU()
		)

		self.e_block_1 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.e_block_2 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.e_block_3 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.e_block_4 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.e_block_5 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.e_block_6 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.e_block_16 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))

		self.e_conv_3 = nn.Sequential(
			nn.Conv2d(in_channels=inner_channels, out_channels=Num_channels, kernel_size=(5, 5), stride=(2, 2), padding = 2),
			nn.BatchNorm2d(Num_channels)
		)

		self.imp_net = self.res_block(in_channels=Num_channels,out_channels=Num_channels,kernel_size=(3,3),stride=(1,1))

		# DECODER

		self.d_up_conv_1 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=Num_channels, out_channels=inner_channels, kernel_size= (5,5), stride=(2,2), padding=2,output_padding=1),
			nn.BatchNorm2d(inner_channels),
			nn.ReLU()
		)

		self.d_block_1 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.d_block_2 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.d_block_3 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.d_block_4 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.d_block_5 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.d_block_6 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))
		self.d_block_16 = self.res_block(in_channels=inner_channels,out_channels=inner_channels,kernel_size=(3,3),stride=(1,1))

		self.d_up_conv_2 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=inner_channels, out_channels=64, kernel_size= (5,5), stride=(2,2), padding=2,output_padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)

		self.d_up_conv_3 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size= (5,5), stride=(2,2), padding=2,output_padding=1),
			nn.BatchNorm2d(3)
		)

		self.hashed_layer = nn.Sequential(
							nn.Linear(Num_channels, K),
							nn.BatchNorm1d(num_features=K),
							nn.Tanh()
						) 

	def forward(self, x):
		# x,mean,std = self.normalize(x)
		ec1 = self.e_conv_1(x)
		ec2 = self.e_conv_2(ec1)
		eblock1 = self.e_block_1(ec2)
		eblock2 = self.e_block_2(eblock1+ec2)
		eblock3 = self.e_block_3(eblock2+eblock1)
		esum1 = eblock3+ec2+eblock2
		eblock4 = self.e_block_4(esum1)
		eblock5 = self.e_block_5(eblock4+esum1)
		eblock6 = self.e_block_6(eblock5+eblock4)
		esum2 = eblock6 + esum1+eblock5
		eblock16 = self.e_block_16(esum2)
		esum6 = eblock16 + ec2+esum2
		ec3 = self.e_conv_3(esum6)
		encode_out = torch.sigmoid(ec3)
		# print(encode_out.size())
		# assert 1==0
		# q = self.softQuantizer(z = encode_out, d = d, pow_ = 16)
		# two_bit_in = self.imp_net(encode_out)
		# two_bit_in = two_bit_in + encode_out
		# two_bit_in = torch.sigmoid(two_bit_in)
		# imp = self.softQuantizer(z = two_bit_in, d = 2, pow_ = 16)
		# z_hat = self.masking(q, imp, d)
		z_hat = encode_out
		return self.decode(z_hat, z_hat), self.hashed_layer(self.give_GAP(z_hat))

	def decode(self, decode_in, imp):
		dc1 = self.d_up_conv_1(decode_in)
		dblock1 = self.d_block_1(dc1)
		dblock2 = self.d_block_2(dblock1+dc1)
		dblock3 = self.d_block_3(dblock2+dblock1)
		dsum1 = dblock3+dc1+dblock2
		dblock4 = self.d_block_4(dsum1)
		dblock5 = self.d_block_5(dblock4+dsum1)
		dblock6 = self.d_block_6(dblock5+dblock4)
		dsum2 = dblock6 + dsum1+dblock5
		dblock16 = self.d_block_16(dsum2)
		dsum6 = dblock16 + dc1 + dsum2
		dc2 = self.d_up_conv_2(dsum6)
		dc3 = self.d_up_conv_3(dc2)
		return dc3, imp

	def res_block(self,in_channels,out_channels,kernel_size,stride):
		net = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels, out_channels, kernel_size, stride),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels, out_channels, kernel_size, stride),
			nn.BatchNorm2d(out_channels)
		)
		return net

	def softQuantizer(self, z, d, pow_):		
		z = z*(2**(d))
		hard_out = torch.floor(z.clone())
		z = z.unsqueeze(-1)
		levels = torch.Tensor(range(2**(d))).cuda()
		abs_dist = z - levels
		norm_dist = -1 * abs_dist**pow_
		level_weights = torch.nn.Softmax(dim=-1)(norm_dist)
		soft_out = torch.sum(level_weights*levels, dim=-1)
		return hard_out - soft_out.detach() + soft_out

	def masking(self,q,imp,d):
		z = q/((2**((2*d)-imp))+epsilon)
		# print("masking input",z)
		# print("masking input size",z.size())
		masked_out = (self.softQuantizer(z,d, pow_=16))*(2**(d-imp))
		return masked_out

	def give_GAP(self, f):
		gap_features = F.adaptive_avg_pool2d(f, (1, 1))
		gap_features = gap_features.squeeze()
		return gap_features