import torch.nn as nn
import torch
import numpy as np
import math

class cauchy_loss(nn.Module):
	"""docstring for cauchy_loss"""
	def __init__(self):
		super(cauchy_loss, self).__init__()
		self.output_dim = None
		self.q_lambda = None
		self.gamma = None
		self.img_last_layer = None
		self.img_label = None

	def cauchy_cross_entropy(self, u, label_u, v=None, label_v=None, gamma=1, normed=True):
		if v is None:
			v, label_v = u ,label_u

		if len(shape(label_u)) == 1:
			label_ip = label_u @ label_v.unsqueeze(1)
		else:
			label_ip = label_u @ label_v.t()

		s = torch.clamp(label_ip, 0.0, 1.0)

		if normed:
			ip_1 = u @ v.t()
			mod_1 =  torch.sqrt(torch.dot(torch.sum(u**2),torch.sum(v**2)+0.000001))
			dist = self.output_dim/2.0 * (1.0- ip_1/mod_1) + 0.000001

		else:
			r_u = torch.sum(u**2, 1)
			r_v = torch.sum(v**2, 1)
			dist = r_u - 2 * (u @ v.t()) + r_v.unsqueeze(1) + 0.001

		cauchy = gamma / (dist + gamma)

		s_t = 2*(s-0.5)
		sum_1 = torch.sum(s)
		sum_all = torch.sum(torch.abs(s_t))
		balance_param = torch.abs(s-1) + s*sum_all/sum_1

		mask = torch.eye(u.shape[0]) == 0 
		cauchy_mask = cauchy[mask]
		s_mask = s[mask]
		balance_p_mask = balance_param[mask]

		all_loss = -s_mask*torch.log(cauchy_mask)-(1-s_mask)*torch.log(1-cauchy_mask)

		return torch.sum(all_loss * balance_p_mask)

	def forward(self):
		# loss function
		self.cos_loss = self.cauchy_cross_entropy(self.img_last_layer, self.img_label, gamma=self.gamma, normed=False)
		self.q_loss_img = (torch.norm(torch.abs(self.img_last_layer)-1.0))**2
		self.q_loss = self.q_lambda * self.q_loss_img
		self.loss = self.cos_loss + self.q_loss
		return self.loss