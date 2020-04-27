import torch

def make_one_hot(labels, C=9):
	labels = labels.unsqueeze(1);
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
	device = "cuda:{}".format(labels.get_device())
	one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(device)
	target = one_hot.scatter_(1, labels.data, 1)
	return target
