# Main notes: #

- Train Code and Architecture changed from previous codes of compression_d5_32
- Used GAP to join 32 feature tensors of shape=input_img.shape to a linear layer(in=32, out=len_hash_vector)
	- Alternatively, direct connection or pooling can pe done
- Losses are calculated for full batch i.e. cosine distance from each of the (batch_size - 1) images of batch (with similarity from labels) is considered
	- i.e. Loss from DCH paper is used
	- Concept of triplet loss, and modified cauchy losses from MICCAI2019 paper are **NOT IMPLEMENTED**

# Doubts: #

- MSE used to measure quantization loss instead of proposed cauchy_quantization_loss in DCH implementation supplied by the paper
	- Used MSE but kept provision for both in my code
- Undecided when to stop for stage-I and start stage-II training
- Undecided hyperparameters:
	- length_hash_code
	- learning rate for stage 2
	- weight to quantization in cauchy loss function
