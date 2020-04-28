# Main notes: #

## Tried Architectures: ##
1) Resnet Core -> hash layers
2) **Resnet Core -> (decoder, hash layers)**
3) Encoder -> Decoder <Only reconstruction>

## Some notes: ##
- Train Code and Architecture changed from previous codes of autoencoder
- Used GAP to join 32 feature tensors of encoder to a linear layer(in=32, out=len_hash_vector) **Difficult for model to backprop. Hash loss is difficult to  minimize**
	- Alternatively, direct connection or pooling can pe done **Seems better as of now**
	- Currently on linear bottleneck between encoder and decoder with *bad image reconstruction* but *okay hashes* 
 
- Losses namely: cosine cross entropy, cauchy quantization *(better than mse for hashes)*, mse, mssim *(seems useless as of now)* 
	- Concept of triplet loss, and modified cauchy losses from MICCAI2019 paper are **NOT IMPLEMENTED**
