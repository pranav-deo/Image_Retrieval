import torch
import numpy as np
import matplotlib.pyplot as plt
# import torch.nn as nn
# from hash_utils import get_kmeans

# path_latent = ["/home/pranav/ML/medal/Image_Retrieval/results/ae/out_tensors/parts_small/enc_{}.pt".format(ii) for ii in range(48)]
# latent_ = [torch.load(path_latent[ii], map_location={'cuda:1': 'cuda:0'}) for ii in range(48)]
# lat = torch.cat(latent_)
# torch.save(lat, "/home/pranav/ML/medal/Image_Retrieval/results/ae/out_tensors/all_encoded_small.pt")

path_latent = "../results/ae/out_tensors/all_encoded_small.pt"
latent_ = torch.load(path_latent, map_location={'cuda:1': 'cuda:0'})
# print(latent_.size())

samples = np.array([0, 10, 20, 30, 40, 50, 60, 70])
samples += 5

lat = latent_[samples, :, :, :].detach().cpu().numpy()
_, a = plt.subplots(len(samples) + 1, 8)

for pic in range(0, len(samples)):
    lat_ = lat[pic, :, :, :]
    lat_ = np.squeeze(lat_)
    # print(lat_.shape)
    for ii in range(8):
        a[ii][pic].imshow(lat_[ii, :, :])
        a[ii][pic].set_title("dim {}".format(ii))
        a[ii][pic].axis("off")

for ii in range(len(samples)):
    a[len(samples)][ii].imshow(np.squeeze(np.median(lat[ii, :, :, :], axis=0, keepdims=True)))
    a[len(samples)][ii].set_title("Averaged")
    a[len(samples)][ii].axis("off")

plt.show()

# avgpool = nn.AdaptiveMaxPool2d((1, 1))
# lat = avgpool(latent_).squeeze()
# kmeans = get_kmeans(lat.detach().cpu().numpy())
# print(kmeans.labels_)
