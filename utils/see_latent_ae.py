import torch
import numpy as np
import matplotlib.pyplot as plt

path_latent = ["/media/pranav/storage/ML/out_tensors/h/h_{}.pt".format(ii) for ii in range(7180)]
path_labels = ["/media/pranav/storage/ML/out_tensors/l/l_{}.pt".format(ii) for ii in range(7180)]
path_input = ["/media/pranav/storage/ML/out_tensors/in/in_{}.pt".format(ii) for ii in range(7180)]

latent_ = [torch.load(path_latent[ii], map_location={'cuda:0': 'cpu'}) for ii in range(7180)]
labels_ = [torch.load(path_labels[ii], map_location={'cuda:0': 'cpu'}) for ii in range(7180)]
input_ = [torch.load(path_input[ii], map_location={'cuda:0': 'cpu'}) for ii in range(7180)]

lat = torch.cat(latent_).detach().cpu().numpy()
lab = torch.cat(labels_).detach().cpu().numpy()
inp = torch.cat(input_).detach().cpu().numpy()

np.save(lat, "/media/pranav/storage/ML/out_tensors/latent.npy")
np.save(lab, "/media/pranav/storage/ML/out_tensors/labels.npy")
np.save(inp, "/media/pranav/storage/ML/out_tensors/images.npy")

# path_latent = "../results/ae_hash/out_tensors/out_encoded_small.pt"
# latent_ = torch.load(path_latent, map_location={'cuda:1': 'cuda:0'})
# # print(latent_.size())

# samples = np.array([0, 10, 20, 30, 40, 50, 60, 70])
# samples += 5

# lat = latent_[samples, :, :, :].detach().cpu().numpy()
# # _, a = plt.subplots(len(samples) + 1, 8)
# _, a = plt.subplots(1, len(samples))

# # for pic in range(0, len(samples)):
# #     lat_ = lat[pic, :, :, :]
# #     lat_ = np.squeeze(lat_)
# #     # print(lat_.shape)
# #     for ii in range(8):
# #         a[ii][pic].imshow(lat_[ii, :, :])
# #         a[ii][pic].set_title("dim {}".format(ii))
# #         a[ii][pic].axis("off")

# # for ii in range(len(samples)):
# #     a[len(samples)][ii].imshow(np.squeeze(np.median(lat[ii, :, :, :], axis=0, keepdims=True)))
# #     a[len(samples)][ii].set_title("Averaged")
# #     a[len(samples)][ii].axis("off")

# for ii in range(len(samples)):
#     a[ii].imshow(np.rollaxis(lat[ii, :, :, :], 0, 3))
#     a[ii].axis("off")


# plt.show()

# avgpool = nn.AdaptiveMaxPool2d((1, 1))
# lat = avgpool(latent_).squeeze()
# kmeans = get_kmeans(lat.detach().cpu().numpy())
# print(kmeans.labels_)
