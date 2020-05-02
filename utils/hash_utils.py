import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import pylab as pl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import scipy.spatial as S

import sys
np.set_printoptions(threshold=sys.maxsize)


def get_tensor_paths():
    path_latent = ["../results/ae_hash/out_tensors/parts_small/hash/hash_{}.pt".format(ii) for ii in range(45)]
    path_labels = ["../results/ae_hash/out_tensors/parts_small/lab/lab_{}.pt".format(ii) for ii in range(45)]
    return path_latent, path_labels


def load_tensors():
    path_latent, path_labels = get_tensor_paths()
    latent_ = [torch.load(path_latent[ii], map_location={'cuda:1': 'cuda:0'}) for ii in range(45)]
    labels_ = [torch.load(path_labels[ii], map_location={'cuda:1': 'cuda:0'}) for ii in range(45)]
    latent = torch.cat(latent_).cpu().detach().numpy()
    labels = torch.cat(labels_).cpu().detach().numpy()
    return latent, labels


def scatter(x, colors):
    palette = np.array(sns.color_palette("hls", 9))
    f = plt.figure(figsize=(48, 48))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=100, c=palette[colors.astype(np.int)])
    ax.axis('on')
    ax.axis('tight')

    txts = []
    for i in range(9):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=64)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def plot_tsne(latent_binary, kmeans_labels):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent_binary)
    sns.palplot(np.array(sns.color_palette("hls", 9)))
    scatter(tsne_results, kmeans_labels)
    plt.show()


def save_array_as_text(array, name, precision):
    np.savetxt('{}.out'.format(name), array, delimiter=',', fmt='%.{}f'.format(precision))


def get_kmeans(array):
    kmeans = KMeans(n_clusters=9, random_state=0).fit(array)
    return kmeans


def plot_confusion_matrix(y, x):
    cm = confusion_matrix(y, x)
    pl.title('Confusion matrix')
    sns.heatmap(cm, annot=True)
    plt.xlabel('K means clusters')
    plt.ylabel('True classes')
    pl.show()


def plot_hamming_distance(latent, labels, num_vec, show_dist=False):
    rand_vectors = np.random.randint(low=0, high=latent.shape[0], size=num_vec)
    lat = latent[rand_vectors, :]
    lab = labels[rand_vectors]
    ind = np.argsort(lab)
    lat = lat[ind, :]
    lab = lab[ind]
    dis_mat = np.zeros((num_vec, num_vec))
    for ii in range(lat.shape[0]):
        for jj in range(ii, lat.shape[0]):
            dis_mat[ii, jj] = S.distance.hamming(lat[ii, :], lat[jj, :])

    dis_mat += np.transpose(dis_mat)
    plt.xticks(np.arange(num_vec), lab)
    plt.yticks(np.arange(num_vec), lab)
    ax = plt.gca()
    if show_dist:
        for (j, i), label in np.ndenumerate(dis_mat):
            ax.text(i, j, label, ha='center', va='center')
    plt.title("Hamming distance between random instances")
    plt.xlabel("Real labels")
    plt.xlabel("Real labels")
    plt.imshow(dis_mat, cmap='viridis')
    plt.colorbar()
    plt.show()


def main():
    latent, labels = load_tensors()
    latent_binary = np.sign(latent)
    save_array_as_text(latent, "latent", 2)
    save_array_as_text(latent_binary, "hash", 0)

    assert np.all((latent_binary == 1) + (latent_binary == -1))

    kmeans = get_kmeans(latent_binary)
    # plot_tsne(latent_binary, kmeans.labels_)
    plot_confusion_matrix(labels, kmeans.labels_)
    # plot_hamming_distance(latent_binary, labels, 100)


if __name__ == '__main__':
    main()
