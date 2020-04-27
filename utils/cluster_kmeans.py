import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
# import pandas as pd
# from sklearn.datasets import fetch_mldata
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.patheffects as PathEffects
# import seaborn as sns

import sys
np.set_printoptions(threshold=sys.maxsize)

path_latent = ["./hashed_layer_0.pt"]

# def scatter(x, colors):
# 	# We choose a color palette with seaborn.
# 	palette = np.array(sns.color_palette("hls", 9))

# 	# We create a scatter plot.
# 	f = plt.figure(figsize=(48, 48))
# 	ax = plt.subplot(aspect='equal')
# 	sc = ax.scatter(x[:,0], x[:,1], lw=0, s=100,
# 					c=palette[colors.astype(np.int)])
# 	ax.axis('on')
# 	ax.axis('tight')

# 	# We add the labels for each digit.
# 	txts = []
# 	for i in range(9):
# 		# Position of each label.
# 		xtext, ytext = np.median(x[colors == i, :], axis=0)
# 		txt = ax.text(xtext, ytext, str(i), fontsize=64)
# 		txt.set_path_effects([
# 			PathEffects.Stroke(linewidth=5, foreground="w"),
# 			PathEffects.Normal()])
# 		txts.append(txt)

# 	return f, ax, sc, txts

def main():
	latent = [torch.load(path_latent[0])]
	latent_features = torch.cat(latent).cpu().detach().numpy()
	np.savetxt('test.out', latent_features, delimiter=',', fmt='%.2f')
	latent_features_binary = np.sign(latent_features)
	np.savetxt('test1.out', latent_features_binary, delimiter=',', fmt='%.0f')

	assert np.all((latent_features_binary == 1) + (latent_features_binary == -1))

	kmeans = KMeans(n_clusters=9, random_state=0).fit(latent_features_binary)
	print("Kmeans labels: ",kmeans.labels_)

	# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	# tsne_results = tsne.fit_transform(latent_features_binary)

	# sns.palplot(np.array(sns.color_palette("hls", 9)))
	# scatter(tsne_results, kmeans.labels_)
	# plt.show()

if __name__ == '__main__':
	main()