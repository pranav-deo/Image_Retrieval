"""
Reference from: https://github.com/MLEnthusiast/MHCLN

"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import operator
import time
import argparse

start_time = time.time()

class_names = ['adi', 'back', 'deb', 'lym', 'muc', 'mus', 'norm', 'str', 'tum']

# Loading test instances: latent features, labels, input image data
# Total 7180
num_instances = 7180
path_latent = ["/media/pranav/storage/ML/out_tensors/h/h_{}.pt".format(ii) for ii in range(num_instances)]
path_labels = ["/media/pranav/storage/ML/out_tensors/l/l_{}.pt".format(ii) for ii in range(num_instances)]
path_input = ["/media/pranav/storage/ML/out_tensors/in/in_{}.pt".format(ii) for ii in range(num_instances)]

test_encodings = (np.squeeze(np.array([torch.load(path_latent[ii], map_location={'cuda:0': 'cpu'}).detach().numpy() for ii in range(num_instances)])))
# test_encodings = np.sign(np.squeeze(np.array([torch.load(path_latent[ii], map_location={'cuda:0': 'cpu'}).detach().numpy() for ii in range(num_instances)])))
test_labels = np.squeeze(np.array([torch.load(path_labels[ii], map_location={'cuda:0': 'cpu'}).detach().numpy() for ii in range(num_instances)]))
test_data = [np.squeeze(torch.load(path_input[ii], map_location={'cuda:0': 'cpu'}).detach().numpy()) for ii in range(num_instances)]


def hamming_distance(instance1, instance2):
    return hamming(instance1, instance2)


def vector_distance(i1, i2):
    return np.linalg.norm(i1 - i2)


def get_k_hamming_neighbours(no_nbrs, enc_test, test_img, test_lab, index, dis_type):
    """ Returns array of tuples of neighbours in (index, label, dist) format"""
    _neighbours = []  # 1(query image) + neighbours
    distances = []
    if dis_type == 0:
        global test_encodings
        test_encodings = np.sign(test_encodings)
        enc_test = np.sign(enc_test)
    for i in range(len(test_encodings)):
        if index != i:  # exclude the test instance itself from the search set
            if dis_type == 0:
                dist = hamming_distance(test_encodings[i, :], enc_test)
            else:
                dist = vector_distance(test_encodings[i, :], enc_test)
            distances.append((i, test_labels[i], dist))

    distances.sort(key=operator.itemgetter(2))
    _neighbours.append((index, test_lab, 0))
    for j in range(no_nbrs):
        _neighbours.append((distances[j][0], distances[j][1], distances[j][2]))
    return _neighbours


def plot_img_nbrs(nbrs, f_str):
    fig, ax = plt.subplots(1, len(nbrs))
    # plt.tick_params(axis='both', which='both', bottom='off', left='off')
    for i in range(len(nbrs)):
        ax[i].imshow(np.rollaxis(test_data[nbrs[i][0]], 0, 3))
        if i == 0:
            ax[i].set_title("Distances: ")
            ax[i].set_xlabel("Class: {}".format(str(class_names[nbrs[i][1]])))
        else:
            ax[i].set_title("{}".format(str(nbrs[i][2])))
            ax[i].set_xlabel("{}".format(str(class_names[nbrs[i][1]])))
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
    plt.show()


def main(args):
    global start_time
    print("Time taken to load data: {}".format(time.time() - start_time))
    start_time = time.time()
    out_dir = 'hamming_out/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    idx = args.img
    test_encoding = test_encodings[idx, :]
    test_image = test_data[idx]
    test_label = test_labels[idx]
    neighbours = get_k_hamming_neighbours(no_nbrs=args.k, enc_test=test_encoding, test_img=test_image, test_lab=test_label, index=idx, dis_type=args.dist)

    f_str = './hamming_out/sample_' + str(idx) + '.png'
    print('Time taken for retrieving {0} test images:{1}'.format(len(test_encodings), time.time() - start_time))
    plot_img_nbrs(neighbours, f_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10, help="Number of nearest neighbours")
    parser.add_argument('-img', type=int, default=10, help="index of query image")
    parser.add_argument('-dist', type=int, default=0, help="type of embeddings 0: hamming, 1: euclidean")
    args = parser.parse_args()
    main(args)
