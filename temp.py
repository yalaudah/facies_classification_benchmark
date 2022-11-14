import numpy as np

dataset = 'train'

if dataset == 'train':
    train_seismic = np.load('data/train/train_seismic.npy')
    train_labels = np.load('data/train/train_labels.npy')
elif dataset == 'test1':
    probe_seismic = np.load('data/test_once/test1_seismic.npy')
    probe_labels = np.load('data/test_once/test1_labels.npy')
elif dataset == 'test2':
    probe_seismic = np.load('data/test_once/test2_seismic.npy')
    probe_labels = np.load('data/test_once/test2_labels.npy')

print('Load finished.')