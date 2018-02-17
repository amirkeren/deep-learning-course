import pickle
import tarfile
import requests

import numpy as np
import matplotlib.pyplot as plt

from os.path import isfile, isdir
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer


def download_dataset(cifar10_dataset_url, cifar10_dataset_folder_path, tar_gz_path):
    if isdir(cifar10_dataset_folder_path):
        print('Cifar-10 dataset found')
        return
    print('Downloading Cifar-10 dataset...')
    if not isfile(tar_gz_path):
        r = requests.get(cifar10_dataset_url, stream=True)
        total_length = int(r.headers.get('content-length', 0))
        with open(tar_gz_path, 'wb') as f:
            for data in tqdm(r.iter_content(1), total=total_length, unit='B', unit_scale=True):
                f.write(data)
    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open(tar_gz_path) as tar:
            tar.extractall()
            tar.close()
    print('Cifar-10 dataset downloaded')


def preprocess_and_save_data(cifar10_dataset_folder_path, noise_factor, n_batches):
    print('Preprocessing data...')
    valid_features = []
    valid_labels = []
    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        validation_count = int(len(features) * 0.1)
        _preprocess_and_save(features[:-validation_count], labels[:-validation_count],
                             'preprocess_batch_' + str(batch_i) + '.p', noise_factor)
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])
    _preprocess_and_save(np.array(valid_features), np.array(valid_labels), 'preprocess_validation.p',
                         noise_factor)
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as f:
        batch = pickle.load(f)
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']
    _preprocess_and_save(np.array(test_features), np.array(test_labels), 'preprocess_test.p', noise_factor)
    print('Finished preprocessing')


def load_preprocess_training_batch(batch_id, batch_size):
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, noisy_features, labels = pickle.load(open(filename, mode='rb'))
    return _batch_features_labels(features, noisy_features, labels, batch_size)


def load_testing_batch(batch_size):
    features, noisy_features, labels = pickle.load(open('preprocess_test.p', mode='rb'))
    return _batch_features_labels(features, noisy_features, labels, batch_size)


def load_preprocess_testing():
    return pickle.load(open('preprocess_test.p', mode='rb'))


def load_preprocess_validation():
    return pickle.load(open('preprocess_validation.p', mode='rb'))


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as f:
        batch = pickle.load(f)
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


def show_images(in_imgs, noisy_imgs, reconstructed):
    fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(32, 4))
    for images, row in zip([noisy_imgs, reconstructed, in_imgs], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((32, 32, 3)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)


def print_stats(epoch, session, inputs, feature_batch, targets, label_batch,
                valid_features, valid_labels, keep_prob, cost, accuracy):
    feed_cost = {inputs: feature_batch, targets: label_batch, keep_prob: 1.0}
    feed_valid = {inputs: valid_features, targets: valid_labels, keep_prob: 1.0}
    cost = session.run(cost, feed_cost)
    accuracy = session.run(accuracy, feed_valid)
    print('Epoch {:>2}:  '.format(epoch + 1))
    print("cost: %.4f" % cost, "accuracy: %.4f" % accuracy)
    pass


def _normalize(image):
    image = np.float64(image)
    image /= 255.0
    return image


def _batch_features_labels(features, noisy_features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], noisy_features[start:end], labels[start:end]


def _one_hot_encode(x):
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(10))
    one_hot = label_binarizer.transform(x)
    return one_hot


def _make_noisy(imgs, noise_factor):
    noisy_imgs = []
    for img in imgs:
        noisy_imgs.append(_add_gaussian_noise(img, noise_factor))
    return np.asarray(noisy_imgs)


def _add_gaussian_noise(image_in, noise_factor):
    temp_image = np.float64(np.copy(image_in))
    noise = np.random.randn(32, 32) * noise_factor
    noisy_image = np.zeros(temp_image.shape, np.float64)
    noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
    noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
    noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
    return noisy_image.clip(0, 255)


def _preprocess_and_save(features, labels, filename, noise_factor):
    noisy_features = _make_noisy(features, noise_factor)
    features = _normalize(features)
    noisy_features = _normalize(noisy_features)
    labels = _one_hot_encode(labels)
    pickle.dump((features, noisy_features, labels), open(filename, 'wb'))
