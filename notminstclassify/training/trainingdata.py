import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve

class TrainingData:
    def __init__(self, src_datafile, cached_file):
        self.src_datafile = src_datafile
        self.cached_file = cached_file
        self.features = np.array([])
        self.labels = np.array([])
        self.is_labels_encod = False
        self.is_features_normal = False

    def download_if(self):
        if not os.path.isfile(self.cached_file):
            print('Downloading ' + self.src_datafile + ' ...')
            urlretrieve(self.src_datafile, self.cached_file)
            print('Download Finished')

    def load(self):
        """ Uncompress features and labels from a zip file :param file: The zip file to extract the data from"""
        features = []
        labels = []

        self.download_if()

        with ZipFile(self.cached_file) as zipf:
            print("Loading {}".format(self.cached_file))

            filenames_pbar = tqdm(zipf.namelist(), unit='files')    # progress bar

            # Get features and labels from all files
            for filename in filenames_pbar:
                # Check if the file is a directory
                if not filename.endswith('/'):
                    with zipf.open(filename) as image_file:
                        image = Image.open(image_file)
                        image.load()
                        # Load image data as 1 dimensional array. We're using float32 to save on memory space
                        feature = np.array(image, dtype=np.float32).flatten()

                    # Get the the letter from the filename.  This is the letter of the image.
                    label = os.path.split(filename)[1][0]

                    features.append(feature)
                    labels.append(label)

        self.features = np.array(features)
        self.labels = np.array(labels)

    def resample(self, size_limit):
        """Limit the amount of data to work with a docker container"""
        self.features, self.labels = resample(np.array(self.features), np.array(self.labels), n_samples=size_limit)

    def hot_one_encode_labels(self):
        """Turn labels into numbers and apply One-Hot Encoding"""
        if not self.is_labels_encod:
            encoder = LabelBinarizer()
            encoder.fit(self.labels)
            self.labels = encoder.transform(self.labels)

            # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
            self.labels = self.labels.astype(np.float32)
            self.is_labels_encod = True

    def assert_is_features_normal(self):
        assert self.is_features_normal

    def assert_is_labels_encod(self):
        assert self.is_labels_encod

    def train_test_split(self):
        """Get randomized datasets for training and validation"""
        train_features, valid_features, train_labels, valid_labels = train_test_split(self.features, self.labels, test_size=0.05, random_state=832289)
        return train_features, valid_features, train_labels, valid_labels

class TrainingImageData(TrainingData):
    def normalize_features(self):
        """
        Normalize the (greyscale) image data with Min-Max scaling to a range of [0.1, 0.9]
        :param image_data: The image data to be normalized
        :return: Normalized image data
        """

        if not self.is_features_normal:
            image_data = self.features

            range_min = 0.1
            range_max = 0.9
            x_max = np.max(image_data, axis=0)
            x_min = np.min(image_data, axis=0)
            x_std = (image_data - x_min) / (x_max - x_min)

            image_data_norm = x_std * (range_max - range_min) + range_min

            self.features = image_data_norm
            self.is_features_normal = True
