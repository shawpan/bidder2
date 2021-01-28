import tensorflow as tf
import pandas as pd
import json
import config
import math
import boto3
import os
import numpy as np

__LABEL_COLUMN__ = None
__DATA_STATS__ = None
__STATS_FILES__ = None


def download_stats_file_from_s3():
    """ Download the json value having statistics of the data set
    """
    CONFIG = config.get_config()
    s3 = boto3.resource('s3')
    s3.Bucket(
        CONFIG['S3_BUCKET']).download_file(
        CONFIG['DATA_STATS_FILE_KEY'],
        CONFIG['DATA_STATS_FILE'])


def set_stats_file(stat_files):
    """ Set the files that contain statistics
    """
    CONFIG = config.get_config()
    global __STATS_FILES__
    __STATS_FILES__ = stat_files


def set_stats(STATS=None):
    """ Set statistics to __DATA_STATS__
    """
    CONFIG = config.get_config()
    global __DATA_STATS__
    __DATA_STATS__ = STATS


def get_stats():
    """ Get statistics
    Returns:
        __DATA_STATS__(dict): Dictionary of statistics of the dataset
    """
    CONFIG = config.get_config()
    global __DATA_STATS__
    if __DATA_STATS__ is None:
        download_stats_file_from_s3()
        with open(CONFIG['DATA_STATS_FILE'], 'r') as f:
            __DATA_STATS__ = json.load(f)

    return __DATA_STATS__


def get_column_names():
    """ Get column names
    Returns:
        columns(list): List of column names
    """
    columns = get_stats()['columns']['all']
    return columns


def normalize(stats):
    """ Returns normalize function numeric value
    Args:
        stats(dict): Dictionary of statistics such as mean, std for the column
    Returns:
        fn(function): Function that does normalize operation
    """
    CONFIG = config.get_config()
#     fn = lambda x: tf.where(tf.greater(tf.to_float(x), CONFIG["EPSILON"]), tf.log(tf.to_float(x)), tf.to_float(x))
    def fn(x): return (tf.to_float(x) -
                       stats['mean']) / (stats['std'] + CONFIG["EPSILON"])
#     fn = lambda x: (tf.to_float(x) - stats['min']) / (stats['max'] - stats['min'] + CONFIG["EPSILON"])
    return fn


def get_feature_columns():
    """ Get feature columns
    Returns:
        features(list): List of tf.feature_column
    """
    CONFIG = config.get_config()
    stats = get_stats()

    numeric_features = []
    for key in CONFIG['PREDICT_BID_NUMERIC_COLUMNS']:
        numeric_features.append(
            tf.feature_column.numeric_column(
                key, dtype=tf.int64, normalizer_fn=normalize(
                    stats['stats'][key])))

    categorical_features = []
    for key in CONFIG['PREDICT_BID_CATEGORICAL_COLUMNS']:
        stat = stats['stats'][key]
        bucket_size = stat['unique']
        embedding_size = 6 * \
            math.ceil(
                stat['unique']**0.25) if stat['unique'] > 25 else stat['unique']
        categorical_features.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    key,
                    bucket_size,
                    tf.string
                ),
                embedding_size)
        )
    features = numeric_features + categorical_features

    return features


def get_feature_columns_for_bid_prediction():
    """ Get feature columns
    Returns:
        features(list): List of tf.feature_column
    """
    prepare_csv_column_list_for_bid_prediction()
    return get_feature_columns()


def get_label_column():
    """ Get label column
    Returns:
        __LABEL_COLUMN__(str): The name of the label column
    """
    return __LABEL_COLUMN__


def set_label_column(label_column):
    """ Set label column
    """
    global __LABEL_COLUMN__
    __LABEL_COLUMN__ = label_column


def _parse_line(line):
    """ Line parser
    Args:
        line(str): A string read from csv file
    Returns:
        (features, labels)(tuple): Tuple of feature and label values
    """
    CONFIG = config.get_config()
#     print(line)
    # Decode the line into its fields
    features = tf.decode_csv(
        line,
        field_delim=CONFIG['CSV_SEPARATOR'],
        record_defaults=config.get_default_values_for_csv_columns(),
        na_value='null')
    features = dict(zip(get_column_names(), features))
#     tf.print(features['deliveryid'])
    for column in get_column_names():
        if column not in get_label_column() + \
                [CONFIG['WEIGHT_COLUMN']] + CONFIG['PREDICT_BID_NUMERIC_COLUMNS'] + CONFIG['PREDICT_BID_CATEGORICAL_COLUMNS']:
            features.pop(column)

    if get_label_column() is None:
        return features

    # Separate the label from the features
    won_label = features.pop('won')
    targetbid_label = features.pop('targetbid')
#     labels = {
#         'won': won_label,
#         'targetbid': [ targetbid_label, won_label ]
#     }
    labels = [ targetbid_label, won_label ]
#     for label in get_label_column():
#         labels.append(features.pop(label))

    return features, labels


def prepare_csv_column_list_for_bid_prediction():
    """ Prepare feature columns selection
    """
    CONFIG = config.get_config()
    set_label_column(CONFIG['PREDICT_BID_LABELS'])


def train_input_fn_for_predict_bid(batch_size=1, num_epochs=1):
    """ Return input_fn for training dataset
    Args:
        batch_size(int): Number of instances per batch
        num_epochs(int): Number of training steps
    Returns:
        input_fn(function): input generator function for training dataset
    """
    CONFIG = config.get_config()
    filenames = []
    for path in CONFIG['PREDICT_BID_DATASET_TRAIN']:
        new_files = []
        if os.path.isdir(path):
            new_files = [os.path.join(path, p) for p in os.listdir(path)]
        else:
            new_files = [path]
        filenames = filenames + new_files
    prepare_csv_column_list_for_bid_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)


def validation_input_fn_for_predict_bid(batch_size=1, num_epochs=1):
    """ Return input_fn for validation dataset
    Args:
        batch_size(int): Number of instances per batch
        num_epochs(int): Number of training steps
    Returns:
        input_fn(function): input generator function for validation dataset
    """
    CONFIG = config.get_config()
    filenames = []
    for path in CONFIG['PREDICT_BID_DATASET_VAL']:
        new_files = []
        if os.path.isdir(path):
            new_files = [os.path.join(path, p) for p in os.listdir(path)]
        else:
            new_files = [path]
        filenames = filenames + new_files
    prepare_csv_column_list_for_bid_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)


def test_input_fn_for_predict_bid(filenames=None):
    """ Return input_fn for test dataset
    Args:
        filenames(list): List of filenames
    Returns:
        input_fn(function): input generator function for test dataset
    """
    CONFIG = config.get_config()
    if filenames is None:
        filenames = CONFIG['PREDICT_BID_DATASET_TEST']
    prepare_csv_column_list_for_bid_prediction()

    return csv_input_fn(
        filenames,
        batch_size=1,
        num_epochs=1,
        is_shuffle=False)


def csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True):
    """ Return dataset in batches from a CSV file
    Args:
        csv_path(str): CSV path file
        batch_size(int): Number of instances to return
        num_epochs(int): Number of training steps
        is_shuffle(bool): true if shuffle the dataset
    Returns:
        (features, labels)(tuple): Tuple of feature and label values
    """
    def read_file(filename):
        return tf.data.TextLineDataset(
            filename, compression_type='GZIP').skip(1)

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.flat_map(read_file)
#     dataset = dataset.apply(tf.data.experimental.map_and_batch(
#         _parse_line,
#         batch_size,
#         num_parallel_calls=4)
#     )
    dataset = dataset.map(_parse_line, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)

    # Shuffle, repeat, and batch the examples.
    if is_shuffle:
#         dataset = dataset.apply(
#             tf.data.experimental.shuffle_and_repeat(
#                 100, seed=42))
          dataset = dataset.shuffle(batch_size*100, seed=42)
          dataset = dataset.repeat(count=None)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels
