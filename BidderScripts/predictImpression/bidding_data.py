import tensorflow as tf
import pandas as pd
import json
import config
import math
import boto3
import os
import numpy as np

CONFIG = None

LABEL_COLUMN = None
DATA_STATS = None
STATS_FILES = None

def download_stats_file_from_s3():
    CONFIG = config.get_config()
    s3 = boto3.resource('s3')
    s3.Bucket(CONFIG['S3_BUCKET']).download_file(CONFIG['DATA_STATS_FILE_KEY'], CONFIG['DATA_STATS_FILE'])
    
def set_stats_file(stat_files):
    CONFIG = config.get_config()
    global STATS_FILES
    STATS_FILES = stat_files

def set_stats(STATS=None):
    CONFIG = config.get_config()
    global DATA_STATS
    DATA_STATS = STATS
    
def get_stats():
    CONFIG = config.get_config()
    global DATA_STATS
    if DATA_STATS is None:
        download_stats_file_from_s3()
        with open(CONFIG['DATA_STATS_FILE'], 'r') as f:
            DATA_STATS = json.load(f)
            
    return DATA_STATS

def get_column_names():
    columns = get_stats()['columns']['all']

    return columns

def normalize(stats):
    CONFIG = config.get_config()
#     fn = lambda x: tf.where(tf.greater(tf.to_float(x), CONFIG["EPSILON"]), tf.log(tf.to_float(x)), tf.to_float(x))
    fn = lambda x: (tf.to_float(x) - stats['mean']) / (stats['std'] + CONFIG["EPSILON"])
#     fn = lambda x: (tf.to_float(x) - stats['min']) / (stats['max'] - stats['min'] + CONFIG["EPSILON"])
    return fn

def get_feature_columns_linear():
    CONFIG = config.get_config()
    stats = get_stats()
    
    numeric_features = []
    for key in CONFIG['PREDICT_IMP_NUMERIC_COLUMNS_LINEAR']:
        numeric_features.append(
            tf.feature_column.numeric_column(key, dtype=tf.int64, normalizer_fn = normalize(stats['stats'][key]))
        )

    categorical_features = []
    for key in CONFIG['PREDICT_IMP_CATEGORICAL_COLUMNS_LINEAR']:
        stat = stats['stats'][key]
        embedding_size = 6.0 * math.ceil(stat['unique']**0.25) if stat['unique'] > 25 else stat['unique']
        categorical_features.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    key,
                    stat['unique'],
                    tf.string
                ),
                embedding_size + 1)
        )
    features = numeric_features + categorical_features

    return features

def get_feature_columns_dnn():
    CONFIG = config.get_config()
    stats = get_stats()
    
    numeric_features = []
    for key in CONFIG['PREDICT_IMP_NUMERIC_COLUMNS_DNN']:
        numeric_features.append(
            tf.feature_column.numeric_column(key, dtype=tf.int64, normalizer_fn = normalize(stats['stats'][key]))
        )

    categorical_features = []
    for key in CONFIG['PREDICT_IMP_CATEGORICAL_COLUMNS_DNN']:
        stat = stats['stats'][key]
        embedding_size = 6.0 * math.ceil(stat['unique']**0.25) if stat['unique'] > 25 else stat['unique']
        categorical_features.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    key,
                    stat['unique'],
                    tf.string
                ),
                embedding_size + 1)
        )
    features = numeric_features + categorical_features

    return features

def get_feature_columns():
    CONFIG = config.get_config()
    stats = get_stats()
    
    numeric_features = []
    for key in CONFIG['PREDICT_IMP_NUMERIC_COLUMNS']:
        numeric_features.append(
            tf.feature_column.numeric_column(key, dtype=tf.int64, normalizer_fn = normalize(stats['stats'][key]))
        )

    categorical_features = []
    for key in CONFIG['PREDICT_IMP_CATEGORICAL_COLUMNS']:
        stat = stats['stats'][key]
        bucket_size = stat['unique']
        embedding_size = 6 * math.ceil(stat['unique']**0.25) if stat['unique'] > 25 else stat['unique']
        categorical_features.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    key,
                    bucket_size ,
                    tf.string
                ),
                embedding_size)
        )
    features = numeric_features + categorical_features

    return features

def get_feature_columns_for_imp_prediction():
    prepare_csv_column_list_for_imp_prediction()
    return get_feature_columns()

def get_feature_columns_for_linear_imp_prediction():
    prepare_csv_column_list_for_imp_prediction()
    return get_feature_columns_linear()

def get_feature_columns_for_dnn_imp_prediction():
    prepare_csv_column_list_for_imp_prediction()
    return get_feature_columns_dnn()

def get_label_column():
    return LABEL_COLUMN

def set_label_column(label_column):
    global LABEL_COLUMN
    LABEL_COLUMN = label_column
    
""" Parse the CSV file of bidding data
Arguments:
    line: string, string of comma separated instance values
"""
def _parse_line(line):
    CONFIG = config.get_config()
#     print(line)
    # Decode the line into its fields
    features = tf.decode_csv(line, field_delim=CONFIG['CSV_SEPARATOR'], record_defaults=config.get_default_values_for_csv_columns(), na_value='null')
    features = dict(zip(get_column_names(), features))
#     tf.print(features['deliveryid'])
    for column in get_column_names():
        if column not in get_label_column() + [CONFIG['WEIGHT_COLUMN']] + CONFIG['PREDICT_IMP_NUMERIC_COLUMNS'] + CONFIG['PREDICT_IMP_CATEGORICAL_COLUMNS']:
            features.pop(column)
        
    if get_label_column() is None:
        return features
    
    # Separate the label from the features
    labels = []
    for label in get_label_column():
        labels.append(features.pop(label))
    
    return features, labels

def prepare_csv_column_list_for_imp_prediction():
    CONFIG = config.get_config()
    set_label_column(CONFIG['PREDICT_IMP_LABELS'])

""" IMP prediction """
def train_input_fn_for_predict_imp(batch_size=1, num_epochs=1):
    CONFIG = config.get_config()
    filenames = []
    for path in CONFIG['PREDICT_IMP_DATASET_TRAIN']:
        new_files = []
        if os.path.isdir(path):
            new_files = [ os.path.join(path, p) for p in os.listdir(path) ]
        else:
            new_files = [ path ]
        filenames = filenames + new_files
    prepare_csv_column_list_for_imp_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)

def validation_input_fn_for_predict_imp(batch_size=1, num_epochs=1):
    CONFIG = config.get_config()
    filenames = []
    for path in CONFIG['PREDICT_IMP_DATASET_VAL']:
        new_files = []
        if os.path.isdir(path):
            new_files = [ os.path.join(path, p) for p in os.listdir(path) ]
        else:
            new_files = [ path ]
        filenames = filenames + new_files
    prepare_csv_column_list_for_imp_prediction()

    return csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True)

def test_input_fn_for_predict_imp(filenames=None):
    CONFIG = config.get_config()
    if filenames is None:
        filenames = CONFIG['PREDICT_IMP_DATASET_TEST']
    prepare_csv_column_list_for_imp_prediction()

    return csv_input_fn(filenames, batch_size=1, num_epochs=1, is_shuffle=False)

""" Return dataset in batches from a CSV file
Arguments:
    csv_path: string, CSV path file
    batch_size: integer, Number of instances to return
Returns:
    dataset tensor parsed from csv
"""
def csv_input_fn(filenames, batch_size, num_epochs, is_shuffle=True):
    def read_file(filename):
#         tf.print('Chosen file')
#         tf.print(filename)
        return tf.data.TextLineDataset(filename, compression_type='GZIP').skip(1)
    
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    
#     if is_shuffle:
#         dataset = dataset.shuffle(1000, seed=42)
    
    dataset = dataset.flat_map(read_file)
    
        # Shuffle, repeat, and batch the examples.
    if is_shuffle:
#         dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(100, seed=42))
        dataset = dataset.shuffle(batch_size*100, seed=42)
        dataset = dataset.repeat(count=None)

#     dataset = dataset.apply(tf.data.experimental.map_and_batch(_parse_line, batch_size, num_parallel_calls=4))
    dataset = dataset.map(_parse_line, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
        
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels
