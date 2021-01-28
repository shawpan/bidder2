from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import shutil
import functools
import os
import time
os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '3600000'
os.environ['S3_REQUEST_TIMEOUT'] = '3600000'

import bidding_data
import config
import pandas as pd
import numpy as np

OUTPUT_DIR = ''
BATCH_SIZE = 512
NUM_EPOCHS = 5000
EVAL_STEPS = 100
ADANET_LEARNING_RATE = 1e-05
ADANET_ITERATIONS = 2
RANDOM_SEED = 42
DROPOUT = 0.1
LAYER_NUMBER = 5
NODE_NUMBER = 512

parser = argparse.ArgumentParser()
# parser.add_argument('--task_type', default="ps", type=str, help='distribute task type')
parser.add_argument('--model_dir', default='', type=str, help='model dir')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('--clean', default=0, type=int, help='Clean previously trained data')
parser.add_argument('--is_test', default=0, type=int, help='Is Test')
parser.add_argument('--epochs', default=NUM_EPOCHS, type=int,
                    help='number of training steps')
parser.add_argument('--learning_rate', default=ADANET_LEARNING_RATE, type=float,
                    help='learning rate')
parser.add_argument('--dropout_rate', default=DROPOUT, type=float,
                    help='dropout rate')
parser.add_argument('--layer_number', default=LAYER_NUMBER, type=int,
                    help='layer number')
parser.add_argument('--node_number', default=NODE_NUMBER, type=int,
                    help='node number')
parser.add_argument('--config_file', default='config.json', type=str,
                    help='config file path')


def set_global_variables():
    global CONFIG, OUTPUT_DIR, BATCH_SIZE, NUM_EPOCHS, EVAL_STEPS
    global ADANET_LEARNING_RATE, ADANET_ITERATIONS, RANDOM_SEED, DROPOUT, LAYER_NUMBER, NODE_NUMBER
    
    CONFIG = config.get_config()
    
    OUTPUT_DIR = '{}/{}/'.format(CONFIG['OUTPUT_DIR_PREDICT_IMP'], int(time.time()) )
    BATCH_SIZE = CONFIG['PREDICT_IMP_BATCH_SIZE'] # 512
    NUM_EPOCHS = CONFIG['PREDICT_IMP_NUM_EPOCHS'] # 4000
    EVAL_STEPS = CONFIG['PREDICT_IMP_EVAL_STEPS'] # 100
    ADANET_LEARNING_RATE = CONFIG['PREDICT_IMP_ADANET_LEARNING_RATE']
    ADANET_ITERATIONS = CONFIG['PREDICT_IMP_ADANET_ITERATIONS']
    RANDOM_SEED = CONFIG['RANDOM_SEED']
    DROPOUT = CONFIG['PREDICT_IMP_DROPOUT']
    LAYER_NUMBER = 4
    NODE_NUMBER = 512

def get_weight_column():
    if CONFIG['WEIGHT_COLUMN']:
        return CONFIG['WEIGHT_COLUMN']
    return None

def get_feature_weights(features):
    if CONFIG['WEIGHT_COLUMN']:
        return features [CONFIG['WEIGHT_COLUMN'] ]
    return None

def custom_metrics(predictions, features, labels):
#     print(predictions)
#     print(labels)
    if len(CONFIG['PREDICT_IMP_LABELS']) == 1:
        predicted_labels = predictions['class_ids']
    else:
        predicted_labels = tf.round(predictions['probabilities'])
    metrics = {}
#     if len(labels) > 1:
#         metrics['accuracy'] = tf.metrics.accuracy(labels, predicted_labels, name="average")
    index = 0
    feature_weights = get_feature_weights(features)
    for imp_type in CONFIG['PREDICT_IMP_LABELS']:
        sliced_labels = tf.slice(labels,[0,index], [-1, 1])
        sliced_predicted_labels = tf.slice(predicted_labels, [0,index], [-1, 1])
        metrics['accuracy/'+imp_type] = tf.metrics.accuracy(sliced_labels, sliced_predicted_labels, weights=feature_weights, name=imp_type)
        metrics['auc_roc/'+imp_type] = tf.metrics.auc(sliced_labels, sliced_predicted_labels, weights=feature_weights, name=imp_type, curve='ROC', summation_method='careful_interpolation')
        metrics['auc_PR/'+imp_type] = tf.metrics.auc(sliced_labels, sliced_predicted_labels, weights=feature_weights, name=imp_type, curve='PR', summation_method='careful_interpolation')
        metrics['precision/'+imp_type] = tf.metrics.precision(sliced_labels, sliced_predicted_labels, weights=feature_weights, name=imp_type)
        metrics['recall/'+imp_type] = tf.metrics.recall(sliced_labels, sliced_predicted_labels, weights=feature_weights, name=imp_type)
        index = index + 1

    return metrics

def get_binary_classifier_model(hidden_units):
    runConfig = tf.estimator.RunConfig(
        save_checkpoints_steps=CONFIG['PREDICT_IMP_SAVE_CHECKPOINTS_STEPS'],
        save_summary_steps=CONFIG['PREDICT_IMP_SAVE_SUMMARY_STEPS'],
        tf_random_seed=RANDOM_SEED)
    
#     head=get_head()
    
#     estimator = tf.estimator.DNNLinearCombinedClassifier(
#         model_dir=OUTPUT_DIR,
#         linear_feature_columns=bidding_data.get_feature_columns_for_imp_prediction(),
#         linear_optimizer='Ftrl',
#         dnn_feature_columns=bidding_data.get_feature_columns_for_imp_prediction(),
#         dnn_optimizer=lambda: tf.train.AdamOptimizer(learning_rate=ADANET_LEARNING_RATE),
#         dnn_hidden_units=hidden_units,
#         dnn_activation_fn=tf.nn.relu,
#         dnn_dropout=DROPOUT,
#         n_classes=2,
#         weight_column=get_weight_column(),
#         config=runConfig,
#         batch_norm=True)
    
    estimator = tf.estimator.DNNClassifier(
        model_dir = OUTPUT_DIR,
        weight_column=get_weight_column(),
        batch_norm=True,
        hidden_units=hidden_units,
        dropout=DROPOUT,
        feature_columns=bidding_data.get_feature_columns_for_imp_prediction(),
        optimizer=lambda: tf.train.AdamOptimizer(learning_rate=ADANET_LEARNING_RATE),
        config=runConfig)
    
#     estimator = tf.contrib.estimator.add_metrics(estimator, custom_metrics)

    return estimator

""" Get the model definition """
def get_model():
    if len(CONFIG['PREDICT_IMP_LABELS']) > 1:
        return get_dnn_model([ NODE_NUMBER for i in range(LAYER_NUMBER) ])
    return get_binary_classifier_model([ NODE_NUMBER for i in range(LAYER_NUMBER) ])

def ensemble_architecture(result):
  """Extracts the ensemble architecture from evaluation results."""

  architecture = result["architecture/adanet/ensembles"]
  # The architecture is a serialized Summary proto for TensorBoard.
  summary_proto = tf.summary.Summary.FromString(architecture)
  return summary_proto.value[0].tensor.string_val[0]

def get_multi_label_head():
    return tf.contrib.estimator.multi_label_head(
            name="predict_imp",
            weight_column=get_weight_column(),
            n_classes=len(CONFIG['PREDICT_IMP_LABELS']),
            # classes_for_class_based_metrics= [5,6]    
        )

def get_binary_head():
    return tf.contrib.estimator.binary_classification_head(
            name="predict_imp",
            weight_column=get_weight_column()
        )

def get_head():
    if len(CONFIG['PREDICT_IMP_LABELS']) > 1:
        return get_multi_label_head()
    return get_binary_head()

def get_dnn_model(hidden_units):
    runConfig = tf.estimator.RunConfig(
        save_checkpoints_steps=CONFIG['PREDICT_IMP_SAVE_CHECKPOINTS_STEPS'],
        save_summary_steps=CONFIG['PREDICT_IMP_SAVE_SUMMARY_STEPS'],
        tf_random_seed=RANDOM_SEED)
    
    head=get_head()
    
    estimator = tf.estimator.DNNEstimator(
        head=head,
        model_dir = OUTPUT_DIR,
        hidden_units=hidden_units,
        dropout=DROPOUT,
        batch_norm=True,
        feature_columns=bidding_data.get_feature_columns_for_imp_prediction(),
        optimizer=tf.train.AdamOptimizer(learning_rate=ADANET_LEARNING_RATE),
        config=runConfig)
    
    estimator = tf.contrib.estimator.add_metrics(estimator, custom_metrics)

    return estimator

""" Train the model """
def train_and_evaluate():
    estimator = get_model()
    serving_feature_spec = tf.feature_column.make_parse_example_spec(
      bidding_data.get_feature_columns_for_imp_prediction())
    serving_input_receiver_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(serving_feature_spec))

    exporter = tf.estimator.BestExporter(
      name="predict_imp",
#       event_file_pattern='*.tfevents.*',
      serving_input_receiver_fn=serving_input_receiver_fn,
      exports_to_keep=2)

    train_spec = tf.estimator.TrainSpec(
                       input_fn = lambda : bidding_data.train_input_fn_for_predict_imp(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
                       max_steps = NUM_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(
                       input_fn = lambda : bidding_data.validation_input_fn_for_predict_imp(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
                       steps = EVAL_STEPS,
                       exporters=exporter,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10)
    print('S3_REQUEST_TIMEOUT = {}'.format( os.getenv('S3_REQUEST_TIMEOUT') ) )
    print('S3_REQUEST_TIMEOUT_MSEC = {}'.format( os.getenv('S3_REQUEST_TIMEOUT_MSEC') ) )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("Finished training")
    estimator.export_saved_model( 
        os.environ.get('SM_MODEL_DIR'), 
        serving_input_receiver_fn, 
        assets_extra=None, 
        as_text=False, 
        checkpoint_path=None 
    )

def main(argv):
    global BATCH_SIZE, NUM_EPOCHS, ADANET_LEARNING_RATE, DROPOUT, LAYER_NUMBER, NODE_NUMBER
    
    args = parser.parse_args(argv[1:])
#     os.environ['CONFIG_FILE'] = args.config_file
    set_global_variables()
    
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    ADANET_LEARNING_RATE = args.learning_rate
    DROPOUT = args.dropout_rate 
    LAYER_NUMBER = args.layer_number 
    NODE_NUMBER = args.node_number 
    
    if args.is_test > 0:
        pass
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
        if args.clean > 0:
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        train_and_evaluate()

if __name__ == '__main__':
    tf.app.run(main)
