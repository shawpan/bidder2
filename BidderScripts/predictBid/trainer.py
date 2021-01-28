from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

# tf.enable_eager_execution()

import shutil
import functools
import os
import time
import sys

import pandas as pd
import numpy as np

import config
import losses
import bidding_data

__OUTPUT_DIR__ = ''
__BATCH_SIZE__ = 512
__NUM_EPOCHS__ = 1000
__EVAL_STEPS__ = 100
__LEARNING_RATE__ = 1e-5
__RANDOM_SEED__ = 42
__DROPOUT__ = 0.0
__LAYER_NUMBER__ = 4
__NODE_NUMBER__ = 512

# os.environ['S3_REQUEST_TIMEOUT_MSEC'] = '3600000'
# os.environ['S3_REQUEST_TIMEOUT'] = '3600000'

parser = argparse.ArgumentParser()
# parser.add_argument('--task_type', default="ps", type=str, help='distribute task type')
parser.add_argument('--model_dir', default='', type=str, help='model dir')
parser.add_argument(
    '--batch_size',
    default=__BATCH_SIZE__,
    type=int,
    help='batch size')
parser.add_argument(
    '--clean',
    default=0,
    type=int,
    help='Clean previously trained data')
parser.add_argument('--is_test', default=0, type=int, help='Is Test')
parser.add_argument('--epochs', default=__NUM_EPOCHS__, type=int,
                    help='number of training steps')
parser.add_argument(
    '--learning_rate',
    default=__LEARNING_RATE__,
    type=float,
    help='learning rate')
parser.add_argument('--dropout_rate', default=__DROPOUT__, type=float,
                    help='dropout rate')
parser.add_argument('--layer_number', default=__LAYER_NUMBER__, type=int,
                    help='layer number')
parser.add_argument('--node_number', default=__NODE_NUMBER__, type=int,
                    help='node number')
parser.add_argument('--config_file', default='config.json', type=str,
                    help='config file path')


def set_global_variables():
    global CONFIG, __OUTPUT_DIR__, __BATCH_SIZE__, __NUM_EPOCHS__, __EVAL_STEPS__
    global __LEARNING_RATE__, __RANDOM_SEED__, __DROPOUT__, __LAYER_NUMBER__, __NODE_NUMBER__

    CONFIG = config.get_config()

    __OUTPUT_DIR__ = '{}/{}/'.format(
        CONFIG['OUTPUT_DIR_PREDICT_BID'], int(
            time.time()))
    __BATCH_SIZE__ = CONFIG['PREDICT_BID_BATCH_SIZE']  # 512
    __NUM_EPOCHS__ = CONFIG['PREDICT_BID_NUM_EPOCHS']  # 4000
    __EVAL_STEPS__ = CONFIG['PREDICT_BID_EVAL_STEPS']  # 100
    __LEARNING_RATE__ = CONFIG['PREDICT_BID_LEARNING_RATE']
    __RANDOM_SEED__ = CONFIG['RANDOM_SEED']
    __DROPOUT__ = CONFIG['PREDICT_BID_DROPOUT']
    __LAYER_NUMBER__ = CONFIG['PREDICT_BID_LAYER_NUMBER']
    __NODE_NUMBER__ = CONFIG['PREDICT_BID_NODE_NUMBER']

def custom_metrics(predictions, features, labels):
    wons = tf.reshape(labels[:, 1], [-1, 1])
    dist = tf.distributions.Normal(loc=0.0, scale=1.0)
    
#     sess = tf.compat.v1.Session()
#     with sess.as_default():
#         print_op = tf.print("labels: ", labels, "predictions: ", predictions, "features: ", features)
#         with tf.control_dependencies([print_op]):
#             metric = tf.metrics.mean(values=[3.,3.,3.], name='custom_metric')
#     return { 'custom_metric': metric }
    
    target_bids = labels[:, 0]
    predicted_bids = predictions['predictions'][:,0] #tf.exp(logits[:, 0])

    e = tf.reshape((predicted_bids - target_bids), [-1,1])
    
    # error on W
    error_on_W = -dist.log_prob(-e)
    # error on L
    error_on_L = -dist.log_cdf(e)
    
    bid_error_on_lost = error_on_L * (1. - wons)
    bid_error_on_won = error_on_W * wons
    
    return {
        'loss_on_L': tf.metrics.mean(
            bid_error_on_lost*[1, 0],
            name='loss_on_L'
        ),
        'loss_on_W': tf.metrics.mean(
            bid_error_on_won*[1, 0],
            name='loss_on_W'
        )
    }
#     # P < L
#     error_of_P_less_than_L = tf.square(tf.clip_by_value(e, tf.float32.min, 0.))*[1,0]
#     # P(Et <= E)
#     error_of_probability_for_Et_less_than_equal_to_E_on_L = -dist.log_cdf(tf.clip_by_value(e, 0., tf.float32.max))*[1,0]

#     # (P - W)^2
#     error_on_W = tf.square(e)*[1,0]
    
#     return {
#         'loss_P_less_than_L': tf.metrics.mean(
#             error_of_P_less_than_L,
#             name='loss_P_less_than_L'
#         ),
#         'loss_of_probability_for_Et_less_than_equal_to_E_on_L': tf.metrics.mean(
#             error_of_probability_for_Et_less_than_equal_to_E_on_L,
#             name='loss_of_probability_for_Et_less_than_equal_to_E_on_L'
#         ),
#         'loss_on_W': tf.metrics.mean(
#             error_on_W,
#             name='loss_on_W'
#         )
#     }
    
def get_weight_column():
    if CONFIG['WEIGHT_COLUMN']:
        return CONFIG['WEIGHT_COLUMN']
    return None


def get_feature_weights(features):
    if CONFIG['WEIGHT_COLUMN']:
        return features[CONFIG['WEIGHT_COLUMN']]
    return None

def get_run_config():
    return tf.estimator.RunConfig(
        save_checkpoints_steps=CONFIG['PREDICT_BID_SAVE_CHECKPOINTS_STEPS'],
        save_summary_steps=CONFIG['PREDICT_BID_SAVE_SUMMARY_STEPS'],
        tf_random_seed=__RANDOM_SEED__
    )

def get_head():
#     won_head = tf.contrib.estimator.binary_classification_head(
#         weight_column=get_weight_column(),
#         loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
#         name='won'
#     )
    bid_head = tf.contrib.estimator.regression_head(
#         weight_column=get_weight_column(),
        label_dimension=2,
        loss_fn=losses.first_price_auction_loss_clm,
        inverse_link_fn=tf.exp,
        name='targetbid',
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
    )
    return bid_head
#     return tf.contrib.estimator.multi_head([ won_head, bid_head ])

def get_wide_and_deep_model(hidden_units):
    estimator = tf.estimator.DNNLinearCombinedEstimator(
        get_head(),
        model_dir=__OUTPUT_DIR__,
        linear_feature_columns=bidding_data.get_feature_columns_for_bid_prediction(),
        linear_optimizer='Ftrl',
        dnn_feature_columns=bidding_data.get_feature_columns_for_bid_prediction(),
        dnn_optimizer=lambda: tf.train.AdamOptimizer(
            learning_rate=__LEARNING_RATE__),
        dnn_hidden_units=hidden_units,
        dnn_dropout=__DROPOUT__,
        config=get_run_config()
    )
    estimator = tf.contrib.estimator.add_metrics(estimator, custom_metrics)
    
    return estimator

def get_model():
    """ Get the model definition """
    return get_wide_and_deep_model(
        [__NODE_NUMBER__ for i in range(__LAYER_NUMBER__)])


def train_and_evaluate():
    """ Train the model """
    estimator = get_model()
    serving_feature_spec = tf.feature_column.make_parse_example_spec(
        bidding_data.get_feature_columns_for_bid_prediction())
    serving_input_receiver_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(serving_feature_spec))

    exporter = tf.estimator.BestExporter(
        name="predict_bid",
        #       event_file_pattern='*.tfevents.*',
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=2)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: bidding_data.train_input_fn_for_predict_bid(
            batch_size=__BATCH_SIZE__,
            num_epochs=__NUM_EPOCHS__),
        max_steps=__NUM_EPOCHS__)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: bidding_data.validation_input_fn_for_predict_bid(
            batch_size=__BATCH_SIZE__, num_epochs=__NUM_EPOCHS__),
        steps=__EVAL_STEPS__,
        exporters=exporter,
        start_delay_secs=1,  # start evaluating after N seconds
        throttle_secs=10)
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
    tf.logging.set_verbosity(tf.logging.INFO)
    global __BATCH_SIZE__, __NUM_EPOCHS__, __LEARNING_RATE__, __DROPOUT__, __LAYER_NUMBER__, __NODE_NUMBER__

    args = parser.parse_args(argv[1:])
    set_global_variables()

    __BATCH_SIZE__ = args.batch_size
    __NUM_EPOCHS__ = args.epochs
    __LEARNING_RATE__ = args.learning_rate
    __DROPOUT__ = args.dropout_rate
    __LAYER_NUMBER__ = args.layer_number
    __NODE_NUMBER__ = args.node_number

    if args.is_test > 0:
        pass
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
        if args.clean > 0:
            shutil.rmtree(__OUTPUT_DIR__, ignore_errors=True)
        train_and_evaluate()


if __name__ == '__main__':
    tf.app.run(main)
