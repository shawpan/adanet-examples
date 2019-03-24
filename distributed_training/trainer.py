""" Train, Evaluate and Predict Winning Rate of a Bidding Request """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import shutil
import functools
import adanet
import adanet.examples.simple_dnn as simple_dnn

import data
import config
import pandas as pd
import numpy as np
import distribute

CONFIG = config.get_config()


OUTPUT_DIR = CONFIG['OUTPUT_DIR']

BATCH_SIZE = CONFIG['BATCH_SIZE'] # 512
NUM_EPOCHS = CONFIG['NUM_EPOCHS'] # 4000
EVAL_STEPS = CONFIG['EVAL_STEPS'] # 100
ADANET_LEARNING_RATE = CONFIG['ADANET_LEARNING_RATE']
ADANET_ITERATIONS = CONFIG['ADANET_ITERATIONS']
RANDOM_SEED = CONFIG['RANDOM_SEED']

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', default="ps", type=str, help='distribute task type')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('--clean', default=0, type=int, help='Clean previously trained data')
parser.add_argument('--is_test', default=0, type=int, help='Is Test')
parser.add_argument('--train_steps', default=NUM_EPOCHS, type=int,
                    help='number of training steps')

""" Get the model definition """
def get_model():
    return get_adanet_model()

def ensemble_architecture(result):
  """Extracts the ensemble architecture from evaluation results."""

  architecture = result["architecture/adanet/ensembles"]
  # The architecture is a serialized Summary proto for TensorBoard.
  summary_proto = tf.summary.Summary.FromString(architecture)
  return summary_proto.value[0].tensor.string_val[0]

def get_adanet_model():
    # Estimator configuration.
    # distribution_strategy = tf.contrib.distribute.MirroredStrategy()
    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    runConfig = tf.estimator.RunConfig(
        # train_distribute=distribution_strategy,
        # eval_distribute=distribution_strategy,
        session_config=session_config,
        save_checkpoints_steps=100,
        save_summary_steps=100,
        tf_random_seed=RANDOM_SEED)
    estimator = adanet.Estimator(
        model_dir = OUTPUT_DIR,
        # metric_fn=custom_metrics,
        # adanet_loss_decay=0.99,
        head=tf.contrib.estimator.multi_label_head(
            name="name",
            n_classes=len(CONFIG['LABELS']),
            # classes_for_class_based_metrics= [5,6]
        ),
        subnetwork_generator=simple_dnn.Generator(
            learn_mixture_weights=True,
            dropout=CONFIG["DROPOUT"],
            feature_columns=data.get_feature_columns(),
            optimizer=tf.train.AdamOptimizer(learning_rate=ADANET_LEARNING_RATE),
            seed=RANDOM_SEED),
        max_iteration_steps=NUM_EPOCHS // ADANET_ITERATIONS,
        evaluator=adanet.Evaluator(
            input_fn=lambda : data.validation_input_fn(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
            steps=EVAL_STEPS),
        config=runConfig)

    return estimator

""" Train the model """
def train_and_evaluate():
    estimator = get_model()
    serving_feature_spec = tf.feature_column.make_parse_example_spec(
      data.get_feature_columns())
    serving_input_receiver_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(serving_feature_spec))

    exporter = tf.estimator.BestExporter(
      name="name",
      serving_input_receiver_fn=serving_input_receiver_fn,
      exports_to_keep=1)

    train_spec = tf.estimator.TrainSpec(
                       input_fn = lambda : data.train_input_fn(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
                       max_steps = NUM_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(
                       input_fn = lambda : data.validation_input_fn(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS),
                       steps = EVAL_STEPS,
                       exporters=exporter,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("Finished training")

def main(argv):
    global BATCH_SIZE, NUM_EPOCHS
    args = parser.parse_args(argv[1:])
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.train_steps
    if args.is_test > 0:
        pass
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
        distribute.set_tf_config(args.task_type)
        if args.clean > 0:
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        train_and_evaluate()

if __name__ == '__main__':
    tf.app.run(main)
