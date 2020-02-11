"""Adopted import_pb_to_tensorboard for use with modules from tensorflow hub

Reference:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py

Example:
    python import_hub_to_tensorboard.py --log_dir /tmp/tensorflow_logdir --model_spec https://tfhub.dev/google/imagenet/pnasnet_large/classification/2  # noqa
"""

from __future__ import absolute_import, division, print_function

import argparse
import sys

import tensorflow_hub as hub
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.platform import app
from tensorflow.python.summary import summary

try:
    from tensorflow.contrib.tensorrt.ops.gen_trt_engine_op import *  # noqa
except ImportError:
    pass


def import_to_tensorboard(model_spec, log_dir):
    """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.

    Args:
        model_spec: The tensorflow hub model to visualize.
        log_dir: The location for the Tensorboard log to begin visualization
            from.

    Usage:
        Call this function with a tensorflow_hub and desired log directory.
        Launch Tensorboard by pointing it to the log directory.
        View the tensorflow hub model as a graph.
    """
    with session.Session(graph=ops.Graph()) as sess:

        hub.Module(model_spec)  # adds graph to session

        pb_visual_writer = summary.FileWriter(log_dir)
        pb_visual_writer.add_graph(sess.graph)
        print(
            "Model Imported. Visualize by running: "
            "tensorboard --logdir={}".format(log_dir)
        )


def main(unused_args):
    import_to_tensorboard(FLAGS.model_spec, FLAGS.log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_spec",
        type=str,
        default="",
        required=True,
        help="The tensorflow hub model url to visualize."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        required=True,
        help="The location for the Tensorboard log to begin visualization."
    )
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
