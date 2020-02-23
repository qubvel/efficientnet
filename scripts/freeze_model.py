# model conversation scrpit by Manthan Marvaniya(https://github.com/Manthan274)

import cv2
import os
import sys
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import backend as K
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..')))
from efficientnet.tfkeras import EfficientNetB0 as Net
tf.keras.backend.set_learning_phase(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

phi = 0 #change as per model file ans also change model import
num_classes = 1000 #change as per model file
model_path = 'model/path' #complete trained model(.h5) path
save_dir = "save/path"
input_sizes = [224, 240, 260, 300, 380]
image_size = input_sizes[phi]
print(image_size)

model = Net(weights=model_path, classes=num_classes)

frozen_graph = freeze_session(K.get_session(),  output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, save_dir, "frozen_model.pb", as_text=False)
