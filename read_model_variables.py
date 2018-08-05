import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

flags = tf.flags.FLAGS
tf.flags.DEFINE_string(
    'model_ckpt_path', '/home/yuhaitao/code/xuelang_1/model/model.ckpt-480', 'path to checkpoint')

reader = pywrap_tensorflow.NewCheckpointReader(flags.model_ckpt_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print("tensor_name:", key)
    # print(reader.get_tensor(key))
