"""Exports trained model to TensorFlow frozen graph."""

import os
import tensorflow as tf
import time
from tensorflow.python.tools import freeze_graph
#from tf_pose.network_openpose import get_network
#from tf_pose.network_efficient_b0_orgin import EfficientNetwork
#5484864236
#from tf_pose.network_efficient_b0_3stg import EfficientNetwork
#3564554816
from tf_pose.network_mobilenet_v2 import Mobilenetv2Network
#from tf_pose.network_efficient_b0_3stg import EfficientNetwork
from tf_pose.pose_dataset_process_14 import get_dataflow_batch, DataFlowToQueue, PoseDataset
from tf_pose.pose_augment import set_network_input_wh, set_network_scale
from tf_pose.common import get_sample_images
from tf_pose.networks1 import get_network


slim = tf.contrib.slim
flags = tf.app.flags

FLAGS = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
flags.DEFINE_string('checkpoint_path',
                    '/data/sharedata/xiangjing/tf-pose-estimation-master_efficient_b0/models/mobilenet_v2_w1.0_r1.0/train/model_latest-97002', 'Checkpoint path')
#flags.DEFINE_string('checkpoint_path',
#                    '/data/sharedata/xiangjing/tf-pose-estimation-master_efficient_b0/models/ai_train_3stg_312x208/train/model_latest-760004', 'Checkpoint path')


#flags.DEFINE_string('export_path',
#                    '/data/ftpdata/download-modle/efficientb0_pose_256_256_0910.pb',
#                    'Path to output Tensorflow frozen graph.')
flags.DEFINE_string('export_path',
                    '/data/sharedata/xiangjing/tf-pose-estimation-master_efficient_b0/models/mobilenet_v2_w1.0_r1.0/train/model_latest-97002.pb',
                    'Path to output Tensorflow frozen graph.')


#flags.DEFINE_string('resize', '256x256',
#                           'Crop size height.')
flags.DEFINE_string('resize', '224x224',
                           'Crop size height.')

# Input name of the exported model.
_INPUT_NAME = 'truediv'

# Output name of the exported model.
_OUTPUT_NAME = 'Openpose/concat_stage7'

input_width = 224
input_height = 224

def main(unused_argv):

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Prepare to export model to: %s', FLAGS.export_path)


  with tf.Graph().as_default() as graph:
    trainable = tf.constant(False,tf.bool)
    net_input = tf.placeholder(tf.float32, [1, input_height, input_width, 3], name=_INPUT_NAME)
    net = Mobilenetv2Network({'image': net_input},  conv_width=1.0, conv_width2=1.0, trainable=trainable)
    #net = EfficientNetwork({'image':net_input},trainable=trainable)
    #net, pretrain_path, last_layer = get_network('efficient_b0', q_inp_split_id, trainable=istrain)
    #outtensor = net.layers['concat_stage7']
    outtensor = net.layers['concat_stage7']
    semantic_predictions = tf.identity(outtensor, name=_OUTPUT_NAME)
    outtensor = tf.transpose(outtensor,[0,3,1,2],name = 'Openpose/nchw_concat_stage7')
    #net, _, _ = get_network(net_input,is_training=False)
    #semantic_predictions = tf.identity(network, name=_OUTPUT_NAME)
#tf.model_variables()
    saver = tf.train.Saver()
    #os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    tf.gfile.MakeDirs(os.path.dirname(FLAGS.export_path))
    freeze_graph.freeze_graph_with_def_protos(
        tf.get_default_graph().as_graph_def(add_shapes=True),
        saver.as_saver_def(),
        FLAGS.checkpoint_path,
        'Openpose/nchw_concat_stage7',
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=FLAGS.export_path,
        clear_devices=True,
        initializer_nodes=None)

    #float_opers = tf.profiler.ProfileOptionBuilder.float_operation()
    #flops = tf.profiler.profile(None, cmd='graph', options=float_opers)
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOP = ', flops)

if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('export_path')
  tf.app.run()
