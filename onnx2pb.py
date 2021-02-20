
#经测试，生成的模型有问题

import onnx
from onnx_tf.backend import prepare

def onnx2pb(onnx_input_path, pb_output_path):
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model

if __name__ == "__main__":
    #onnx_input_path = 'G:/koutu/portraitonnx/portraitonnx/portraitModel.onnx'
    #pb_output_path = 'G:/koutu/portraitonnx/portraitonnx/portraitModel.pb'
    onnx_input_path = 'G:/koutu/portraitonnx/portraitonnx/portraitModel.onnx'
    pb_output_path = 'G:/koutu/portraitonnx/portraitonnx/portraitModel.pb'
    onnx2pb(onnx_input_path, pb_output_path)
