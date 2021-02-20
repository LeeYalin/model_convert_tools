# coding: utf-8

# import io
import torch
import torch.onnx
import sys
sys.path.append('F:/koutu/PortraitNet-master_continue_pth/model')
from model_mobilenetv2_seg_small import MobileNetV2
from model_BiSeNet import BiSeNet
from model_enet import ENet
from torch.nn import DataParallel

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
def test():
  model = MobileNetV2()
  DEVICES = list(range(0, 4))
  model = torch.nn.DataParallel(model, DEVICES)
  #model = BiSeNet(2)
  #model = ENet(2)
  pthfile = 'C:/Users/Administrator/Desktop/222/shuban384288/model_best.pth'
  #pthfile = ''
  checkpoint = torch.load(pthfile, map_location='cpu')
  # try:

  #   loaded_model.eval()
  # except AttributeError as error:
  #   print(error)
  #print(checkpoint['state_dict'])
  model.load_state_dict(checkpoint['state_dict'],False)
  # model = model.to(device)
 
  #data type nchw
  #dummy_input1 = torch.randn(1, 3, 288, 384)
  dummy_input1 = torch.randn(1, 3, 384, 288)
  input_names = [ "actual_input_1"]
  output_names = ["output1"]
  

  
  torch.onnx.export(model.module, dummy_input1, "v3.onnx", verbose=True, input_names=input_names,
                    output_names=output_names,keep_initializers_as_inputs=True)
  # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)

 
if __name__ == "__main__":
 test()
