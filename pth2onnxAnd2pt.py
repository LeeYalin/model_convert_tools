import torch
import os.path as osp
# An instance of your model.
# import settings
#from network_pt import Efficientnet_EMANet_EdgeV1
# from network_pt import Efficientnet_EMANet_EdgeV2
# from network_model_pt import Efficientnet_EMANet_EdgeV7
# from network_model_pt import Efficientnet_EMANet_EdgeV7_v1
from model_pytorch import CNNnet
from pt_mv3 import MobileNetV3_Small
from ghost_net import ghost_net
#from torch.nn import DataParallel
def load_checkpoints(name):
    #3net_model = Efficientnet_EMANet_EdgeV1(2)
    #DEVICES = list(range(0, 4))
    #net_model = MobileNetV3_Small()
    #net_model = CNNnet()
    net_model = ghost_net()
    #net_model = DataParallel(net_model, DEVICES)
    ckp_path = osp.join(name)
    try:
        obj = torch.load(ckp_path,
                         map_location=lambda storage, loc: storage)#.cuda())
    except FileNotFoundError:
        return

    #dict_model = {}
    #key = list(obj.keys())

    net_model.load_state_dict(obj)
    net_model.eval()
    inputs = torch.randn(1, 1, 128, 128)
    #inputs = torch.randn(1, 1, 8192, 8192)
    #inputs =  torch.randn(1,3,513,513)
    # outputs = torch.rand(1, 2, 512, 512)
    # torch.onnx.export(net_model,inputs,"./output/Efficientnet_EMANet_EdgeV1.onnx")

    #traced_script_module = torch.jit.trace(net_model, inputs)
    #traced_script_module.save("G:/grab/testDepthPoseDemp/testDepthPoseDemp/CNN_model001002_102000.pt")
    torch_out = torch.onnx._export(net_model, inputs, "G:/grab/models_onnx/Ghost_0001_sit_csxq_size5__loss6.272e-06_iter5854.onnx", export_params=True)
load_checkpoints("C:/Users/Administrator/Desktop/222/two_auxiliary_losses/5wan_from0_to_3wanbackgroung2000_to9wan/Ghost_0001_sit_csxq_size5__loss6.272e-06_iter5854.pth")
