import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torchvision.models import VGG16_Weights
from copy import deepcopy
from torchinfo import summary


def init_weights(net, init_type='normal', gain=0.02):
    net = net

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'pretrained':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None and init_type != 'pretrained':
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02):
    for root_child in net.children():
        if hasattr(root_child, 'need_initialization'):
            for children in root_child.children():
                if children in root_child.need_initialization:
                    init_weights(children, init_type, gain=init_gain)
                else:
                    init_weights(children, "pretrained", gain=init_gain)  # for batchnorms
        else:
            # 如果没有 need_initialization 属性，对整个模块进行初始化
            init_weights(root_child, init_type, gain=init_gain)
    return net


def define_FuseNet(num_labels, init_type='xavier', init_gain=0.02, mode='I'):
    net = FusenetGenerator(num_labels, mode=mode)
    return init_net(net, init_type, init_gain)


def define_branches(num_labels, init_type='xavier', init_gain=0.02, mode='RGB'):
    net = FusenetBranchGenerator(num_labels, mode=mode)
    return init_net(net, init_type, init_gain)


##############################################################################
# Classes
##############################################################################

def VGG16_initializator():
    layer_names = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2",
                   "conv4_3", "conv5_1", "conv5_2", "conv5_3"]
    layers = list(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features.children())
    layers = [x for x in layers if isinstance(x, nn.Conv2d)]
    layer_dic = dict(zip(layer_names, layers))
    return layer_dic


def make_layers_from_names(names, model_dic, bn_dim, existing_layer=None):
    layers = []
    if existing_layer is not None:
        layers = [existing_layer, nn.BatchNorm2d(bn_dim, momentum=0.1), nn.ReLU(inplace=True)]
    for name in names:
        layers += [deepcopy(model_dic[name]), nn.BatchNorm2d(bn_dim, momentum=0.1), nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


def make_layers_from_size(sizes):
    layers = []
    for size in sizes:
        layers += [nn.Conv2d(size[0], size[1], kernel_size=3, padding=1), nn.BatchNorm2d(size[1], momentum=0.1),
                   nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class FusenetGenerator(nn.Module):
    def __init__(self, num_labels, mode='I'):
        super(FusenetGenerator, self).__init__()
        batchNorm_momentum = 0.1  # TODO:make param

        self.need_initialization = []  # modules that need initialization
        model_dic = VGG16_initializator()
        self.mode = mode
        rgb_enc, depth_enc, rgbd_enc, rgbd_dec, rgb_dec, depth_dec = False, False, False, False, False, False

        if mode == 'I':
            rgb_enc = True
            depth_enc = True
            rgbd_dec = True
            self.after_fusion_identity = nn.Identity()
            self.hook_names = ['after_fusion_identity', 'CBR5_RGBD_DEC']
        elif mode == 'E':
            rgbd_enc = True
            rgbd_dec = True
            self.after_fusion_identity = nn.Identity()
            self.hook_names = ['after_fusion_identity', 'CBR5_RGBD_ENC']
        elif mode == 'L':
            rgb_enc = True
            depth_enc = True
            rgb_dec = True
            depth_dec = True
            self.cls_head = nn.Conv2d(82, num_labels, kernel_size=1)
            self.concat_identity = nn.Identity()
            self.hook_names = ['concat_identity']
        else:
            raise ValueError(f"Unsupported mode: {mode}. Please choose one mode in ['I', 'E', 'L'].")

        if rgb_enc:
            ##### RGB ENCODER ####
            self.CBR1_RGB_ENC = make_layers_from_names(["conv1_1", "conv1_2"], model_dic, 64)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR2_RGB_ENC = make_layers_from_names(["conv2_1", "conv2_2"], model_dic, 128)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR3_RGB_ENC = make_layers_from_names(["conv3_1", "conv3_2", "conv3_3"], model_dic, 256)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout3 = nn.Dropout(p=0.4)

            self.CBR4_RGB_ENC = make_layers_from_names(["conv4_1", "conv4_2", "conv4_3"], model_dic, 512)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout4 = nn.Dropout(p=0.4)

            self.CBR5_RGB_ENC = make_layers_from_names(["conv5_1", "conv5_2", "conv5_3"], model_dic, 512)

            # after the fusion or the last layer in mode L
            self.dropout5 = nn.Dropout(p=0.4)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        if depth_enc:
            ##### Depth ENCODER ####
            feats_depth = list(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features.children())
            avg = torch.mean(feats_depth[0].weight.data, dim=1)
            avg = avg.unsqueeze(1)

            conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            conv11d.weight.data = avg

            self.CBR1_DEPTH_ENC = make_layers_from_names(["conv1_2"], model_dic, 64, conv11d)
            self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR2_DEPTH_ENC = make_layers_from_names(["conv2_1", "conv2_2"], model_dic, 128)
            self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR3_DEPTH_ENC = make_layers_from_names(["conv3_1", "conv3_2", "conv3_3"], model_dic, 256)
            self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout3_d = nn.Dropout(p=0.4)

            self.CBR4_DEPTH_ENC = make_layers_from_names(["conv4_1", "conv4_2", "conv4_3"], model_dic, 512)
            self.pool4_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout4_d = nn.Dropout(p=0.4)

            self.CBR5_DEPTH_ENC = make_layers_from_names(["conv5_1", "conv5_2", "conv5_3"], model_dic, 512)
            self.pool5_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout5_d = nn.Dropout(p=0.4)

        if rgbd_enc:
            ##### RGB-D ENCODER ####
            feats_RGBD = list(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features.children())
            org = feats_RGBD[0].weight.data
            avg = torch.mean(feats_RGBD[0].weight.data, dim=1).unsqueeze(1)
            res = torch.cat((org, avg), 1)

            conv14d = nn.Conv2d(4, 64, kernel_size=3, padding=1)
            conv14d.weight.data = res

            self.CBR1_RGBD_ENC = make_layers_from_names(["conv1_2"], model_dic, 64, conv14d)
            self.pool1_rgbd = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR2_RGBD_ENC = make_layers_from_names(["conv2_1", "conv2_2"], model_dic, 128)
            self.pool2_rgbd = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR3_RGBD_ENC = make_layers_from_names(["conv3_1", "conv3_2", "conv3_3"], model_dic, 256)
            self.pool3_rgbd = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout3_rgbd = nn.Dropout(p=0.4)

            self.CBR4_RGBD_ENC = make_layers_from_names(["conv4_1", "conv4_2", "conv4_3"], model_dic, 512)
            self.pool4_rgbd = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout4_rgbd = nn.Dropout(p=0.4)

            self.CBR5_RGBD_ENC = make_layers_from_names(["conv5_1", "conv5_2", "conv5_3"], model_dic, 512)
            self.pool5_rgbd = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout5_rgbd = nn.Dropout(p=0.4)

        if rgbd_dec:
            ####  RGBD DECODER  ####
            self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR5_RGBD_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 512]])
            self.dropout5_dec = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR5_RGBD_DEC)

            self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR4_RGBD_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 256]])
            self.dropout4_dec = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR4_RGBD_DEC)

            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR3_RGBD_DEC = make_layers_from_size([[256, 256], [256, 256], [256, 128]])
            self.dropout3_dec = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR3_RGBD_DEC)

            self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR2_RGBD_DEC = make_layers_from_size([[128, 128], [128, 64]])

            self.need_initialization.append(self.CBR2_RGBD_DEC)

            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR1_RGBD_DEC = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=batchNorm_momentum),
                nn.ReLU(),
                nn.Conv2d(64, num_labels, kernel_size=3, padding=1),
            )
            self.need_initialization.append(self.CBR1_RGBD_DEC)

        if rgb_dec:
            ####  RGB DECODER  ####
            self.unpool5_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR5_RGB_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 512]])
            self.dropout5_dec_rgb = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR5_RGB_DEC)

            self.unpool4_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR4_RGB_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 256]])
            self.dropout4_dec_rgb = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR4_RGB_DEC)

            self.unpool3_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR3_RGB_DEC = make_layers_from_size([[256, 256], [256, 256], [256, 128]])
            self.dropout3_dec_rgb = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR3_RGB_DEC)

            self.unpool2_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR2_RGB_DEC = make_layers_from_size([[128, 128], [128, 64]])

            self.need_initialization.append(self.CBR2_RGB_DEC)

            self.unpool1_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR1_RGB_DEC = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=batchNorm_momentum),
                nn.ReLU(),
                nn.Conv2d(64, num_labels, kernel_size=3, padding=1),
            )
            self.need_initialization.append(self.CBR1_RGB_DEC)

        if depth_dec:
            ####  Depth DECODER  ####
            self.unpool5_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR5_D_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 512]])
            self.dropout5_dec_d = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR5_D_DEC)

            self.unpool4_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR4_D_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 256]])
            self.dropout4_dec_d = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR4_D_DEC)

            self.unpool3_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR3_D_DEC = make_layers_from_size([[256, 256], [256, 256], [256, 128]])
            self.dropout3_dec_d = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR3_D_DEC)

            self.unpool2_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR2_D_DEC = make_layers_from_size([[128, 128], [128, 64]])

            self.need_initialization.append(self.CBR2_D_DEC)

            self.unpool1_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR1_D_DEC = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=batchNorm_momentum),
                nn.ReLU(),
                nn.Conv2d(64, num_labels, kernel_size=3, padding=1),
            )
            self.need_initialization.append(self.CBR1_D_DEC)

    def forward(self, rgb_inputs, depth_inputs):
        if self.mode == 'I':
            return self.forward_I(rgb_inputs, depth_inputs)
        elif self.mode == 'E':
            return self.forward_E(rgb_inputs, depth_inputs)
        else:
            return self.forward_L(rgb_inputs, depth_inputs)

    def forward_I(self, rgb_inputs, depth_inputs):

        ########  DEPTH ENCODER  ########
        # Stage 1
        x_1 = self.CBR1_DEPTH_ENC(depth_inputs)
        x, id1_d = self.pool1_d(x_1)

        # Stage 2
        x_2 = self.CBR2_DEPTH_ENC(x)
        x, id2_d = self.pool2_d(x_2)

        # Stage 3
        x_3 = self.CBR3_DEPTH_ENC(x)
        x, id3_d = self.pool4_d(x_3)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_DEPTH_ENC(x)
        x, id4_d = self.pool4_d(x_4)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_DEPTH_ENC(x)

        ########  RGB ENCODER  ########

        # Stage 1
        y = self.CBR1_RGB_ENC(rgb_inputs)
        y = torch.add(y, x_1)
        y = torch.div(y, 2)
        y, id1 = self.pool1(y)

        # Stage 2
        y = self.CBR2_RGB_ENC(y)
        y = torch.add(y, x_2)
        y = torch.div(y, 2)
        y, id2 = self.pool2(y)

        # Stage 3
        y = self.CBR3_RGB_ENC(y)
        y = torch.add(y, x_3)
        y = torch.div(y, 2)
        y, id3 = self.pool3(y)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB_ENC(y)
        y = torch.add(y, x_4)
        y = torch.div(y, 2)
        y, id4 = self.pool4(y)
        y = self.dropout4(y)

        # Stage 5 =========== Intermediate Fusion ================
        y = self.CBR5_RGB_ENC(y)
        y = torch.add(y, x_5)  # fusion
        y = torch.div(y, 2)
        y_size = y.size()

        y, id5 = self.pool5(y)
        y = self.dropout5(y)
        y = self.after_fusion_identity(y)  # [-1, 512, 15, 20]
        ########  RGB-D DECODER  ########

        # Stage 5 dec
        y = self.unpool5(y, id5, output_size=y_size)
        y = self.CBR5_RGBD_DEC(y)  # [-1, 512, 30, 40]
        y = self.dropout5_dec(y)

        # Stage 4 dec
        y = self.unpool4(y, id4)
        y = self.CBR4_RGBD_DEC(y)  # [-1, 512, 60, 80]
        y = self.dropout4_dec(y)

        # Stage 3 dec
        y = self.unpool3(y, id3)
        y = self.CBR3_RGBD_DEC(y)  # [-1, 128, 120, 160]
        y = self.dropout3_dec(y)

        # Stage 2 dec
        y = self.unpool2(y, id2)
        y = self.CBR2_RGBD_DEC(y)  # [-1, 64, 240, 320]

        # Stage 1 dec
        y = self.unpool1(y, id1)
        y = self.CBR1_RGBD_DEC(y)  # [-1, 41, 480, 640]

        return y



class FusenetBranchGenerator(nn.Module):
    def __init__(self, num_labels, mode='I'):
        super(FusenetBranchGenerator, self).__init__()
        batchNorm_momentum = 0.1  # TODO:make param

        self.need_initialization = []  # modules that need initialization
        model_dic = VGG16_initializator()
        self.mode = mode
        self.penultimate_identity = nn.Identity()
        rgb_enc, depth_enc, rgb_dec, depth_dec = False, False, False, False

        if mode == 'RGB':
            rgb_enc = True
            rgb_dec = True
            self.hook_names = ['CBR1_RGB_DEC.0', 'CBR2_RGB_ENC', 'CBR4_RGB_ENC', 'CBR5_RGB_ENC', 'CBR4_RGB_DEC', 'CBR5_RGB_DEC']
        elif mode == 'D':
            depth_enc = True
            depth_dec = True
            self.hook_names = ['CBR1_D_DEC.0', 'CBR2_DEPTH_ENC', 'CBR4_DEPTH_ENC', 'CBR5_DEPTH_ENC', 'CBR4_D_DEC', 'CBR5_D_DEC']
        else:
            raise ValueError(f"Unsupported mode: {mode}. Please choose one mode in ['RGB', 'D'].")

        if rgb_enc:
            ##### RGB ENCODER ####
            self.CBR1_RGB_ENC = make_layers_from_names(["conv1_1", "conv1_2"], model_dic, 64)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR2_RGB_ENC = make_layers_from_names(["conv2_1", "conv2_2"], model_dic, 128)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR3_RGB_ENC = make_layers_from_names(["conv3_1", "conv3_2", "conv3_3"], model_dic, 256)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout3 = nn.Dropout(p=0.4)

            self.CBR4_RGB_ENC = make_layers_from_names(["conv4_1", "conv4_2", "conv4_3"], model_dic, 512)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout4 = nn.Dropout(p=0.4)

            self.CBR5_RGB_ENC = make_layers_from_names(["conv5_1", "conv5_2", "conv5_3"], model_dic, 512)

            # after the fusion or the last layer in mode L
            self.dropout5 = nn.Dropout(p=0.4)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        if depth_enc:
            ##### Depth ENCODER ####
            feats_depth = list(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features.children())
            avg = torch.mean(feats_depth[0].weight.data, dim=1)
            avg = avg.unsqueeze(1)

            conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            conv11d.weight.data = avg

            self.CBR1_DEPTH_ENC = make_layers_from_names(["conv1_2"], model_dic, 64, conv11d)
            self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR2_DEPTH_ENC = make_layers_from_names(["conv2_1", "conv2_2"], model_dic, 128)
            self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR3_DEPTH_ENC = make_layers_from_names(["conv3_1", "conv3_2", "conv3_3"], model_dic, 256)
            self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout3_d = nn.Dropout(p=0.4)

            self.CBR4_DEPTH_ENC = make_layers_from_names(["conv4_1", "conv4_2", "conv4_3"], model_dic, 512)
            self.pool4_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout4_d = nn.Dropout(p=0.4)

            self.CBR5_DEPTH_ENC = make_layers_from_names(["conv5_1", "conv5_2", "conv5_3"], model_dic, 512)
            self.pool5_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout5_d = nn.Dropout(p=0.4)

        if rgb_dec:
            ####  RGB DECODER  ####
            self.unpool5_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR5_RGB_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 512]])
            self.dropout5_dec_rgb = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR5_RGB_DEC)

            self.unpool4_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR4_RGB_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 256]])
            self.dropout4_dec_rgb = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR4_RGB_DEC)

            self.unpool3_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR3_RGB_DEC = make_layers_from_size([[256, 256], [256, 256], [256, 128]])
            self.dropout3_dec_rgb = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR3_RGB_DEC)

            self.unpool2_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR2_RGB_DEC = make_layers_from_size([[128, 128], [128, 64]])

            self.need_initialization.append(self.CBR2_RGB_DEC)

            self.unpool1_rgb = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR1_RGB_DEC = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=batchNorm_momentum),
                nn.ReLU(),
                nn.Conv2d(64, num_labels, kernel_size=3, padding=1)
            )
            self.need_initialization.append(self.CBR1_RGB_DEC)

        if depth_dec:
            ####  Depth DECODER  ####
            self.unpool5_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR5_D_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 512]])
            self.dropout5_dec_d = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR5_D_DEC)

            self.unpool4_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR4_D_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 256]])
            self.dropout4_dec_d = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR4_D_DEC)

            self.unpool3_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR3_D_DEC = make_layers_from_size([[256, 256], [256, 256], [256, 128]])
            self.dropout3_dec_d = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR3_D_DEC)

            self.unpool2_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR2_D_DEC = make_layers_from_size([[128, 128], [128, 64]])

            self.need_initialization.append(self.CBR2_D_DEC)

            self.unpool1_d = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.CBR1_D_DEC = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=batchNorm_momentum),
                nn.ReLU(),
                nn.Conv2d(64, num_labels, kernel_size=3, padding=1)
            )
            self.need_initialization.append(self.CBR1_D_DEC)

    def forward(self, inputs):
        if self.mode == 'RGB':
            return self.forward_rgb(inputs)
        else:
            return self.forward_d(inputs)

    def forward_rgb(self, rgb_inputs):
        ########  RGB ENCODER  ########
        # Stage 1
        y = self.CBR1_RGB_ENC(rgb_inputs)  # [-1, 64, 480, 640]
        y, id1 = self.pool1(y)

        # Stage 2
        y = self.CBR2_RGB_ENC(y)  # [-1, 128, 240, 320]
        y, id2 = self.pool2(y)

        # Stage 3
        y = self.CBR3_RGB_ENC(y)  # [-1, 256, 120, 160]
        y, id3 = self.pool3(y)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB_ENC(y)  # [-1, 512, 60, 80]
        y, id4 = self.pool4(y)
        y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB_ENC(y)  # [-1, 512, 30, 40]
        y_size = y.size()
        y, id5 = self.pool5(y)
        y = self.dropout5(y)

        ########  RGB DECODER  ########
        # Stage 5 dec
        y = self.unpool5_rgb(y, id5, output_size=y_size)
        y = self.CBR5_RGB_DEC(y)  # [-1, 512, 30, 40]
        y = self.dropout5_dec_rgb(y)

        # Stage 4 dec
        y = self.unpool4_rgb(y, id4)
        y = self.CBR4_RGB_DEC(y)  # [-1, 256, 60, 80]
        y = self.dropout4_dec_rgb(y)

        # Stage 3 dec
        y = self.unpool3_rgb(y, id3)
        y = self.CBR3_RGB_DEC(y)  # [-1, 128, 120, 160]
        y = self.dropout3_dec_rgb(y)

        # Stage 2 dec
        y = self.unpool2_rgb(y, id2)
        y = self.CBR2_RGB_DEC(y)  # [-1, 64, 240, 320]

        # Stage 1 dec
        y = self.unpool1_rgb(y, id1)
        y = self.CBR1_RGB_DEC(y)  # [-1, num_labels, 480, 640]
        return y

    def forward_d(self, depth_inputs):
        ########  DEPTH ENCODER  ########
        # Stage 1
        x_1 = self.CBR1_DEPTH_ENC(depth_inputs)  # [-1, 64, 480, 640]
        x, id1_d = self.pool1_d(x_1)

        # Stage 2
        x_2 = self.CBR2_DEPTH_ENC(x)  # [-1, 128, 240, 320]
        x, id2_d = self.pool2_d(x_2)

        # Stage 3
        x_3 = self.CBR3_DEPTH_ENC(x)  # [-1, 256, 120, 160]
        x, id3_d = self.pool4_d(x_3)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_DEPTH_ENC(x)  # [-1, 512, 60, 80]
        x, id4_d = self.pool4_d(x_4)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_DEPTH_ENC(x)  # [-1, 512, 30, 40]
        x_size = x_5.size()
        x, id5_d = self.pool5_d(x_5)
        x = self.dropout5_d(x)

        ########  DEPTH DECODER  ########
        # Stage 5 dec
        x = self.unpool5_d(x, id5_d, output_size=x_size)
        x = self.CBR5_D_DEC(x)  # [-1, 512, 30, 40]
        x = self.dropout5_dec_d(x)

        # Stage 4 dec
        x = self.unpool4_d(x, id4_d)
        x = self.CBR4_D_DEC(x)  # [-1, 256, 60, 80]
        x = self.dropout4_dec_d(x)

        # Stage 3 dec
        x = self.unpool3_d(x, id3_d)
        x = self.CBR3_D_DEC(x)  # [-1, 128, 120, 160]
        x = self.dropout3_dec_d(x)

        # Stage 2 dec
        x = self.unpool2_d(x, id2_d)
        x = self.CBR2_D_DEC(x)  # [-1, 64, 240, 320]

        # Stage 1 dec
        x = self.unpool1_d(x, id1_d)
        x = self.CBR1_D_DEC(x)  # [-1, num_labels, 480, 640]
        return x


if __name__ == '__main__':
    x_rgb = torch.randn(1, 3, 480, 640)
    x_depth = torch.randn(1, 1, 480, 640)

    # Tea.-MM
    # model = define_FuseNet(num_labels=41, mode='L')
    # summary(model, (x_rgb.shape, x_depth.shape))
    # output = model(x_rgb, x_depth)
    # print(output.shape)

    # Stu.-RGB
    model = define_branches(num_labels=41, mode='RGB')
    output = model(x_rgb)
    print(output.shape)

    # Stu.-D
    # model = define_branches(num_labels=41, mode='D')
    # output = model(x_depth)
    # print(output.shape)
