import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchinfo import summary

'''
From official code 'psp_family.py'.
'''

def init_layers(layers):
    for layer in layers:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)


class SelfAttention(nn.Module):
    # Take audio self-attention for example.
    def __init__(self, audio_emb_dim, hidden_dim=64, device=None):
        super(SelfAttention, self).__init__()
        self.device = device
        self.phi = nn.Linear(audio_emb_dim, hidden_dim)
        self.theta = nn.Linear(audio_emb_dim, hidden_dim)
        self.g = nn.Linear(audio_emb_dim, hidden_dim)
        layers = [self.phi, self.theta, self.g]
        init_layers(layers)

    def forward(self, audio_feature):
        # audio_feature: [bs, seg_num=10, 128]
        bs, seg_num, audio_emb_dim = audio_feature.shape
        phi_a = self.phi(audio_feature)
        theta_a = self.theta(audio_feature)
        g_a = self.g(audio_feature)
        a_seg_rel = torch.bmm(phi_a, theta_a.permute(0, 2, 1)) # [bs, seg_num, seg_num]
        a_seg_rel = a_seg_rel / torch.sqrt(torch.FloatTensor([audio_emb_dim]).to(self.device))
        a_seg_rel = F.relu(a_seg_rel)
        a_seg_rel = (a_seg_rel + a_seg_rel.permute(0, 2, 1)) / 2
        sum_a_seg_rel = torch.sum(a_seg_rel, dim=-1, keepdim=True)
        a_seg_rel = a_seg_rel / (sum_a_seg_rel + 1e-8)
        a_att = torch.bmm(a_seg_rel, g_a)
        a_att_plus_ori = a_att + audio_feature
        return a_att_plus_ori, a_seg_rel


class AVGA(nn.Module):
    """Audio-guided visual attention used in AVEL.
    AVEL:Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrained videos. InECCV, 2018
    """
    def __init__(self, a_dim=128, v_dim=512, hidden_size=512, map_size=49):
        super(AVGA, self).__init__()
        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(a_dim, hidden_size)
        self.affine_video = nn.Linear(v_dim, hidden_size)
        self.affine_v = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_g = nn.Linear(hidden_size, map_size, bias=False)
        self.affine_h = nn.Linear(map_size, 1, bias=False)

        init.xavier_uniform_(self.affine_v.weight)
        init.xavier_uniform_(self.affine_g.weight)
        init.xavier_uniform_(self.affine_h.weight)
        init.xavier_uniform_(self.affine_audio.weight)
        init.xavier_uniform_(self.affine_video.weight)

    def forward(self, audio, video):
        # audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        V_DIM = video.size(-1)
        v_t = video.view(video.size(0) * video.size(1), -1, V_DIM) # [bs*10, 49, 512]
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t)) # [bs*10, 49, 512]
        a_t = audio.view(-1, audio.size(-1)) # [bs*10, 128]
        a_t = self.relu(self.affine_audio(a_t)) # [bs*10, 512]
        content_v = self.affine_v(v_t) + self.affine_g(a_t).unsqueeze(2) # [bs*10, 49, 49] + [bs*10, 49, 1]

        z_t = self.affine_h((torch.tanh(content_v))).squeeze(2) # [bs*10, 49]
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map, [bs*10, 1, 49]
        c_t = torch.bmm(alpha_t, V).view(-1, V_DIM) # [bs*10, 1, 512]
        video_t = c_t.view(video.size(0), -1, V_DIM) # attended visual features, [bs, 10, 512]
        return video_t


class LSTM_A_V(nn.Module):
    def __init__(self, a_dim, v_dim, hidden_dim=128, seg_num=10, device=None):
        super(LSTM_A_V, self).__init__()
        self.device = device
        self.lstm_audio = nn.LSTM(a_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=0.0)
        self.lstm_video = nn.LSTM(v_dim, hidden_dim, 1, batch_first=True, bidirectional=True, dropout=0.0)

    def init_hidden(self, a_fea, v_fea):
        bs, seg_num, a_dim = a_fea.shape
        # hidden_a = (torch.zeros(2, bs, a_dim), torch.zeros(2, bs, a_dim))
        # hidden_v = (torch.zeros(2, bs, a_dim), torch.zeros(2, bs, a_dim))
        hidden_a = (torch.zeros(2, bs, a_dim).to(self.device), torch.zeros(2, bs, a_dim).to(self.device))  # changed by Gray: del double()
        hidden_v = (torch.zeros(2, bs, a_dim).to(self.device), torch.zeros(2, bs, a_dim).to(self.device))  # changed by Gray: del double()
        return hidden_a, hidden_v

    def forward(self, a_fea, v_fea):
        # a_fea, v_fea: [bs, 10, 128]
        hidden_a, hidden_v = self.init_hidden(a_fea, v_fea)
        # Bi-LSTM for temporal modeling
        self.lstm_video.flatten_parameters() # .contiguous()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(a_fea, hidden_a)
        lstm_video, hidden2 = self.lstm_video(v_fea, hidden_v)

        return lstm_audio, lstm_video


class PSP(nn.Module):
    """Postive Sample Propagation module"""

    def __init__(self, a_dim=256, v_dim=256, hidden_dim=256, out_dim=256, device=None):
        super(PSP, self).__init__()
        self.device = device
        self.v_L1 = nn.Linear(v_dim, hidden_dim, bias=False)
        self.v_L2 = nn.Linear(v_dim, hidden_dim, bias=False)
        self.v_fc = nn.Linear(v_dim, out_dim, bias=False)
        self.a_L1 = nn.Linear(a_dim, hidden_dim, bias=False)
        self.a_L2 = nn.Linear(a_dim, hidden_dim, bias=False)
        self.a_fc = nn.Linear(a_dim, out_dim, bias=False)
        self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1) # default=0.1
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6)

        # self.v_lstm_fc = nn.Linear(hidden_dim, 1, bias=False)
        # self.a_lstm_fc = nn.Linear(hidden_dim, 1, bias=False)

        layers = [self.v_L1, self.v_L2, self.a_L1, self.a_L2, self.a_fc, self.v_fc]
        self.init_weights(layers)

    def init_weights(self, layers):
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            # nn.init.orthogonal(layer.weight)
            # nn.init.kaiming_normal_(layer.weight, mode='fan_in')

    def forward(self, a_fea, v_fea, thr_val):
        # a_fea: [bs, 10, 256], a_lstm
        # v_fea: [bs, 10, 256], v_lstm
        # thr_val: the hyper-parameter for pruing process
        v_branch1 = self.dropout(self.activation(self.v_L1(v_fea))) #[bs, 10, hidden_dim]
        v_branch2 = self.dropout(self.activation(self.v_L2(v_fea)))
        a_branch1 = self.dropout(self.activation(self.a_L1(a_fea)))
        a_branch2 = self.dropout(self.activation(self.a_L2(a_fea)))

        beta_va = torch.bmm(v_branch2, a_branch1.permute(0, 2, 1)) # row(v) - col(a), [bs, 10, 10]
        beta_va /= torch.sqrt(torch.tensor([v_branch2.shape[2]], dtype=torch.float32, device=self.device))
        # beta_va /= torch.sqrt(torch.FloatTensor([v_branch2.shape[2]]))
        beta_va = F.relu(beta_va) # ReLU
        beta_av = beta_va.permute(0, 2, 1) # transpose

        sum_v_to_a = torch.sum(beta_va, dim=-1, keepdim=True)
        beta_va = beta_va / (sum_v_to_a + 1e-8) # [bs, 10, 10]
        gamma_va = (beta_va > thr_val).float() * beta_va
        sum_v_to_a = torch.sum(gamma_va, dim=-1, keepdim=True)  # l1-normalization
        gamma_va = gamma_va / (sum_v_to_a + 1e-8)

        sum_a_to_v = torch.sum(beta_av, dim=-1, keepdim=True)
        beta_av = beta_av / (sum_a_to_v + 1e-8)
        gamma_av = (beta_av > thr_val).float() * beta_av
        sum_a_to_v = torch.sum(gamma_av, dim=-1, keepdim=True)
        gamma_av = gamma_av / (sum_a_to_v + 1e-8)

        a_pos = torch.bmm(gamma_va, a_branch2)
        v_psp = v_fea + a_pos

        v_pos = torch.bmm(gamma_av, v_branch1)
        a_psp = a_fea + v_pos

        v_psp = self.dropout(self.relu(self.v_fc(v_psp)))
        a_psp = self.dropout(self.relu(self.a_fc(a_psp)))
        v_psp = self.layer_norm(v_psp)
        a_psp = self.layer_norm(a_psp)

        a_v_fuse = torch.mul(v_psp + a_psp, 0.5)
        return a_v_fuse, v_psp, a_psp


class Classify(nn.Module):
    def __init__(self, hidden_dim=256, category_num=28):
        super(Classify, self).__init__()
        self.L1 = nn.Linear(hidden_dim, 64, bias=False)
        self.L2 = nn.Linear(64, category_num, bias=False)
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.xavier_uniform_(self.L2.weight)
    def forward(self, feature):
        out = F.relu(self.L1(feature))
        out = self.L2(out)
        # out = F.softmax(out, dim=-1)
        return out


class AVSimilarity(nn.Module):
    """ function to compute audio-visual similarity"""
    def __init__(self,):
        super(AVSimilarity, self).__init__()

    def forward(self, v_fea, a_fea):
        # fea: [bs, 10, 256]
        v_fea = F.normalize(v_fea, dim=-1)
        a_fea = F.normalize(a_fea, dim=-1)
        cos_simm = torch.sum(torch.mul(v_fea, a_fea), dim=-1) # [bs, 10]
        return cos_simm


class fully_psp_net(nn.Module):
    '''
    System flow for fully supervised audio-visual event localization.
    '''
    def __init__(self, vis_fea_type='vgg', flag='psp', a_dim=128, v_dim=512, hidden_dim=128, category_num=28, thr_val=0.099, device=None):
        super(fully_psp_net, self).__init__()
        self.vis_fea_type = vis_fea_type
        self.flag = flag
        if self.vis_fea_type == 'vgg':
            self.v_init_dim = 512
        else:
            self.v_init_dim = 1024
        self.thr_val=thr_val
        self.fa = nn.Sequential(
            nn.Linear(a_dim, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        )
        self.fv = nn.Sequential(
            nn.Linear(self.v_init_dim, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        )
        self.linear_v = nn.Linear(v_dim, a_dim)
        self.relu = nn.ReLU()
        self.attention = AVGA(v_dim=self.v_init_dim)
        self.lstm_a_v = LSTM_A_V(a_dim=a_dim, v_dim=hidden_dim, hidden_dim=hidden_dim, device=device)
        self.psp = PSP(a_dim=a_dim*2, v_dim=hidden_dim*2, device=device)
        self.av_simm = AVSimilarity()

        self.v_classify = Classify(hidden_dim=256)
        self.a_classify = Classify(hidden_dim=256)

        self.L1 = nn.Linear(2*hidden_dim, 64, bias=False)
        self.L2 = nn.Linear(64, category_num, bias=False)
  
        self.event_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, 1)
        )
        self.category_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, category_num)
        )

        layers = [self.L1, self.L2]
        self.init_layers(layers)
        self.fusion_avg_identity = nn.Identity()
        self.hook_names = ['lstm_a_v', 'psp', 'category_classifier', 'fusion_avg_identity']

    def init_layers(self, layers):
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, video, audio):
        # audio: [bs, 10, 128]
        # video: [bs, 10, 7, 7, 512]
        # pdb.set_trace()
        bs, seg_num, H, W, v_dim = video.shape
        fa_fea = self.fa(audio)
        video_t = self.attention(fa_fea, video) # [bs, 10, 512]
        video_t = self.fv(video_t) # [bs, 10, 128]
        lstm_audio, lstm_video = self.lstm_a_v(fa_fea, video_t)
        # print('lstm_audio.shape: ', lstm_audio.shape)
        # print('lstm_video.shape: ', lstm_video.shape)
        fusion, final_v_fea, final_a_fea = self.psp(lstm_audio, lstm_video, thr_val=self.thr_val) # [bs, 10, 256]
        avps = self.av_simm(final_v_fea, final_a_fea)

        event_logits = self.event_classifier(fusion).squeeze(-1) # [B, 10]
        avg_fea = fusion.mean(dim=1) # [B, 256]
        fusion_avg_identity = self.fusion_avg_identity(avg_fea)
        category_logits = self.category_classifier(fusion_avg_identity) # [B, 28]
        if self.flag == "sspsp":
            return event_logits, category_logits, final_v_fea, final_a_fea
        elif self.vis_fea_type == 'vgg':
            return event_logits, category_logits, avps, fusion
        else:
            return event_logits, category_logits, avps, fusion, final_v_fea, final_a_fea


class AudioBranch(nn.Module):
    def __init__(self):
        super(AudioBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.hook_names = ['conv1', 'conv2', 'conv3']

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        return x


class VisualBranch(nn.Module):
    def __init__(self):
        super(VisualBranch, self).__init__()
        # 三层卷积
        self.conv1 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(256)
        self.conv2 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2)  # 使用池化层减小特征图尺寸
        # 全局平均池化，输出大小为 (batch_size, 320, 1, 1, 1)
        # self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.hook_names = ['conv1', 'conv2', 'conv3']

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 经过第一层卷积和池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 经过第二层卷积和池化
        x = F.relu(self.bn3(self.conv3(x)))  # 经过第三层卷积
        # x = self.global_max_pool(x.squeeze(2))  # 全局平均池化降维
        x = x.reshape(x.size(0), -1)  # 展平，输出大小将是 (batch_size, 320)
        return x


class DualStreamCNN_VGGS(nn.Module):
    def __init__(self, cls_num):
        super(DualStreamCNN_VGGS, self).__init__()
        self.audio_branch = AudioBranch()
        self.visual_branch = VisualBranch()

        # MLP
        self.fc1 = nn.Linear(384, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, cls_num)
        self.hook_names = ['audio_branch', 'visual_branch', 'fc1', 'fc2', 'fc3']

    def forward(self, x_visual, x_audio):
        visual_features = self.visual_branch(x_visual)  # 128
        audio_features = self.audio_branch(x_audio)  # 256

        # MLP for classification
        x = F.relu(self.fc1(torch.cat((audio_features, visual_features), dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AudioBranchNet_VGGS(nn.Module):
    def __init__(self, cls_num):
        super(AudioBranchNet_VGGS, self).__init__()
        self.audio_branch = AudioBranch()

        # MLP
        self.fc1 = nn.Linear(256, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, cls_num)
        self.hook_names = ['audio_branch', 'fc1', 'fc2', 'fc3']

    def forward(self, x_audio):
        audio_features = self.audio_branch(x_audio)

        # MLP for classification
        x = F.relu(self.fc1(audio_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VisualBranchNet_VGGS(nn.Module):
    def __init__(self, cls_num):
        super(VisualBranchNet_VGGS, self).__init__()
        self.visual_branch = VisualBranch()

        # MLP
        self.fc1 = nn.Linear(128, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, cls_num)
        self.hook_names = ['visual_branch', 'fc1', 'fc2', 'fc3']

    def forward(self, x_visual):
        visual_features = self.visual_branch(x_visual)

        # MLP for classification
        x = F.relu(self.fc1(visual_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



if __name__ == "__main__":
    cuda_id = 0
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    input_tensor_V = torch.randn(1, 10, 7, 7, 512).to(device)
    input_tensor_A = torch.randn(1, 10, 128).to(device)

    # Tea.-MM I
    # model = fully_psp_net(vis_fea_type='vgg', flag='cpsp', category_num=141, thr_val=0.099, device=device).to(device)
    # hook test ==================================================
    # from utils import hooks_builder
    # hooks_t, features_t = hooks_builder(model, model.hook_names)
    # ============================================================
    # output = model(input_tensor_V, input_tensor_A)
    # print(output[1].shape)
    # summary(model, (input_tensor_V.shape, input_tensor_A.shape))

    # Tea.-MM I
    # model = DualStreamCNN_VGGS(cls_num=141).to(device)
    # res = model(input_tensor_V.permute(0, 4, 1, 2, 3), input_tensor_A)  # [bs, 8]
    # print(res.shape)

    # Tea.-MM L
    # model = LateFusionDualStreamCNN_VGGS(cls_num=141).to(device)
    # res = model(input_tensor_V.permute(0, 4, 1, 2, 3), input_tensor_A)  # [bs, 8]
    # print(res.shape)

    # Stu.-V
    # model = VisualBranchNet_VGGS(cls_num=141)
    # summary(model, input_tensor_V.permute(0, 4, 1, 2, 3).shape)
    # out = model(input_tensor_V.permute(0, 4, 1, 2, 3))
    # print(out.shape)

    # Stu.-A
    # model = AudioBranchNet_VGGS(cls_num=141)
    # summary(model, input_tensor_A.shape)
    # out = model(input_tensor_A)
    # print(out.shape)
