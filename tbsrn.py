import paddle
import paddle.nn as nn
import numpy as np
import math
import itertools
import copy
import pickle

# 输出value and shape对齐
class mish(nn.Layer):  
    def __init__(self, ):
        super().__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (paddle.tanh(nn.Softplus()(x)))
        return x

# 输出shape对齐
class GruBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, direction='bidirect')

    def forward(self, x):
        # x: b, c, w, h
        x = self.conv1(x)
        x = x.transpose([0, 2, 3, 1])  # b, w, h, c
        b = x.shape
        x = x.reshape([b[0] * b[1], b[2], b[3]])  # b*w, h, c
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.reshape([b[0], b[1], b[2], b[3]])
        x = x.transpose([0, 3, 1, 2])
        return x

# 输出shape对齐
class UpsampleBLock(nn.Layer):
    def __init__(self, in_channels, up_scale):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

# 输出value and shape对齐 (Note 是否有更细参数a_2, b_2)
class LayerNorm(nn.Layer):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super().__init__()
        # print(features)
        a_2 = self.create_parameter([features],nn.initializer.Assign(np.ones([features],dtype=np.float32)))
        self.add_parameter('a_2',a_2)
        b_2 = self.create_parameter([features],nn.initializer.Assign(np.zeros([features],dtype=np.float32)))
        self.add_parameter('b_2',b_2)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 输出value and shape对齐 
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = paddle.zeros([d_model, height, width])
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = paddle.exp(paddle.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = paddle.arange(0., width, dtype=paddle.float32).unsqueeze(1)
    pos_h = paddle.arange(0., height, dtype=paddle.float32).unsqueeze(1)
    pe[0:d_model:2, :, :] = paddle.sin(pos_w * div_term).transpose([1, 0]).unsqueeze(1).tile([1, height, 1])
    pe[1:d_model:2, :, :] = paddle.cos(pos_w * div_term).transpose([1, 0]).unsqueeze(1).tile([1, height, 1])
    pe[d_model::2, :, :] = paddle.sin(pos_h * div_term).transpose([1, 0]).unsqueeze(2).tile([1, 1, width])
    pe[d_model + 1::2, :, :] = paddle.cos(pos_h * div_term).transpose([1, 0]).unsqueeze(2).tile([1, 1, width])

    return pe

# 输出 shape对齐 
class PositionwiseFeedForward(nn.Layer):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))

def masked_fill(x, mask, value):
    mask = paddle.tile(mask, [int(x/y) for x,y in zip(x.shape, mask.shape)])
    x[mask == 1] = float('-inf')
    return x

# 输出value and shape对齐 
def attention(query, key, value, mask=None, dropout=None, align=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.shape[-1]
    assert len(key.shape) == 4
    scores = paddle.matmul(query, key.transpose([0, 1, 3, 2])) \
             / math.sqrt(d_k)
    if mask is not None:
        # print(mask)
        #scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = masked_fill(scores, mask==0, float('-inf'))
    else:
        pass

    p_attn = nn.functional.softmax(scores, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return paddle.matmul(p_attn, value), p_attn

def clones(layer, N):
    "Produce N identical layers."
    return nn.LayerList([copy.deepcopy(layer) for _ in range(N)])

# 输出 shape 对齐 
class MultiHeadedAttention(nn.Layer):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).reshape([nbatches, -1, self.h, self.d_k]).transpose([0, 2, 1, 3])
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose([0, 2, 1, 3]) \
            .reshape([nbatches, -1, self.h * self.d_k])

        return self.linears[-1](x), attention_map

# 输出 shape 对齐 
class FeatureEnhancer(nn.Layer):

    def __init__(self):
        super().__init__()

        self.multihead = MultiHeadedAttention(h=4, d_model=128, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=128)

        self.pff = PositionwiseFeedForward(128, 128)
        self.mul_layernorm3 = LayerNorm(features=128)

        self.linear = nn.Linear(128,64)

    def forward(self, conv_feature):
        '''
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        '''
        batch = conv_feature.shape[0]
        # may need to add to(device)
        position2d = positionalencoding2d(64,16,64).unsqueeze(0).reshape([1,64,1024])
        position2d = position2d.tile([batch,1,1])
        conv_feature = paddle.concat([conv_feature, position2d],1) # batch, 128(64+64), 32, 128
        result = conv_feature.transpose([0, 2, 1])
        origin_result = result
        result = self.mul_layernorm1(origin_result + self.multihead(result, result, result, mask=None)[0])
        origin_result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))
        result = self.linear(result)
        return result.transpose([0, 2, 1])

# 输出 shape 对齐 
class RecurrentResidualBlock(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2D(channels)#, momentum=0.1, use_global_stats = True)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(channels)#, momentum=0.1, use_global_stats = True)
        self.gru2 = GruBlock(channels, channels)
        self.feature_enhancer = FeatureEnhancer()

        for p in self.parameters():
            if p.dim() > 1:
                nn.initializer.XavierUniform(p)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        size = residual.shape
        residual = residual.reshape([size[0],size[1],-1])
        residual = self.feature_enhancer(residual)
        residual = residual.reshape([size[0], size[1], size[2], size[3]])
        return x + residual

# TBSRN 输出shape对齐
class TBSRN(nn.Layer):
    def __init__(self, scale_factor=2, width=128, height=32, STN=True, srb_nums=5, mask=False, hidden_units=32, input_channel=3):
        super(TBSRN, self).__init__()

        self.conv = nn.Conv2D(input_channel, 3,3,1,1)
        self.bn = nn.BatchNorm2D(3)#, momentum=0.1, use_global_stats = True)
        self.relu = nn.ReLU()

        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2D(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2 * hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2D(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2D(2 * hidden_units),#, momentum=0.1, use_global_stats = True)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2D(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}
        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = paddle.tanh(block[str(self.srb_nums + 3)])
        return output

# STNHead 输出shape对齐
def conv3x3_block(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  n = 3*3*out_planes
  conv_layer = nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, weight_attr=nn.initializer.Normal(0, math.sqrt(2. / n)))

  block = nn.Sequential(
    conv_layer,
    nn.BatchNorm2D(out_planes),#, momentum=0.1, use_global_stats = True),
    nn.ReLU(),
  )
  return block

class STNHead(nn.Layer):
  def __init__(self, in_planes = 3, num_ctrlpoints = 20, activation='none'):
    super().__init__()

    self.in_planes = in_planes
    self.num_ctrlpoints = num_ctrlpoints
    self.activation = activation
    self.stn_convnet = nn.Sequential(
                          conv3x3_block(in_planes, 32), # 32*64
                          nn.MaxPool2D(kernel_size=2, stride=2),
                          conv3x3_block(32, 64), # 16*32
                          nn.MaxPool2D(kernel_size=2, stride=2),
                          conv3x3_block(64, 128), # 8*16
                          nn.MaxPool2D(kernel_size=2, stride=2),
                          conv3x3_block(128, 256), # 4*8
                          nn.MaxPool2D(kernel_size=2, stride=2),
                          conv3x3_block(256, 256), # 2*4,
                          nn.MaxPool2D(kernel_size=(1,2), stride=(1,2)),
                          conv3x3_block(256, 256)) # 1*2

    self.stn_fc1 = nn.Sequential(
                      nn.Linear(2*256, 512, weight_attr=nn.initializer.Normal(0, 0.001)),
                      nn.BatchNorm1D(512),#, momentum=0.1),
                      nn.ReLU())
    ctrl_points = self.init_stn()
    self.stn_fc2 = nn.Linear(512, num_ctrlpoints*2, 
                    weight_attr=nn.initializer.Constant(0.0),
                    bias_attr=nn.initializer.Assign(ctrl_points.reshape(-1)) )


  def init_stn(self,):
    margin = 0.01
    sampling_num_per_side = int(self.num_ctrlpoints / 2)
    ctrl_pts_x = np.linspace(margin, 1.-margin, sampling_num_per_side)
    ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
    ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1-margin)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
    return ctrl_points

  def forward(self, x):
    x = self.stn_convnet(x)
    batch_size, _, h, w = x.shape
    x = x.reshape([batch_size, -1])
    img_feat = self.stn_fc1(x)
    x = self.stn_fc2(0.1 * img_feat)
    x = x.reshape([-1, self.num_ctrlpoints, 2])
    return img_feat, x

# TPSSpatialTransformer 输出shape对齐
def grid_sample(input, grid, canvas = None):
  output = nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
  if canvas is None:
    return output
  else:
    input_mask = input.data.new(input.size()).fill_(1)
    output_mask = F.grid_sample(input_mask, grid)
    padded_output = output * output_mask + canvas * (1 - output_mask)
    return padded_output

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
  N = input_points.shape[0]
  M = control_points.shape[0]
  pairwise_diff = input_points.reshape([N, 1, 2]) - control_points.reshape([1, M, 2])
  pairwise_diff_square = pairwise_diff * pairwise_diff
  pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
  repr_matrix = 0.5 * pairwise_dist * paddle.log(pairwise_dist)
  mask = repr_matrix != repr_matrix
  return masked_fill(repr_matrix, mask, 0)

# output_ctrl_pts are specified, according to our task.
def build_output_control_points(num_control_points, margins):
  margin_x, margin_y = margins
  num_ctrl_pts_per_side = num_control_points // 2
  ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
  ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
  ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
  ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
  ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
  # ctrl_pts_top = ctrl_pts_top[1:-1,:]
  # ctrl_pts_bottom = ctrl_pts_bottom[1:-1,:]
  output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
  output_ctrl_pts = paddle.to_tensor(output_ctrl_pts_arr,dtype=paddle.float32)
  return output_ctrl_pts


# demo: ~/test/models/test_tps_transformation.py
class TPSSpatialTransformer(nn.Layer):

  def __init__(self, output_image_size=[16, 64], num_control_points=20, margins=[0.05, 0.05]):
    super().__init__()
    self.output_image_size = output_image_size
    self.num_control_points = num_control_points
    self.margins = margins

    self.target_height, self.target_width = output_image_size
    target_control_points = build_output_control_points(num_control_points, margins)


    with open('inverse_kernel.pkl', 'rb') as f:
        inverse_kernel = pickle.load(f)
    inverse_kernel = paddle.to_tensor(inverse_kernel)
    with open('target_coordinate_repr.pkl', 'rb') as f:
        target_coordinate_repr = pickle.load(f)
    target_coordinate_repr = paddle.to_tensor(target_coordinate_repr)


    # register precomputed matrices
    self.register_buffer('inverse_kernel', inverse_kernel)
    self.register_buffer('padding_matrix', paddle.zeros([3, 2]))
    self.register_buffer('target_coordinate_repr', target_coordinate_repr)
    self.register_buffer('target_control_points', target_control_points)

  def forward(self, input, source_control_points):
    assert source_control_points.ndimension() == 3
    assert source_control_points.shape[1] == self.num_control_points
    assert source_control_points.shape[2] == 2
    batch_size = source_control_points.shape[0]

    Y = paddle.concat([source_control_points, self.padding_matrix.expand([batch_size, 3, 2])], 1)
    mapping_matrix = paddle.matmul(self.inverse_kernel, Y)
    source_coordinate = paddle.matmul(self.target_coordinate_repr, mapping_matrix)

    grid = source_coordinate.reshape([-1, self.target_height, self.target_width, 2])
    # grid = paddle.where(grid>0, grid, paddle.zeros(grid.shape))
    # grid = paddle.where(grid<1, grid, paddle.ones(grid.shape))
    grid = paddle.clip(grid, 0 , 1)
    grid = 2.0 * grid - 1.0
    output_maps = grid_sample(input, grid, canvas=None)
    return output_maps, source_coordinate