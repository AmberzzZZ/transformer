## swin

    official repo: https://github.com/microsoft/Swin-Transformer
    keras version: https://github.com/keras-team/keras-io/blob/master/examples/vision/swin_transformers.py

    related papers:
    origin Swin: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    Swin for object detection: End-to-End Semi-Supervised Object Detection with Soft Teacher
    Swin for segmentation: Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation

    swin family:
    swin-T: (224,224), C=96, W=7, num_layers=[2,2,6,2], num_heads=[3,6,12,24], residual_drop=0.2
    swin-S: (224,224), C=96, W=7, num_layers=[2,2,18,2], num_heads=[3,6,12,24], residual_drop=0.3
    swin-B: (224,224) / (384,384), W=7/12, C=128, num_layers=[2,2,18,2], num_heads=[4,8,16,32], residual_drop=0.5
    swin-L: (224,224) / (384,384), W=7/12, C=196, num_layers=[2,2,18,2], num_heads=[6,12,24,48]

    what's new in swin:
    * hierarchical: 一般ViT都是桶型结构，fp过程中resolution不变，浅层计算量不友好，而且不好应用于FPN及后续dense任务
    * window attention: window比patch高一个level，将att分解成window-based global att和local att，减少计算量，而且attention layer的sequence长度不随input size变化了，easy to transfer
        ** attn_mask
        ** relative_positional_bias
    * activation: GeLU

    #### input embedding ####
    input size:
    given window_size=7
    因为有5倍下采样和window_split，所以input_size应该是32和7的倍数，所以default=224

    patch embeddings:
    * 将input img转换成token sequence，每个token来自一个patch
    * patch到token的映射通过conv2d，kernel和stride都是patch_size，filter是embedding_dim
    * 再reshape，就是(b,L,D)的token sequence了，L是patch_num，D是embedding_dim
    * (+PE)
    * Dropout

    positional embeddings:
    * no pos: 目前为止仅发现Google的MLP-Mixer是不使用PE的，说是隐式地学到了
    * abs pos: 大多数ViT的做法，基于input size计算一组1D/2D的固定值
    * relative pos: 本文的做法，不加abs PE，但是在MSA的QKV softmax层里面添加bias

    #### basic stage ####
    4 stages: 交替的swin-block和downsample-block

    #### swin-block ####
    WMSA-MLP-SWMSA-MLP
    包含两个swinTransformerBlock，一个window-based，一个shifted-window-based

    window partition:
    * 首先将(b,HW,C)的token sequence还原成二维(b,H,W,C)，H & W是grid size
    * 然后在(H,W)上面划分window，得到(b,H'W',ww,C)的token sequence
    * 每个local window，包含ww个token
    * 全局被划分为H'W'个window

    window-based MSA:
    * 在每个local window内部计算global attention
    * 输入是(b,N,ww,C)，
    * attention的计算是在最后两维上

    cyclic shift:
    * shift_size是半个window的size，这样相邻window之间就有交互了
    * 使用tf.manip.roll方法来实现


    #### patch-merging (downsample-block) #####
    * 先是恢复成2维特征图(b,h,w,c)
    * 然后间隔采样出stride2的featuremap，concat到空间维度，这样resolution减半，channel增加到了4倍
    * 再经过一个线性层，调整通道数为2倍
    * 比pooling保留的信息多


    #### classification head ####
    * swin里面没有cls token，stage4 最终输出[b,H/32*W/32,8C]的token embeddings
    * 对所有的token embeddings求平均，类似GAP，[b,8C]
    * 然后送入linear classifier，[b,n_classes]


    relative position index:
    * fixed, given window_size=7
    * local window内token的长度为7x7=49
    * 用来描述window中任意两点的相对位置关系：[49, 49]
        ** 第一步, 绝对位置->相对位置关系，[N,N,2], 2 for (delta_x, delta_y)
        ** 第二步，平移，+ (7-1), to start from 0
        ** 第三步, 2维->1维，[N,N,1], 1 for delta_x * 进位digit + delta_y
    * shared among windows in a layer
    * 常量

    relative position bias:
    * trainable, for each head, for each block, given window_size=7
    * 相对位置关系的range是[-6,6]+6=[0,12]，进位digit是13
    * 所以一维的相对位置关系的range是[0,13*13-1]
    * 用来学习不同位置关系编码对应的value，shape是 (13*13,n_heads)
    * 每次，从fixed [49,49,1]的relative mat里面取当前的位置关系值，作为当前的attn bias
    * truncated normal distribution: 初始用截断的正态分布填充

    window attention:
    * 将特征图分解成互不重叠的window，每个window包含M*M个patch
    * 在每个windows内部做self-attention，每个window参数共享————window-based local attention
    * window_size=7: 要求特征图尺寸要能整除7，否则pooling
    * shifted-window:
        用来建立相邻windows之间的connection
        given window_size=M: 划分windows的时候不从左上角开始，而是wh各平移M//2
        等价于把featuremap平移一部分然后正常partition：tf.manip.roll / torch.roll
        【重要！！！attn_mask！！！】平移了左/上的像素以后，右/下的windows不可避免地由不相邻patch拼接组成，
        这种拼接window计算attention的时候要限定在自己的window area内

    convert weights:
    有个问题，relative_position_bias是与window_size相关的，window尺寸改变就mismatch了，需要bi-cubic interpolation


    ###### training details 在论文附录里 ######
    ---- training ImageNet-1K ------
    * AdamW
    * 300 epochs / 20 linear warmup
    * cosine decay learning rate scheduler: 1e-3, weight decay 0.05
    * batch size 1024
    * gradient clipping with a max norm of 1
    * aug: RandAugment, Mixup, Cutmix, random erasing, but not repeated augmentation and EMA (no enhance)
    ---- pretrain ImageNet-22K ------
    * AdamW
    * 60 epochs / 5 linear warm-up
    * batch size 4096
    * linear decay lr scheduler: 1e-3, weight decay 0.01
    ---- finetuning ImageNet-1K ------
    * 30 epochs
    * batch size 1024
    * constant lr 1e-5, weight decay 1e-8
    * set stochastic depth ratio to 0.1

    *** 我实际训练下来，感觉cosineLR对收敛不太友好，第一个cycle结束那个阶跃的lr把精度又给拉低了，
    还有就是保存模型时候如果没保存优化器状态，reload进来又是一个全新的训练状态


    ###### init weights ######
    源代码里对dense/LN进行了初始化
    * dense: weight - trunc_normal(std=.02), bias - Constant(0)
    * LN：bias - Constant(0), weight - Constant(1.)
    * 对第一层的Conv2D没看到特殊初始化


## swin V2

    几个关注项：
    * scaling up：the bigger the better
    * instability issue：residual path上的值加到id path上会导致大模型不稳定，提出post-norm，在add之前norm residual value
    * transfer：原来的PE切换到其他resolution会掉点，设计了新的PE————log-spaced continous position bias (Log-CPB)





    代码上看主要区别就在于WindowAttention里面
    - 多了个relative_coords_table参数，positional embeeding有重定义
    - 多了个logit_scale，QK vector的归一化值有重定义



## swinUnet

    pytorch official: https://github.com/HuCaoFighting/Swin-Unet

    bottleneck可以看作swin encoder的stage4

    PatchExpand
    * input: (h,w,c)
    * 先是fc层，将通道数加宽，(h,w,4c)
    * 然后相邻4个特征图组合成一个
    * output: (2h,2w,c//2)

    FinalPatchExpand_X4
    * input: (h,w,c)
    * 先是fc层，将通道数加宽，(h,w,16c)
    * 然后相邻16个特征图组合成一个
    * output: (4h,4w,c)

    BasicLayer_up
    * input: 前一个level的decoder的输出(h,w,c) & 当前level的encoder的输出(h,w,c)
    * 先是concat，(h,w,2c)
    * 然后是fc层，(h,w,c)
    * 然后是swin block，(h,w,c)
    * 然后是PatchExpand，(2h,2w,c//2)
    * output: (2h,2w,c//2)


## nnFormer



## swin down-stream task1: semanic segmentation

    swin使用UperNet作为base framework进行语义分割

    * UPerNet: backbone+FPN(-PPM)+heads, standard FCN
    * swin version:
        ** 官方版本configs/base/models/upernet_swin.py,
        ** backbone用swin transformer
        ** decode_head: mmseg/decode_heads/uper_head.py, FPN+PPM+fusion head, FPN是(lateral conv, bilinear upsamp, pair-wise add, out conv), fusion head([P2,P3,P4,P5], bilinear upsamp to P2, concat, out convs)
        ** auxiliary_head: mmseg/decode_heads/fcn_head.py, 这里面的实现是(P3x8->convs+out conv), 不知道用这个x8map做啥


## swin down-stream task2: object detection

    swin使用二阶段架构进行目标检测，用swin-back替换之前的CNN-back

    先实现第一阶段，swin-rpn: swinback+fpn+rpnhead

    swinback
    * features: 是每个level的swinblocks的输出
    * add_norm: 输出之前再加一层layerNorm
    * 输出4个level的feature：[x4,C], [x8,2C], [x16,4C], [x32,8C]

    fpn
    * feats: 1x1 conv对齐现有feature，3x3 s2 conv创建new level feature
    * fusion: add
    * transfer task: 3x3 conv
    * conv with bias, norm=None, act=None

    rpn
    * shared convs: 1个3x3 conv with relu
    * heads: 1x1 conv, anchor-based

    3-cls rpn VS one-stage-detector:
    * light head
    * rough supervision: L1 vs giou
    * box的encoder方式

    ** 待解决：multi-scale，训练时，预先定义几个固定的尺度，每个epoch随机选择一个尺度进行训练，keras静态图咋弄？



## SimMIM

    official repo: https://github.com/microsoft/SimMIM



## visualization

    暂时没找到现成的，我的方案：
    transformer的similarity weight是softmax之后那个mat score，but in window-based situation？
    - given scores: [b,nWinH,nWinW,nH,window_size,window_size]
    - among windows [nWinH,nWinW]：拼接
    - among maps / shifted maps：一个是global att，一个是masked global att，覆盖的部分计算平均
    - among multi-heads [nH]：代表不同特征，应该分开看
    - cross layers：不同尺度，分开看




























