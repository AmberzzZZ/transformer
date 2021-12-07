
## swin
    
    official repo: https://github.com/microsoft/Swin-Transformer
    keras version: https://github.com/keras-team/keras-io/blob/master/examples/vision/swin_transformers.py

    related papers:
    origin Swin: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    Swin for object detection: End-to-End Semi-Supervised Object Detection with Soft Teacher
    Swin for segmentation: Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation

    swin family:
    swin-T: (224,224), C=96, num_layers=[2,2,6,2], num_heads=[3,6,12,24]
    swin-S: (224,224), C=96, num_layers=[2,2,18,2], num_heads=[3,6,12,24]
    swin-B: (224,224) / (384,384), C=128, num_layers=[2,2,18,2], num_heads=[4,8,16,32]
    swin-L: (224,224) / (384,384), C=196, num_layers=[2,2,18,2], num_heads=[6,12,24,48]

    what's new in swin: 
    * hierarchical: 一般ViT都是桶型结构，fp过程中resolution不变，浅层计算量不友好，而且不好应用于FPN及后续dense任务
    * window attention: window比patch高一个level，将att分解成window-based global att和local att，减少计算量
    * activation: GeLU


    #### input embedding ####
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
    * 用来描述window中任意两点的相对位置关系：[wh, wh]，两个wh分别表示window map上任意一点
    * 初始相对距离度量分为h和w两个axis，range from [0,2h-1]和[0,2w-1]
    # * 2-dim coords可以合并成1-dim：采用两个digit->两位数的转换方式
    * shared among windows
    * 常量

    relative position bias: 
    * 用来保存任意一对相对位置的position bias：[2h-1, 2w-1, n_heads]
    * truncated normal distribution: 初始用截断的正态分布填充
    * relative position index中保存的所有相对距离，都能在relative position bias找到一组bias: [wh,wh,n_heads]
    * learnable

    window attention:
    * 将特征图分解成互不重叠的window，每个window包含M*M个patch
    * 在每个windows内部做self-attention，每个window参数共享————window-based local attention
    * window_size=7: 要求特征图尺寸要能整除7，否则pooling
    * shifted-window: 
        如果没有shifted-window，每个stage的感受野才2倍，不然都不变的
        given window_size=M: 划分windows的时候不从左上角开始，而是wh各平移M//2
        等价于把featuremap平移一部分然后正常partition
        tf.manip.roll / torch.roll


## swin V2


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





















