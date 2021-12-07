
## vision transformer (ViT)
    
    官方repo: https://github.com/google-research/vision_transformer
    third repo: https://github.com/lucidrains/vit-pytorch

    task: supervised classification

    inputs: 将图片切成不重叠的16x16块，然后flatten，然后用learnable的线性层降维，然后添加cls token，然后加上PE
        * image patch sequence & trainable linear projection
        * PE: trainable 1d embedding
        * x0: trainable pretended 1d embedding
        * [x0, patch_embeddings, ] + PEs
        实现上，是通过一个一层卷积，kernel size和stride都是patch size，将每个ch3-patch线性映射成一个emb-dim vec

    model: transformer encoder
        * patch_size
        * hidden_size: through all
        * MSA layer: 没有mask，最简单的版本

    MLP head: 


    GeLU:
        Gaussian error linear unit: x * P(X <= x), where P(X) ~ N(0, 1)
        if approx:
            y = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        else:
            y = 0.5 * x * (1 + erf(x / sqrt(2)))

    LN:
        https://www.geek-book.com/src/docs/keras/keras/keras.io/api/layers/normalization_layers/layer_normalization/index.html
        trainable的情况下，given inputs [b,(hwd),c]，参数量是2*(hwd)*c，所以用在1D比较正常一点


    子类继承模型：class ***(keras.Model)
        * init里面定义层不能复用
        * 批量定义的layer list里面每个layer必须在self作用空间下声明
        * checkpoint只能save_weights不能save_model，因为不支持get_config()和序列化


    training details:
    * cosine learning rate
    * Adam + L2 reg: momentum=0.9, wd=1e-5

    主要缺陷：
    * 模型量级太大，batch size大，tpu级别训练
    * 训练数据量必须要大，不大精度不行，基本没办法在自己的数据集上train from scratch

    基于ViT的提升有：DeiT, T2T-ViT, LV-ViT
    


## LV-ViT

    官方repo: https://github.com/zihangJiang/TokenLabeling
    
    patch embedding
    4-layer conv, kernel size [7,3,3,8], stride [2,1,1,8], filters 64
    [conv-bn-relu]-[conv-bn-relu]-[conv-bn-relu]-[conv-bias]

    re-labeling
    用另一个模型inference training set，给出一个K-dim dense score map
    在训练我们的模型的时候，random crop以后，基于cropped score map重新计算label

    token labeling
    基于re-labeling的dense score map，我们能够进一步地给到每一个token一个独立的K-dim label
    每个token的label和prediction能够独立计算一个CE：auxiliary token labeling loss

    mixtoken
    针对token grids，以cutMix的形式(crop box)，而不是noisy drop
    crop box的长宽是服从beta分布的（大概率落在较小值，从而保证总体的label是beta分布）
    token label是token individual的，所以mixtoken不影响每个token的label学习，所以源代码在计算token loss之前，将crop patches复原，这样token gt labels就不用转换了
    mixtoken本质上还是在augment原图，所以只影响cls token的prediction，cls token要基于随机mask重新计算

    loss
    cls_token对应的out embedding [b,D]接上MLP prediction head，预测总体的类别概率
    其他token对应的out embedding [b,N,D]接上shared MLP prediction head，学习每个token的类别预测，求全部tokens的平均
    再加权求和：cls_loss + 0.5*token_loss

    encoder block
    * stochastic depth (dropblock): random drop by sample
    * residual_scale: 给residual downscale有提升，scale=2

    training details
    * lr: linear scaling by batch 1e-3*batch_size/1024, 5 warmup epochs + cosine decay
    * AdamW: weight_decay=5e-2
    * batch_size: 1024
    * dropout = 0.
    * dropconnect = .1
    * randAug, mixup





    












