## seq2seq model
    
    example: character-level english to french

    ### 数据
    数据集格式: english sentence \t french sentence \t ...
    target: english
    input: french
    起始符&终止符: '\t' + decoder_input_sentence + '\n'
    characabulary: 有字母，数字，空格，标点符号，decoder_input_sentence还有起始符&终止符，decoder_target_sentence比它早一步，没有起始符，有终止符
    one-hot: sentence范围内one-hot for each charac, 其余的填充用空格的one-hot

    ### data dim
    encoder_input_data: 
        [batch_size, max_sentence_length, num_eng_characters]
        each element is a one-hot vec (standing for a specific charac)

    decoder_input_data: 
        [batch_size, max_sentence_length, num_eng_characters]
        each element is a one-hot vec

    decoder_target_data: 
        [batch_size, max_sentence_length, num_eng_characters]
        offset the decoder_input_data by one step: 
            decoder_target比decoder_input早一步，decoder这一步的预测/gt作为下一步的input
            decoder_target_data[:,t,:] = decoder_input_data[:,t+1,:]

    ### model
        model input: 
            encoder input: time-distributed input sequence
            decoder input: target sequence starting with \t
            期望输入数据尺寸: (batch_size, timesteps, data_dim)
        model output: predict sequence, (batch_size, timesteps, data_dim)
        model target: target sequence, (batch_size, timesteps, data_dim)
        encoder_states: LSTM有两个(hidden state和cell state)，和每个时刻的输出维度相同(batch_size, latent_dim)
        initial_state: 用于指定RNN层的初始状态，
            decoder起始时刻输入的是起始符，hidden state就来自encoder编码得到的信息，
            后续每个时刻的输入可以是前一时刻的预测/gt，循环的状态就是逐渐积累的前文信息

        * 如果使用函数式API的Model类模型，我们会定义input layer的shape=(timestep, emb_dim)
        * 如果使用Sequential模型，我们要在第一个LSTM层里显示定义input_shape=(timesteps, data_dim)
        * batch_size is set for axis-0 by default for both the methods above
        * 但是在stateful的LSTM layer里面，batch_size要被显示声明：
            model.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape=(batch_size, timesteps, data_dim)))

    ### teacher-forcing or reinjection
    decoder的输出是预测sequence（gt是与之对应的target sequence），输入是比预测序列one-step-lag-behind的输出序列————你只能拿到并复用已经预测出的东西
    如果用teacher-forcing，decoder的输入是target sequence，
    如果用reinjection，decoder的输入是predicted sequence
    【QUESTION】为啥teacher-forcing更常使用？

    ### inference
    decoder模型在inference阶段，变动比较大:
    1. 要显式的指定每一个time step的hidden state
    2. 要显式的实现逐步预测


    what if:
    * GRU: 用GRU就是少了cell state，内部结构基于LSTM，输出跟常规RNN一样
    * word-level: 
        vocabulary比characabulary要大得多，one-hot embedding维度太大了，而且过于稀疏，
        可以在输入层后面添加Embedding层，将词向量转换成固定尺寸的稠密向量

    further improvements:
    * attention
    * bi-rnn
    * deeper: stacking layers



## Bi-directional RNN
    出发点：
        * LSTM在一定程度上缓解了RNN的长距离依赖问题，但不是完全解决
        * 一个很常见的场景，句子填空，既需要结合上文，又需要参考下文，单向的RNN不能胜任
        * 或者一个图像任务，需要结合全局信息进行分类，单向的RNN获取的信息是不完整的

    实现：将sequence倒序再解析一遍
    
    example: MNIST classification

    keras layer Bidirectional: 
        * https://github.com/keras-team/keras/blob/d71247dcd805e58110a784b03cf2fcbaa1c837c8/keras/layers/wrappers.py
        * fw和bw层的输入是同一个inputs
        * fw和bw层可以是同一个种layer，也可以是不同的layer
        * fw和bw层的输出在emb_dim上concat在一起（也有其他fusion mode），本例中lstm的out dim是128，所以bi-lstm的out dim是256



## attention
    出发点：
        basic的seq2seq模型，encoder从输入序列中编码得到一个context vector([b,dim])，然后在解码阶段，
        这个固定的Context Vector作为initial_state，编码input整体的信息，输入给decoder
        考虑机翻这个场景，翻译当前词的时候不是与输入序列中每个element都是强相关的，绝大多数情况只与对应词相关

    实现：建立输入语句中第j个单词与输出语句中第i个单词的匹配程度，每个step使用一个加权的context vec
        * for each decoding step
        * s是decoder的输出，[1,dim]，当前单词
        * h是encoder的输出，[N,dim]，所有输入词向量
        * $e_{ij} = a(s_{i-1},h_j)$
        * $\alpha_{ij} = \frac{exp(e_{ij})}{\sum_k exp(e_{ik})}$
        * $c_i = \sum_j \alpha_{ij} h_j$


    example: character-level english to french

    这里面的attention是learnable attention，类似se-block，计算每个embedding与其他embedding的线性映射value，然后softmax



## keras MultiHeadAttention layer
    tf2.4.1 keras估计要2.3以上
    https://github.com/keras-team/keras/blob/70d7d07bd186b929d81f7a8ceafff5d78d8bd701/keras/layers/multi_head_attention.py

    given sequence length N, batch size B, key dim d, num_heads m, value_dim dv:

    step1: projects `query`, `key` and `value`, 
    * each is a list of tensors of length `num_attention_heads`
    * each tensor [B, N, d]
    * trainable variables Wq,Wk,Wv(biases)

    step2: compute attention
    * dot(Q,K)
    * scaled by hidden dim d, [B,N,N]
    * softmax to obtain attention probabilities, [B,N,1]
    * dropout，我们的实现中dropout放在MSA layer后面，因为drop的是一整个特征维度，放在哪都行
    * reweight content vec V, [B,N,mdv]

    step3: final dense
    * concat multi-heads along d-axis, [B,N,mdv]
    * linear projection to d, [B,N,dv]



## transformer: attention is all you need
    这里面的attention是transformer attention，established on Q,K,V
    可以用来表征token embedding的不同维度：https://zhuanlan.zhihu.com/p/158952064
    * Q: query，词的查询向量
    * K: key，词的被查向量
    * V: value，词的内容向量

    multi-head self-attention:
    * dq=dk=qv=dmodel/h=64
    * h=8

    encoder:
    N=6, encoder由6个相同的attention block构成
    每个attention block包含：MSA，FF，LN，residual, dropout
        MSA：MultiHeadAttention + add&norm
        FF：dense + relu/gelu + dense
    输入：input embedding + ppositional embedding
    self-attention: single input, x=q=k=v



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















    
















