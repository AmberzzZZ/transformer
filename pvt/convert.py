# convert pretrained MiT into keras weights
import torch
import numpy as np
from mit import mit_b0, mit_b2


model_weights = torch.load("weights/mit_b2.pth")   # order_dict
print('total variables', len(model_weights.keys()))


# aggregate
cnt = 0
for b in range(1,5):   # block1-block4
    print('=================== block %d =================' % b)

    print('------------- patch embedding --------------')
    patchEmb_weights = {k:v for k,v in model_weights.items() if 'patch_embed%d' % (b) in k}
    print({k:v.shape for k,v in patchEmb_weights.items()})    # conv-bias, norm-bias
    cnt += len(patchEmb_weights.keys())

    print('------------- EMSA layers --------------')
    # 2 attention blocks(MSA+MLP)
    block_weights = {k:v for k,v in model_weights.items() if 'block%d' % (b) in k}
    # print({k:v.shape for k,v in block_weights.items()})
    print({k for k,v in block_weights.items()})
    cnt += len(block_weights.keys())

    print('------------- norm --------------')
    norm_weights = {k:v for k,v in model_weights.items() if k[:4]=='norm' and 'norm%d' % (b) in k}
    print({k:v.shape for k,v in norm_weights.items()})    # norm-weight, norm-bias
    cnt += len(norm_weights.keys())

    print('=================== end of block %d =================' % b)


print('------------- cls head --------------')
head = {k:v for k,v in model_weights.items() if 'head' in k}
print({k:v.shape for k,v in head.items()})    # conv-weight, conv-bias
cnt += len(head.keys())

print('total cnt', cnt)
assert cnt==len(model_weights.keys()), 'number of variables not match'


keras_model = mit_b2(input_shape=(512,512,3), n_classes=1000)
sr_ratios=[8, 4, 2, 1]
for layer in keras_model.layers:
    if not layer.get_weights():
        continue
    print('------------- layer name: %s -------------' % layer.name)
    if 'patch_embed' in layer.name:
        # conv weight & bias
        # ln weight & bias
        stage_id, layer_ = layer.name.split('.')
        stage_id = int(stage_id.strip('patch_embed'))
        layer_ = {'conv': 'proj', 'norm': 'norm'}[layer_]
        torch_weights = {k:v for k,v in model_weights.items() if 'patch_embed%d.%s' % (stage_id,layer_) in k}
        # print({k:v.shape for k,v in torch_weights.items()})    # conv-bias, norm-bias
        keras_weights = [np.transpose(v,(2,3,1,0)) if 'proj.weight' in k else v for k,v in torch_weights.items()]
        print('setting weights...')
        layer.set_weights(keras_weights)

    elif 'EAttB' in layer.name:
        stage_id, block_id = layer.name.split('.')
        stage_id = int(stage_id.strip('EAttB'))
        block_id = int(block_id)
        torch_weights = {k:v for k,v in model_weights.items() if 'block%d.%d' % (stage_id,block_id) in k}
        # print({k:v.shape for k,v in torch_weights.items()})

        # norm1 & norm2: redidual norm
        norm1_weights = [torch_weights['block%d.%d.norm1.weight' % (stage_id,block_id)],
                         torch_weights['block%d.%d.norm1.bias' % (stage_id,block_id)]]
        norm2_weights = [torch_weights['block%d.%d.norm2.weight' % (stage_id,block_id)],
                         torch_weights['block%d.%d.norm2.bias' % (stage_id,block_id)]]
        mlp_weights = [np.transpose(torch_weights['block%d.%d.mlp.fc1.weight' % (stage_id,block_id)], (1,0)),
                       torch_weights['block%d.%d.mlp.fc1.bias' % (stage_id,block_id)],
                       np.transpose(torch_weights['block%d.%d.mlp.fc2.weight' % (stage_id,block_id)], (1,0)),
                       torch_weights['block%d.%d.mlp.fc2.bias' % (stage_id,block_id)],
                       np.transpose(torch_weights['block%d.%d.mlp.dwconv.dwconv.weight' % (stage_id,block_id)], (2,3,0,1)),
                       torch_weights['block%d.%d.mlp.dwconv.dwconv.bias' % (stage_id,block_id)]]
        attn_weights = [np.transpose(torch_weights['block%d.%d.attn.q.weight' % (stage_id,block_id)], (1,0)),
                        torch_weights['block%d.%d.attn.q.bias' % (stage_id,block_id)],
                        np.transpose(torch_weights['block%d.%d.attn.kv.weight' % (stage_id,block_id)], (1,0)),
                        torch_weights['block%d.%d.attn.kv.bias' % (stage_id,block_id)],
                        np.transpose(torch_weights['block%d.%d.attn.proj.weight' % (stage_id,block_id)], (1,0)),
                        torch_weights['block%d.%d.attn.proj.bias' % (stage_id,block_id)]]
        if sr_ratios[stage_id-1]>1:
            attn_weights += [np.transpose(torch_weights['block%d.%d.attn.sr.weight' % (stage_id,block_id)], (2,3,1,0)),
                             torch_weights['block%d.%d.attn.sr.bias' % (stage_id,block_id)],
                             torch_weights['block%d.%d.attn.norm.weight' % (stage_id,block_id)],
                             torch_weights['block%d.%d.attn.norm.bias' % (stage_id,block_id)]]

        cnt = 0
        for sub_l in layer.layers:
            if sub_l.weights:
                print('------ sub layer: ', sub_l.name)
                # print([i.shape for i in sub_l.weights])
                keras_weights = [norm1_weights, attn_weights, norm2_weights, mlp_weights][cnt]
                # print([i.shape for i in keras_weights])
                sub_l.set_weights(keras_weights)
                cnt += 1

    elif layer.name[:4]=='norm':
        stage_id = int(layer.name[-1])
        torch_weights = {k:v for k,v in model_weights.items() if k[:4]=='norm' and 'norm%d' % (stage_id) in k}
        # print({k:v.shape for k,v in torch_weights.items()})    # norm-weight, norm-bias
        print('setting weights...')
        layer.set_weights(torch_weights.values())

    else:  # the final fc
        torch_weights = {k:v for k,v in model_weights.items() if 'head' in k}
        # print({k:v.shape for k,v in torch_weights.items()})    # norm-weight, norm-bias
        keras_weights = [np.transpose(v,(1,0)) if 'weight' in k else v for k,v in torch_weights.items()]
        print('setting weights...')
        layer.set_weights(keras_weights)
keras_model.save_weights("weights/mit_b2.h5")





#




