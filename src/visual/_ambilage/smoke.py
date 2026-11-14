"""显式微型夹具。仅用于连通性验证，产物禁止进入正式评估。"""
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from .latent_refinement import AmbiLAGEVae
from .segmentation import _heuristic_segmentation

class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=16, scaling_factor=.5, shift_factor=.1)
    def encode(self, x):
        value = F.avg_pool2d(x, 4).repeat(1, 6, 1, 1)[:, :16]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))
    def decode(self, z):
        return SimpleNamespace(sample=F.interpolate(z[:, :3], scale_factor=4, mode='bilinear', align_corners=False))

class TinyTokenizer:
    context_length=77
    sot_token_id=1
    eot_token_id=2
    def encode(self, text):
        return [3+ord(c)%61 for c in text]

    def __call__(self, texts, context_length=None):
        length=context_length or self.context_length
        output=torch.zeros((len(texts),length),dtype=torch.long)
        for i,text in enumerate(texts):
            tokens=[self.sot_token_id,*self.encode(text),self.eot_token_id]
            if len(tokens)>length:
                raise ValueError('合成分词器拒绝截断')
            output[i,:len(tokens)]=torch.tensor(tokens)
        return output

class TinyText(nn.Module):
    context_length=77
    def encode_text(self, tokens):
        mean = tokens.float().mean(-1, keepdim=True)/64
        return 1.+torch.cos(mean + torch.arange(12,device=tokens.device)[None]/12)

class TinySurrogate:
    def __init__(self, index=0):
        self.device=torch.device('cpu');self.index=index
        self.tokenizer=TinyTokenizer();self.model=TinyText()
    def encode_image(self, image):
        values=F.adaptive_avg_pool2d(image,(2,2)).flatten(1)+.1
        return SimpleNamespace(global_embedding=F.normalize(values.roll(self.index,-1),dim=-1),patch_tokens=values[:,None])
    def snapshot(self):
        return {'name':f'tiny_test_{self.index}','test_fixture':True}

class TinyPurifier:
    config=SimpleNamespace(reverse_steps=10)
    def __call__(self,image,*,seed,gradient):
        value=image
        for _ in range(10):
            value=.9*value+.1*F.avg_pool2d(F.pad(value,(1,1,1,1),mode='replicate'),3,stride=1)
        return value
    def snapshot(self):
        return {'kind':'tiny_test_fixture','reverse_steps':10,'test_fixture':True}

class SmokeStages:
    test_fixture=True
    def __init__(self, config, device):
        if torch.device(device).type!='cpu' or config.steps>3:
            raise ValueError('轻量夹具仅允许中央处理器及至多三步优化')
        self.config=config
    def close(self):
        pass
    def plan(self, source, category):
        self.source=np.asarray(Image.open(source).convert('RGB')).copy()
        return dict(source_boxes=[[0.,0.,1.,1.]],print_regions=[],edit_prompt='explicit tiny fixture',test_fixture=True)
    def detect_print(self,source):
        return [],{'test_fixture':True,'backend':'none_declared_synthetic'}
    def segment(self,image,boxes):
        image=np.asarray(Image.open(image).convert('RGB'))
        if max(image.shape[:2])>64:
            raise ValueError('轻量夹具图像长边不能超过六十四')
        return _heuristic_segmentation(image),{'test_fixture':True,'segmenter':'synthetic_fixture'}
    def generate_anchor(self, initial, prompt, seed):
        return self.source.copy(),{'test_fixture':True,'backend':'identity_fixture'}
    def gate(self,source,candidate,category):
        return {'passed':True,'reasons':['synthetic fixture only'],'test_fixture':True}
    def make_vae(self):
        return AmbiLAGEVae('cpu',model=TinyAutoencoder(),test_fixture=True)
