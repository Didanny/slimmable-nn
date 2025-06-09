import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Union, List, Dict, Any, cast
from slimmable_networks.models.slimmable_ops import USBatchNorm2d, USConv2d, USLinear

cifar10_pretrained_weight_urls = {
    'vit': 'https://github.com/Didanny/activation-sparsity/releases/download/vit/cifar10_vit.pt',
}

cifar100_pretrained_weight_urls = {
    'vit': 'https://github.com/Didanny/activation-sparsity/releases/download/vit/vit_cifar100.pt',
}

tinyimagenet_pretrained_weight_urls = {
    'vit': 'https://github.com/Didanny/activation-sparsity/releases/download/vit/tinyimagenet_vit.pt',
}

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            USLinear(feats, mlp_hidden, us=[False, True]),
            nn.GELU(),
            nn.Dropout(dropout),
            USLinear(mlp_hidden, feats, us=[True, False]),
            # nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o
    
class ViT(nn.Module):
    def __init__(
        self, 
        in_c:int=3, 
        num_classes:int=10, 
        img_size:int=32, 
        patch:int=8, 
        dropout:float=0., 
        num_layers:int=7,
        hidden:int=384, 
        mlp_hidden:int=384*4, 
        head:int=8, 
        is_cls_token:bool=True):
        
        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )


    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out
    
def _vit(
    arch: str,
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    image_size: int = 32,
    **kwargs: Any
) -> ViT:
    model = ViT(
        in_c=3,
        img_size=image_size,
        patch=8,
        dropout=0.0,
        num_layers=7,
        hidden=384,
        mlp_hidden=384,
        head=12,
        **kwargs,
    )
    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              map_location='cpu')
        model.load_state_dict(checkpoint['params'])
    return model

    
def cifar10_usvit(*args, **kwargs) -> ViT: pass
def cifar100_usvit(*args, **kwargs) -> ViT: pass
def svhn_usvit(*args, **kwargs) -> ViT: pass
def tinyimagenet_usvit(*args, **kwargs) -> ViT: pass

thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100", "svhn", "tinyimagenet"]:
    model_name = 'usvit'
    method_name = f"{dataset}_{model_name}"
    if dataset == "cifar100":
        model_urls = cifar100_pretrained_weight_urls
        num_classes = 100
        image_size = 32
    elif dataset == "tinyimagenet":
        model_urls = tinyimagenet_pretrained_weight_urls
        num_classes = 200
        image_size = 64
    elif dataset == "cifar10":
        model_urls = cifar10_pretrained_weight_urls
        num_classes = 10
        image_size = 32
    setattr(
        thismodule,
        method_name,
        partial(_vit,
                arch=model_name,
                model_urls=model_urls,
                num_classes=num_classes,
                image_size=image_size)
    )