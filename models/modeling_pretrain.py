# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn import functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange

from .modeling_finetune import (
    Block,
    PatchEmbed,
    _cfg,
    get_sinusoid_encoding_table,
)


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class NonLinearNeck(nn.Module):
    """The non-linear neck.

    Structure: fc-bn-[relu-fc-bn] where the substructure in [] can be repeated.
    For the default setting, the repeated time is 1.
    The neck can be used in many algorithms, e.g., SimCLR, BYOL, SimSiam.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of fc layers. Defaults to 2.
        with_bias (bool): Whether to use bias in fc layers (except for the
            last). Defaults to False.
        with_last_bn (bool): Whether to add the last BN layer.
            Defaults to True.
        with_last_bn_affine (bool): Whether to have learnable affine parameters
            in the last BN layer (set False for SimSiam). Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 with_bias=False,
                 with_last_bn=True,
                 with_last_bn_affine=True,
                 with_last_bias=False,
                 with_avg_pool=True,):
        super(NonLinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn0 = nn.SyncBatchNorm(hid_channels, eps=1e-6)

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(hid_channels, this_channels, bias=with_bias))
                self.add_module(f'bn{i}',
                                nn.SyncBatchNorm(this_channels, eps=1e-6))
                self.bn_names.append(f'bn{i}')
            else:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(
                        hid_channels, this_channels, bias=with_last_bias))
                if with_last_bn:
                    self.add_module(
                        f'bn{i}', 
                        nn.SyncBatchNorm(this_channels, affine=with_last_bn_affine, eps=1e-6))
                    self.bn_names.append(f'bn{i}')
                else:
                    self.bn_names.append(None)
            self.fc_names.append(f'fc{i}')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avgpool(x)
        else:
            x = x[:,0,:]
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.bn0(x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            x = fc(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                x = bn(x)
        return x.unsqueeze(dim=1)


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 final_norm=True, 
                 init_values=None,
                 tubelet_size=2,
                 use_learnable_pos_emb=False,
                 with_cp=False,
                 all_frames=16,
                 cos_attn=False):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) if final_norm else nn.Identity()
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        for blk in self.blocks:
            if self.with_cp:
                x_vis = cp.checkpoint(blk, x_vis)
            else:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 patch_size=16,
                 num_classes=768,
                 encoder_embed_dim=768, 
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 num_patches=196,
                 tubelet_size=2,
                 with_cp=False,
                 cos_attn=False):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * tubelet_size * patch_size**2
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.with_cp = with_cp

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, embed_dim, bias=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, mask, decode_mask=None):
        decode_vis = mask if decode_mask is None else ~decode_mask

        x = self.encoder_to_decoder(x)  # [B, N_vis, C_d]
        B, _, C = x.shape
        
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(
            x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[decode_vis].reshape(B, -1, C)

        # [B, N, C_d]
        x = torch.cat(
            [x + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        # NOTE: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        for blk in self.blocks:
            if self.with_cp:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)

        return_token_num = pos_emd_mask.shape[1]
        if return_token_num > 0:
            # only return the mask tokens predict pixels
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            # [B, N, 3*16^2]
            x = self.head(self.norm(x))
        return x


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,  # decoder_num_classes=768
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        teacher_model=None,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        use_learnable_pos_emb=False,
        tubelet_size=2,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
        with_cp=False,
        all_frames=16,
        cos_attn=False,
        momentum=0.996, 
        ct_weight=1.0, rc_weight=1.0, 
        loss_type='infoNCE', 
        temperature=0.07, 
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            with_cp=with_cp,
            all_frames=all_frames,
            cos_attn=cos_attn)
        self.target_encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            final_norm=False,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            with_cp=with_cp,
            all_frames=all_frames,
            cos_attn=cos_attn)

        self.feature_decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=encoder_embed_dim,
            encoder_embed_dim=encoder_embed_dim, 
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            with_cp=with_cp,
            cos_attn=cos_attn)

        self.teacher_projector = nn.Linear(
            in_features=encoder_embed_dim, 
            out_features=teacher_model.encoder.embed_dim, 
            bias=False
        )
        self.projector = NonLinearNeck(
            in_channels=encoder_embed_dim,
            hid_channels=encoder_embed_dim*2,
            out_channels=encoder_embed_dim//3,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False
        )
        self.target_projector = NonLinearNeck(
            in_channels=encoder_embed_dim,
            hid_channels=encoder_embed_dim*2,
            out_channels=encoder_embed_dim//3,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False
        )

        self.predictor = NonLinearNeck(
            in_channels=encoder_embed_dim//3,
            hid_channels=encoder_embed_dim*2,
            out_channels=encoder_embed_dim//3,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False
        )
        
        self.momentum = momentum
        self.ct_weight = ct_weight
        self.rc_weight = rc_weight
        self.loss_type = loss_type
        self.t = temperature

        for param_b, param_m in zip(self.encoder.parameters(),
                                    self.target_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
        
        for param_b, param_m in zip(self.projector.parameters(),
                                    self.target_projector.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
            
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        for param_b, param_m in zip(self.encoder.parameters(),
                                    self.target_encoder.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)

        for param_b, param_m in zip(self.projector.parameters(),
                                    self.target_projector.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                    1. - self.momentum)

    def forward_loss(self, pred, target, mask, decode_mask, proj_s, proj_t):
        loss_rc = F.smooth_l1_loss(pred, target, reduction='none')
        loss_rc = loss_rc.mean(dim=-1)
        cal_loss_mask = mask[~decode_mask].reshape(pred.shape[0], -1)
        loss_rc = (loss_rc * cal_loss_mask).sum() / cal_loss_mask.sum()
        
        pred_s = self.predictor(proj_s)
        pred_s = F.normalize(pred_s.squeeze(1), dim=-1, p=2)
        proj_t = F.normalize(proj_t.squeeze(1), dim=-1, p=2)
        if self.loss_type == 'byol':
            loss_ct = (2 - 2 * (pred_s * proj_t.detach()).sum(dim=-1)).mean()
        elif self.loss_type == 'infoNCE':
            proj_t = concat_all_gather(proj_t)

            score = torch.matmul(pred_s,proj_t.transpose(1, 0).detach())

            score = score / self.t

            bs = score.size(0)
            label = (torch.arange(bs, dtype=torch.long) +
                    bs * torch.distributed.get_rank()).cuda()

            loss_ct = 2 * self.t * F.cross_entropy(score, label)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} is not implemented")
        
        return self.rc_weight * loss_rc, self.ct_weight * loss_ct

    def forward(self, x_s, x_t, gt, mask, decode_mask=None):
        gt = self.teacher_model(x_s, decode_mask)  # [B, N, C_e]
        x_s = self.encoder(x_s, mask)  # [B, N_vis, C_e]
        x_t = self.target_encoder(x_t, decode_mask)  # [B, N, C_e]
        
        pred_feature = self.feature_decoder(x_s, mask, decode_mask)
        pred_pixel = self.teacher_projector(pred_feature)

        proj_s = self.projector(torch.mean(pred_feature, dim=1, keepdim=True))
        proj_t = self.target_projector(torch.mean(x_t, dim=1, keepdim=True))
        
        loss = self.forward_loss(pred_pixel, gt, mask, decode_mask, proj_s, proj_t)

        return loss

    @torch.no_grad()
    def inference(self, x, mask, decode_mask=None):
        x = self.encoder(x, mask)
        pred_pixel = self.decoder(x, mask, decode_mask)
        return pred_pixel


@register_model
def pretrain_videomae_small_patch8_112(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=112,
        patch_size=8,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=384,  # 8 * 8 * 3 * 2
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # 16 * 16 * 3 * 2
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # 16 * 16 * 3 * 2
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # 16 * 16 * 3 * 2
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,  # 16 * 16 * 3 * 2
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_videomae_giant_patch14_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=14,
        encoder_embed_dim=1408,
        encoder_depth=40,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1176,  # 14 * 14 * 3 * 2,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
