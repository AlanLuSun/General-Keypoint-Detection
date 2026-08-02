# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Any, Tuple, Union
from enum import Enum

import torch
from torch import nn

from .backbones import dinov3_vitl16, Weights as BackboneWeights, convert_path_or_url_to_url
from .utils import DINOV3_BASE_URL


class DINOTxtWeights(Enum):
    LVTD2300M = "LVTD2300M"


# returns dinotxt model and tokenizer
def dinov3_vitl16_dinotxt_tet1280d20h24l(
    *,
    pretrained: bool = True,
    weights: Union[DINOTxtWeights, str] = DINOTxtWeights.LVTD2300M,
    backbone_weights: Union[BackboneWeights, str] = BackboneWeights.LVD1689M,
    bpe_path_or_url: str = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz",
    check_hash: bool = False,
) -> Tuple[nn.Module, Any]:
    from dinov3.eval.text.dinotxt_model import DINOTxt, DINOTxtConfig
    from dinov3.eval.text.text_transformer import TextTransformer
    from dinov3.eval.text.tokenizer import get_tokenizer

    dinotxt_config = DINOTxtConfig(
        embed_dim=2048,
        vision_model_freeze_backbone=True,
        vision_model_train_img_size=224,
        vision_model_use_class_token=True,
        vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean",
        vision_model_patch_token_layer=1,  # which layer to take patch tokens from
        # 1 - last layer, 2 - second last layer, etc.
        text_model_freeze_backbone=False,
        text_model_num_head_blocks=0,
        text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None,
        freeze_logit_scale=False,
    )
    vision_backbone = dinov3_vitl16(pretrained=pretrained, weights=backbone_weights)
    text_backbone = TextTransformer(
        context_length=77,
        vocab_size=49408,
        dim=1280,
        num_heads=20,
        num_layers=24,
        ffn_ratio=4,
        is_causal=True,
        ls_init_value=None,
        dropout_prob=0.0,
    )
    model = DINOTxt(model_config=dinotxt_config, vision_backbone=vision_backbone, text_backbone=text_backbone)
    if pretrained:
        model.visual_model.backbone = vision_backbone
        model.eval()
        if type(weights) is DINOTxtWeights and weights == DINOTxtWeights.LVTD2300M:
            url = f"{DINOV3_BASE_URL}/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
        elif type(weights) is DINOTxtWeights and weights != DINOTxtWeights.LVTD2300M:
            raise AssertionError(f"Unsuported weights for DINOTxt: {weights}")
        else:
            url = convert_path_or_url_to_url(weights)
        vision_head_and_text_encoder_state_dict = torch.hub.load_state_dict_from_url(url, check_hash=check_hash)
        model.load_state_dict(vision_head_and_text_encoder_state_dict, strict=False)
    else:
        model.init_weights()
    return model, get_tokenizer(bpe_path_or_url=bpe_path_or_url)

#only load text encoder use original build function
def dinov3_vitl16_dinotxt(
    *,
    pretrained: bool = True,
    weights: Union[DINOTxtWeights, str] = DINOTxtWeights.LVTD2300M,
    bpe_path_or_url: str = "/project/vonneumann1/cl2025/GKD/network/dinov3_kd/bpe_simple_vocab_16e6.txt.gz",
    check_hash: bool = False,
    text_layer_to_tune: int = None,  # int 类型 -1, all tune; 0, no layer to tune (all freeze); 1, proj to tune; >=2, proj + the last n-1 CausalSelfAttentionBlock layers to tune
) -> Tuple[nn.Module, Any]:
    from dinov3.eval.text.dinotxt_model import DINOTxt, DINOTxtConfig
    from dinov3.eval.text.text_transformer import TextTransformer
    from dinov3.eval.text.tokenizer import get_tokenizer
    
    # 创建与完整模型相同的配置
    dinotxt_config = DINOTxtConfig(
        embed_dim=2048,
        vision_model_freeze_backbone=True,
        vision_model_train_img_size=224,
        vision_model_use_class_token=True,
        vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean",
        vision_model_patch_token_layer=1,
        text_model_freeze_backbone=False,
        text_model_num_head_blocks=0,
        text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None,
        freeze_logit_scale=False,
    )
    
    # 创建文本骨干网络
    text_backbone = TextTransformer(
        context_length=77,
        vocab_size=49408,
        dim=1280,
        num_heads=20,
        num_layers=24,
        ffn_ratio=4,
        is_causal=True,
        ls_init_value=None,
        dropout_prob=0.0,
    )
    
    # 使用与完整模型相同的构建方式构建文本模型
    from dinov3.eval.text.dinotxt_model import build_text_model
    text_model = build_text_model(
        dinotxt_config.embed_dim,
        dinotxt_config.text_backbone_config,
        dinotxt_config.text_model_freeze_backbone,
        dinotxt_config.text_model_num_head_blocks,
        dinotxt_config.text_model_head_blocks_is_causal,
        dinotxt_config.text_model_head_blocks_drop_prob,
        dinotxt_config.text_model_tokens_pooler_type,
        dinotxt_config.text_model_use_linear_projection,
        backbone=text_backbone,
    )
    
    # 如果预训练，加载权重到文本模型
    if pretrained:
        if type(weights) is DINOTxtWeights and weights == DINOTxtWeights.LVTD2300M:
            url = f"{DINOV3_BASE_URL}/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
        elif type(weights) is DINOTxtWeights and weights != DINOTxtWeights.LVTD2300M:
            raise AssertionError(f"Unsupported weights for DINOTxt: {weights}")
        else:
            url = convert_path_or_url_to_url(weights)
            
        # 加载完整的状态字典
        full_state_dict = torch.hub.load_state_dict_from_url(url, check_hash=check_hash)
        
        # 正确的权重键名映射
        text_model_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('text_model.'):
                # 移除 'text_model.' 前缀，直接映射到模型结构
                new_key = key.replace('text_model.', '')
                text_model_state_dict[new_key] = value
        
        # 加载权重到文本模型
        if text_model_state_dict:
            missing_keys, unexpected_keys = text_model.load_state_dict(text_model_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in text model: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in text model: {unexpected_keys}")
            
            # 检查加载结果
            total_params = len(text_model.state_dict())
            loaded_params = len([k for k in text_model_state_dict.keys() if k in text_model.state_dict()])
            print(f"Successfully loaded {loaded_params}/{total_params} parameters for text model")
            
            del text_model_state_dict  #释放中间变量内存
        else:
            print("Warning: No text model weights found in the pretrained model")
            # 如果没有找到匹配的权重，初始化模型权重
            text_model.init_weights()

        del full_state_dict  #释放中间变量内存
    
    # 根据 text_layer_to_tune 参数设置微调策略
    if text_layer_to_tune is not None:
        _setup_text_model_finetuning(text_model, text_layer_to_tune)

    # 获取 tokenizer
    tokenizer = get_tokenizer(bpe_path_or_url=bpe_path_or_url)
    
    return text_model, tokenizer

def _setup_text_model_finetuning(text_model: nn.Module, text_layer_to_tune: int):
    """
    根据 text_layer_to_tune 参数设置文本模型的微调策略
    
    Args:
        text_model: 文本模型
        text_layer_to_tune: 微调层数控制参数
            -1: 所有层都微调
            0: 所有层都冻结
            1: 只微调 head.linear_projection
            >=2: 微调 head.linear_projection + ln_final + 倒数 n-1 个 CausalSelfAttentionBlock 层
    """
    # 首先冻结所有参数
    for param in text_model.parameters():
        param.requires_grad = False
    
    # 如果 text_layer_to_tune == -1，解冻所有参数
    if text_layer_to_tune == -1:
        for param in text_model.parameters():
            param.requires_grad = True
        print("All text model parameters are set to trainable")
        return
    
    # 如果 text_layer_to_tune == 0，保持所有参数冻结
    if text_layer_to_tune == 0:
        print("All text model parameters are frozen")
        return
    
    # 解冻 head.linear_projection (总是需要解冻)
    for name, param in text_model.head.named_parameters():
        if 'linear_projection' in name:
            param.requires_grad = True
    print("head.linear_projection is set to trainable")
    
    # 如果 text_layer_to_tune == 1，只解冻 linear_projection
    if text_layer_to_tune == 1:
        return
    
    # 如果 text_layer_to_tune >= 2，解冻 ln_final + 最后 n-1 个 CausalSelfAttentionBlock 层
    num_layers = len(text_model.backbone.blocks)
    num_layers_to_tune = text_layer_to_tune - 1  # 减去 linear_projection
    
    # 解冻 ln_final LayerNorm
    for name, param in text_model.backbone.ln_final.named_parameters():
        param.requires_grad = True
    print("backbone.ln_final (LayerNorm) is set to trainable")
    
    if num_layers_to_tune > num_layers:
        print(f"Warning: text_layer_to_tune={text_layer_to_tune} exceeds total layers {num_layers}. Tuning all {num_layers} layers.")
        num_layers_to_tune = num_layers
    
    # 解冻最后 num_layers_to_tune 个 CausalSelfAttentionBlock 层
    start_layer = num_layers - num_layers_to_tune
    for i in range(start_layer, num_layers):
        for name, param in text_model.backbone.blocks[i].named_parameters():
            param.requires_grad = True
        print(f"backbone.blocks[{i}] (CausalSelfAttentionBlock) is set to trainable")
    
    # 打印统计信息
    total_params = sum(p.numel() for p in text_model.parameters())
    trainable_params = sum(p.numel() for p in text_model.parameters() if p.requires_grad)
    #print(f"Text model: {trainable_params}/{total_params} parameters are trainable ({trainable_params/total_params*100:.2f}%)")