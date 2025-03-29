import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau


def build_optimizer(model, config):
    """
    构建优化器和学习率调度器
    
    Args:
        model: 模型
        config: 配置参数，包含学习率、权重衰减等
    
    Returns:
        optimizer: 优化器
        scheduler: 学习率调度器
    """
    # 默认参数
    lr = getattr(config, 'lr', 1e-4)
    weight_decay = getattr(config, 'weight_decay', 0.05)
    betas = getattr(config, 'betas', (0.9, 0.999))
    optimizer_type = getattr(config, 'optimizer', 'adamw')
    layer_decay = getattr(config, 'layer_decay', None)
    
    # 如果启用了层级学习率衰减，使用get_layer_wise_lr函数
    if layer_decay is not None:
        param_groups = get_layer_wise_lr(model, lr, weight_decay, layer_decay)
    else:
        # 参数分组 - 对不同类型的参数使用不同的学习率和权重衰减
        param_groups = []
        
        # 对normalization层和bias不使用权重衰减
        no_decay_params = []
        decay_params = []
        
        # 分类器参数 - 可以使用更高的学习率
        classifier_params_decay = []
        classifier_params_no_decay = []
        
        # 融合层参数
        fusion_params_decay = []
        fusion_params_no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'classifier' in name:
                if 'norm' in name or 'bias' in name:
                    classifier_params_no_decay.append(param)
                else:
                    classifier_params_decay.append(param)
            elif 'fusion_layer' in name:
                if 'norm' in name or 'bias' in name:
                    fusion_params_no_decay.append(param)
                else:
                    fusion_params_decay.append(param)
            elif 'norm' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # 基础网络参数
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": lr})
        param_groups.append({"params": decay_params, "weight_decay": weight_decay, "lr": lr})
        
        # 分类器参数 - 使用稍高的学习率
        classifier_lr = lr * 5.0  # 分类器学习率可以更高
        if classifier_params_decay:
            param_groups.append({"params": classifier_params_decay, 
                                "weight_decay": weight_decay, 
                                "lr": classifier_lr})
        if classifier_params_no_decay:
            param_groups.append({"params": classifier_params_no_decay, 
                                "weight_decay": 0.0, 
                                "lr": classifier_lr})
        
        # 融合层参数 - 使用中等学习率
        fusion_lr = lr * 2.0
        if fusion_params_decay:
            param_groups.append({"params": fusion_params_decay, 
                                "weight_decay": weight_decay, 
                                "lr": fusion_lr})
        if fusion_params_no_decay:
            param_groups.append({"params": fusion_params_no_decay, 
                                "weight_decay": 0.0, 
                                "lr": fusion_lr})
    
    # 选择优化器
    if optimizer_type.lower() == 'adamw':
        optimizer = AdamW(param_groups, lr=lr, betas=betas)
    elif optimizer_type.lower() == 'sgd':
        momentum = getattr(config, 'momentum', 0.9)
        optimizer = SGD(param_groups, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    # 学习率调度器
    scheduler_type = getattr(config, 'scheduler', 'cosine')
    warmup_epochs = getattr(config, 'warmup_epochs', 5)
    epochs = getattr(config, 'epochs', 100)
    min_lr = getattr(config, 'min_lr', 1e-6)
    
    if scheduler_type == 'cosine':
        # 先进行warmup，然后使用余弦退火
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=min_lr
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    elif scheduler_type == 'plateau':
        # 根据验证集性能调整学习率
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=True
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    return optimizer, scheduler


def get_layer_wise_lr(model, lr=1e-4, weight_decay=0.05, layer_decay=0.75):
    """
    为不同层设置不同的学习率，支持多模态Swin Transformer模型
    
    Args:
        model: 多模态Swin Transformer模型
        lr: 基础学习率
        weight_decay: 权重衰减
        layer_decay: 层间学习率衰减因子
    
    Returns:
        param_groups: 参数分组
    """
    param_groups = []
    
    # 处理多个Swin Transformer模型
    for i, swin_transformer in enumerate(model.swin_transformers):
        # 获取模型的所有层
        num_layers = len(swin_transformer.layers1) + len(swin_transformer.layers2) + \
                     len(swin_transformer.layers3) + len(swin_transformer.layers4)
        
        # 为不同层设置不同的学习率
        layer_index = 0
        
        # 处理patch embedding层
        patch_embed_params_decay = []
        patch_embed_params_no_decay = []
        
        for name, param in swin_transformer.patch_embed.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'norm' in name or 'bias' in name:
                patch_embed_params_no_decay.append(param)
            else:
                patch_embed_params_decay.append(param)
        
        # Patch embedding层使用最低的学习率
        patch_embed_lr = lr * (layer_decay ** num_layers)
        
        if patch_embed_params_decay:
            param_groups.append({
                "params": patch_embed_params_decay,
                "lr": patch_embed_lr,
                "weight_decay": weight_decay,
                "layer_name": f"swin_{i}_patch_embed"
            })
        
        if patch_embed_params_no_decay:
            param_groups.append({
                "params": patch_embed_params_no_decay,
                "lr": patch_embed_lr,
                "weight_decay": 0.0,
                "layer_name": f"swin_{i}_patch_embed_no_decay"
            })
        
        # 处理各个stage的层
        for stage_idx, stage_layers in enumerate([
            swin_transformer.layers1, 
            swin_transformer.layers2, 
            swin_transformer.layers3, 
            swin_transformer.layers4
        ]):
            for j, layer in enumerate(stage_layers):
                # 深层使用较大学习率
                layer_lr = lr * (layer_decay ** (num_layers - layer_index - 1))
                layer_index += 1
                
                # 对该层的参数进行分组
                decay_params = []
                no_decay_params = []
                
                for name, param in layer.named_parameters():
                    if not param.requires_grad:
                        continue
                        
                    if 'norm' in name or 'bias' in name:
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
                
                if decay_params:
                    param_groups.append({
                        "params": decay_params,
                        "lr": layer_lr,
                        "weight_decay": weight_decay,
                        "layer_name": f"swin_{i}_stage_{stage_idx}_layer_{j}"
                    })
                
                if no_decay_params:
                    param_groups.append({
                        "params": no_decay_params,
                        "lr": layer_lr,
                        "weight_decay": 0.0,
                        "layer_name": f"swin_{i}_stage_{stage_idx}_layer_{j}_no_decay"
                    })
    
    # 处理融合层
    fusion_params_decay = []
    fusion_params_no_decay = []
    
    if hasattr(model, 'fusion_layer'):
        for name, param in model.fusion_layer.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'norm' in name or 'bias' in name:
                fusion_params_no_decay.append(param)
            else:
                fusion_params_decay.append(param)
        
        # 融合层使用较高的学习率
        fusion_lr = lr * 2.0
        
        if fusion_params_decay:
            param_groups.append({
                "params": fusion_params_decay,
                "lr": fusion_lr,
                "weight_decay": weight_decay,
                "layer_name": "fusion_layer"
            })
        
        if fusion_params_no_decay:
            param_groups.append({
                "params": fusion_params_no_decay,
                "lr": fusion_lr,
                "weight_decay": 0.0,
                "layer_name": "fusion_layer_no_decay"
            })
    
    # 处理分类器头
    classifier_params_decay = []
    classifier_params_no_decay = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name and param.requires_grad:
            if 'norm' in name or 'bias' in name:
                classifier_params_no_decay.append(param)
            else:
                classifier_params_decay.append(param)
    
    # 分类器使用最高的学习率
    classifier_lr = lr * 5.0
    
    if classifier_params_decay:
        param_groups.append({
            "params": classifier_params_decay,
            "lr": classifier_lr,
            "weight_decay": weight_decay,
            "layer_name": "classifier"
        })
    
    if classifier_params_no_decay:
        param_groups.append({
            "params": classifier_params_no_decay,
            "lr": classifier_lr,
            "weight_decay": 0.0,
            "layer_name": "classifier_no_decay"
        })
    
    return param_groups