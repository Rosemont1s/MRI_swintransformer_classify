import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from layers.build import MultiInputSwinTransformerForClassification
from optimizers import build_optimizer
from dataset import MultiModalMRIDataset  # 假设您已经创建了这个数据集类

class Config:
    """配置类，用于存储训练参数"""
    def __init__(self):
        # 模型参数
        self.img_size = (240, 240, 155)  # MRI图像尺寸
        self.num_classes = 2  # 二分类问题
        self.in_channels = 1  # 每个模态的输入通道数
        self.out_channels = 768  # 特征维度
        self.num_modalities = 5  # 4个MRI + 1个掩码
        self.feature_size = 48  # 特征大小
        self.drop_rate = 0.0  # 基础dropout率
        self.attn_drop_rate = 0.0  # 注意力dropout率
        self.dropout_path_rate = 0.1  # 路径dropout率
        self.classifier_drop_rate = 0.3  # 分类器dropout率
        self.fusion_method = "concat"  # 特征融合方法: concat, add, attention
        
        # 训练参数
        self.batch_size = 2  # 批次大小
        self.epochs = 100  # 训练轮数
        self.lr = 1e-4  # 基础学习率
        self.weight_decay = 0.05  # 权重衰减
        self.optimizer = "adamw"  # 优化器类型
        self.scheduler = "cosine"  # 学习率调度器类型
        self.warmup_epochs = 5  # 预热轮数
        self.min_lr = 1e-6  # 最小学习率
        self.layer_decay = 0.75  # 层级学习率衰减因子
        self.clip_grad_norm = 1.0  # 梯度裁剪
        
        # 数据参数
        self.data_dir = '/home/yankai/MRI_swintransformer_classify-main/data/data_with_seg' # 数据目录
        self.label_file = "labels.csv"  # 标签文件
        self.label_type = "both"  # 标签类型: "who", "ki67", "both"
        self.train_ratio = 0.8  # 训练集比例
        self.val_ratio = 0.1  # 验证集比例
        self.test_ratio = 0.1  # 测试集比例
        self.num_workers = 4  # 数据加载线程数
        
        # 其他参数
        self.seed = 42  # 随机种子
        self.save_dir = "checkpoints"  # 模型保存目录
        self.log_dir = "logs"  # 日志保存目录
        self.save_freq = 5  # 每多少轮保存一次模型
        self.eval_freq = 1  # 每多少轮评估一次模型
        self.use_amp = True  # 是否使用混合精度训练
        self.resume = ""  # 恢复训练的模型路径
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备


def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(config):
    """创建数据加载器"""
    # 创建数据集
    dataset = MultiModalMRIDataset(
        data_dir=config.data_dir,
        label_type=config.label_type,
        label_file=config.label_file,
        augment=True,  # 训练集使用数据增强
        target_size=config.img_size
    )
    
    # 创建验证集和测试集（不使用数据增强）
    val_dataset = MultiModalMRIDataset(
        data_dir=config.data_dir,
        label_type=config.label_type,
        label_file=config.label_file,
        augment=False,
        target_size=config.img_size
    )
    
    test_dataset = MultiModalMRIDataset(
        data_dir=config.data_dir,
        label_type=config.label_type,
        label_file=config.label_file,
        augment=False,
        target_size=config.img_size
    )
    
    # 划分数据集
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_end = int(config.train_ratio * dataset_size)
    val_end = train_end + int(config.val_ratio * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def build_model(config):
    """创建模型"""
    model = MultiInputSwinTransformerForClassification(
        img_size=config.img_size,
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_modalities=config.num_modalities,
        feature_size=config.feature_size,
        drop_rate=config.drop_rate,
        attn_drop_rate=config.attn_drop_rate,
        dropout_path_rate=config.dropout_path_rate,
        fusion_method=config.fusion_method,
    )
    
    # 修改分类器的dropout率
    if hasattr(config, 'classifier_drop_rate') and config.classifier_drop_rate > 0:
        model.classifier1[0] = nn.Dropout(p=config.classifier_drop_rate)
        model.classifier2[0] = nn.Dropout(p=config.classifier_drop_rate)
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, config, epoch, scaler=None):
    """训练一个轮次"""
    model.train()
    total_loss = 0
    all_preds_who = []
    all_preds_ki67 = []
    all_labels_who = []
    all_labels_ki67 = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # 将输入和标签移动到设备上
        inputs = [x.to(device) for x in inputs]
        
        # 处理标签
        if config.label_type == "both":
            who_label, ki67_label = labels
            who_label = who_label.to(device)
            ki67_label = ki67_label.to(device)
        else:
            # 单一标签类型
            labels = labels.to(device)
            if config.label_type == "who":
                who_label = labels
                ki67_label = None
            else:  # ki67
                who_label = None
                ki67_label = labels
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        if config.use_amp:
            with autocast():
                outputs_who, outputs_ki67 = model(inputs)
                
                # 计算损失
                loss = 0
                if who_label is not None:
                    loss_who = criterion(outputs_who, who_label)
                    loss += loss_who
                
                if ki67_label is not None:
                    loss_ki67 = criterion(outputs_ki67, ki67_label)
                    loss += loss_ki67
                
                if config.label_type == "both":
                    loss = loss / 2  # 平均两个损失
                
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if config.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs_who, outputs_ki67 = model(inputs)
            
            # 计算损失
            loss = 0
            if who_label is not None:
                loss_who = criterion(outputs_who, who_label)
                loss += loss_who
            
            if ki67_label is not None:
                loss_ki67 = criterion(outputs_ki67, ki67_label)
                loss += loss_ki67
            
            if config.label_type == "both":
                loss = loss / 2  # 平均两个损失
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                
            # 更新参数
            optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 获取预测结果
        if who_label is not None:
            preds_who = torch.argmax(outputs_who, dim=1).cpu().numpy()
            all_preds_who.extend(preds_who)
            all_labels_who.extend(who_label.cpu().numpy())
        
        if ki67_label is not None:
            preds_ki67 = torch.argmax(outputs_ki67, dim=1).cpu().numpy()
            all_preds_ki67.extend(preds_ki67)
            all_labels_ki67.extend(ki67_label.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'avg_loss': total_loss / (batch_idx + 1)
        })
    
    # 计算指标
    metrics = {'loss': total_loss / len(train_loader)}
    
    if who_label is not None:
        metrics['who_acc'] = accuracy_score(all_labels_who, all_preds_who)
    
    if ki67_label is not None:
        metrics['ki67_acc'] = accuracy_score(all_labels_ki67, all_preds_ki67)
    
    return metrics


def validate(model, val_loader, criterion, device, config):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds_who = []
    all_preds_ki67 = []
    all_labels_who = []
    all_labels_ki67 = []
    all_probs_who = []
    all_probs_ki67 = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for inputs, labels in pbar:
            # 将输入和标签移动到设备上
            inputs = [x.to(device) for x in inputs]
            
            # 处理标签
            if config.label_type == "both":
                who_label, ki67_label = labels
                who_label = who_label.to(device)
                ki67_label = ki67_label.to(device)
            else:
                # 单一标签类型
                labels = labels.to(device)
                if config.label_type == "who":
                    who_label = labels
                    ki67_label = None
                else:  # ki67
                    who_label = None
                    ki67_label = labels
            
            # 前向传播
            outputs_who, outputs_ki67 = model(inputs)
            
            # 计算损失
            loss = 0
            if who_label is not None:
                loss_who = criterion(outputs_who, who_label)
                loss += loss_who
            
            if ki67_label is not None:
                loss_ki67 = criterion(outputs_ki67, ki67_label)
                loss += loss_ki67
            
            if config.label_type == "both":
                loss = loss / 2  # 平均两个损失
            
            # 统计
            total_loss += loss.item()
            
            # 获取预测结果
            if who_label is not None:
                probs_who = F.softmax(outputs_who, dim=1).cpu().numpy()
                preds_who = np.argmax(probs_who, axis=1)
                all_preds_who.extend(preds_who)
                all_labels_who.extend(who_label.cpu().numpy())
                all_probs_who.extend(probs_who[:, 1])  # 假设正类的概率在第二列
            
            if ki67_label is not None:
                probs_ki67 = F.softmax(outputs_ki67, dim=1).cpu().numpy()
                preds_ki67 = np.argmax(probs_ki67, axis=1)
                all_preds_ki67.extend(preds_ki67)
                all_labels_ki67.extend(ki67_label.cpu().numpy())
                all_probs_ki67.extend(probs_ki67[:, 1])
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item()
            })
    
    # 计算指标
    metrics = {'loss': total_loss / len(val_loader)}
    
    if who_label is not None and len(all_labels_who) > 0:
        metrics['who_acc'] = accuracy_score(all_labels_who, all_preds_who)
        metrics['who_precision'] = precision_score(all_labels_who, all_preds_who, average='binary')
        metrics['who_recall'] = recall_score(all_labels_who, all_preds_who, average='binary')
        metrics['who_f1'] = f1_score(all_labels_who, all_preds_who, average='binary')
        metrics['who_auc'] = roc_auc_score(all_labels_who, all_probs_who)
    
    if ki67_label is not None and len(all_labels_ki67) > 0:
        metrics['ki67_acc'] = accuracy_score(all_labels_ki67, all_preds_ki67)
        metrics['ki67_precision'] = precision_score(all_labels_ki67, all_preds_ki67, average='binary')
        metrics['ki67_recall'] = recall_score(all_labels_ki67, all_preds_ki67, average='binary')
        metrics['ki67_f1'] = f1_score(all_labels_ki67, all_preds_ki67, average='binary')
        metrics['ki67_auc'] = roc_auc_score(all_labels_ki67, all_probs_ki67)
    
    return metrics


def test(model, test_loader, criterion, device, config):
    """测试模型"""
    # 测试过程与验证过程相同
    return validate(model, test_loader, criterion, device, config)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, is_best=False):
    """保存检查点"""
    os.makedirs(config.save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
        'config': config.__dict__
    }
    
    # 保存最新的检查点
    torch.save(checkpoint, os.path.join(config.save_dir, 'latest.pth'))
    
    # 每隔一定轮次保存一次
    if (epoch + 1) % config.save_freq == 0:
        torch.save(checkpoint, os.path.join(config.save_dir, f'epoch_{epoch+1}.pth'))
    
    # 保存最佳模型
    if is_best:
        torch.save(checkpoint, os.path.join(config.save_dir, 'best.pth'))


def load_checkpoint(model, optimizer, scheduler, config):
    """加载检查点"""
    if not os.path.exists(config.resume):
        print(f"Checkpoint {config.resume} does not exist, starting from scratch")
        return 0, {}
    
    print(f"Loading checkpoint from {config.resume}")
    checkpoint = torch.load(config.resume, map_location=config.device)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint.get('metrics', {})
    
    return start_epoch, metrics


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train a multi-modal MRI classifier')
    parser.add_argument('--config', type=str, default='', help='Path to config file')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, default='', help='Path to data directory')
    parser.add_argument('--label_file', type=str, default='labels.csv', help='Name of label file')
    parser.add_argument('--label_type', type=str, default='both', choices=['who', 'ki67', 'both'], help='Type of label to use')
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    # 更新配置
    if args.resume:
        config.resume = args.resume
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.label_file:
        config.label_file = args.label_file
    if args.label_type:
        config.label_type = args.label_type
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # 创建模型
    model = build_model(config)
    model = model.to(config.device)
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建优化器和学习率调度器
    optimizer, scheduler = build_optimizer(model, config)
    
    # 创建混合精度训练的缩放器
    scaler = GradScaler() if config.use_amp else None
    
    # 如果指定了恢复训练的检查点，加载模型
    start_epoch = 0
    best_metrics = {}
    if config.resume:
        start_epoch, best_metrics = load_checkpoint(model, optimizer, scheduler, config)
    
    # 记录最佳验证指标
    best_val_metric = best_metrics.get('val_metric', 0.0)
    
    # 训练循环
    for epoch in range(start_epoch, config.epochs):
        # 训练一个轮次
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.device, config, epoch, scaler)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 每隔一定轮次评估一次
        if (epoch + 1) % config.eval_freq == 0:
            # 验证
            val_metrics = validate(model, val_loader, criterion, config.device, config)
            
            # 打印指标
            print(f"Epoch {epoch+1}/{config.epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            if 'who_acc' in train_metrics:
                print(f"Train WHO Acc: {train_metrics['who_acc']:.4f}")
            if 'ki67_acc' in train_metrics:
                print(f"Train Ki67 Acc: {train_metrics['ki67_acc']:.4f}")
            
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            if 'who_acc' in val_metrics:
                print(f"Val WHO Acc: {val_metrics['who_acc']:.4f}")
                print(f"Val WHO F1: {val_metrics['who_f1']:.4f}")
                print(f"Val WHO AUC: {val_metrics['who_auc']:.4f}")
            if 'ki67_acc' in val_metrics:
                print(f"Val Ki67 Acc: {val_metrics['ki67_acc']:.4f}")
                print(f"Val Ki67 F1: {val_metrics['ki67_f1']:.4f}")
                print(f"Val Ki67 AUC: {val_metrics['ki67_auc']:.4f}")
            
            # 确定当前模型是否是最佳模型
            current_val_metric = 0.0
            if config.label_type == "both":
                if 'who_acc' in val_metrics and 'ki67_acc' in val_metrics:
                    current_val_metric = (val_metrics['who_acc'] + val_metrics['ki67_acc']) / 2
            elif config.label_type == "who" and 'who_acc' in val_metrics:
                current_val_metric = val_metrics['who_acc']
            elif config.label_type == "ki67" and 'ki67_acc' in val_metrics:
                current_val_metric = val_metrics['ki67_acc']
            
            is_best = current_val_metric > best_val_metric
            if is_best:
                best_val_metric = current_val_metric
                print(f"New best model with val metric: {best_val_metric:.4f}")
            
            # 保存检查点
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    'train': train_metrics,
                    'val': val_metrics,
                    'val_metric': current_val_metric
                },
                config=config,
                is_best=is_best
            )
    
    # 加载最佳模型进行测试
    best_checkpoint = torch.load(os.path.join(config.save_dir, 'best.pth'), map_location=config.device)
    model.load_state_dict(best_checkpoint['model'])
    
    # 测试
    test_metrics = test(model, test_loader, criterion, config.device, config)
    
    # 打印测试指标
    print("\nTest Results:")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    if 'who_acc' in test_metrics:
        print(f"Test WHO Acc: {test_metrics['who_acc']:.4f}")
        print(f"Test WHO Precision: {test_metrics['who_precision']:.4f}")
        print(f"Test WHO Recall: {test_metrics['who_recall']:.4f}")
        print(f"Test WHO F1: {test_metrics['who_f1']:.4f}")
        print(f"Test WHO AUC: {test_metrics['who_auc']:.4f}")
    if 'ki67_acc' in test_metrics:
        print(f"Test Ki67 Acc: {test_metrics['ki67_acc']:.4f}")
        print(f"Test Ki67 Precision: {test_metrics['ki67_precision']:.4f}")
        print(f"Test Ki67 Recall: {test_metrics['ki67_recall']:.4f}")
        print(f"Test Ki67 F1: {test_metrics['ki67_f1']:.4f}")
        print(f"Test Ki67 AUC: {test_metrics['ki67_auc']:.4f}")


if __name__ == "__main__":
    main()