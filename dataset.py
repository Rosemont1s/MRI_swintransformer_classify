import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
import random
from scipy import ndimage
import pandas as pd

class MultiModalMRIDataset(Dataset):
    """
    多模态MRI数据集类，用于加载和处理脑膜瘤多模态MRI数据
    支持4种MRI序列（T1、T2、T1c、FLAIR）和1个分割掩码
    """
    def __init__(
            self, 
            data_dir, 
            modalities=["t1w", "t2w", "t1c", "t2flair", "seg"],
            transform=None,
            target_size=(240, 240, 155),
            augment=True,
            label_type="both",  # "who", "ki67", "both"
            label_file="labels.csv"
        ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录，包含患者子目录
            modalities: 模态列表，默认为["t1", "t2", "t1ce", "flair", "mask"]
            transform: 数据变换
            target_size: 目标图像大小
            augment: 是否进行数据增强
            label_type: 标签类型，"who"表示WHO分级，"ki67"表示Ki67指数，"both"表示两者都要
            label_file: 标签文件名
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.transform = transform
        self.target_size = target_size
        self.augment = augment
        self.label_type = label_type
        
        # 获取所有患者ID
        self.patient_ids = self._get_patient_ids()

        # 加载标签
        self.labels = self._load_labels(label_file)
        
        # 过滤掉没有标签的患者
        self._filter_patients_with_labels()
        
        print(f"加载了 {len(self.patient_ids)} 个有效患者数据")
    
    def _get_patient_ids(self):
        """获取所有患者ID"""
        patient_ids = []
        for patient_dir in self.data_dir.iterdir():
            if patient_dir.is_dir():
                # 检查是否包含所有需要的模态
                has_all_modalities = True
                for modality in self.modalities:
                    modality_files = list(patient_dir.glob(f"*{modality}*.nii.gz"))
                    if not modality_files:
                        has_all_modalities = False
                        break
                
                if has_all_modalities:
                    patient_ids.append(patient_dir.name)
        
        return sorted(patient_ids)
    
    def _load_labels(self, label_file):
        """加载标签"""
        # 尝试从labels.csv加载标签
        label_file_path = Path(os.path.join(self.data_dir, label_file))
        print(f"尝试加载标签文件: {label_file_path}")
        
        if label_file_path.exists():
            # 使用dtype参数指定patient_id列为字符串类型，保留前导零
            df = pd.read_csv(label_file_path, dtype={"patient_id": str})
            labels = {}
            
            for _, row in df.iterrows():
                patient_id = row["patient_id"]
                
                # 处理WHO分级 - 转换为二分类 (1-2为低级别，3为高级别)
                who_grade = int(row["who_grade"]) if "who_grade" in df.columns else None
                who_class = 1 if who_grade == 2 else 0 if who_grade in [1] else None
                
                # 处理Ki67指数 - 直接使用ki67_binary列
                ki67_class = int(row["ki67_binary"]) if "ki67_binary" in df.columns else None
                
                # 根据标签类型加载不同的标签
                if self.label_type == "who" and who_class is not None:
                    labels[patient_id] = who_class
                elif self.label_type == "ki67" and ki67_class is not None:
                    labels[patient_id] = ki67_class
                elif self.label_type == "both" and who_class is not None and ki67_class is not None:
                    labels[patient_id] = {
                        "who_grade": who_class,
                        "ki67_index": ki67_class
                    }
            
            print(f"从{label_file_path}加载了{len(labels)}个标签")
            print(f"示例patient_id: {list(labels.keys())[:5]}")
            return labels
        else:
            print(f"找不到标签文件: {label_file_path}")
        print("无法找到标签文件，创建随机标签用于测试")
        return {}
    
    def _load_index_mapping(self):
        """加载index.txt映射关系"""
        index_map = {}
        index_file = Path("d:/work/zhongzhong/MRI_swintransformer_classify/process_label/index.txt")
        
        if index_file.exists():
            with open(index_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(' -> ')
                    if len(parts) == 2:
                        original_id = parts[0]
                        new_id = parts[1]
                        index_map[new_id] = original_id
        
        return index_map
    
    def _filter_patients_with_labels(self):
        """过滤掉没有标签的患者"""
        valid_patient_ids = []
        for patient_id in self.patient_ids:
            if patient_id in self.labels:
                valid_patient_ids.append(patient_id)
        
        self.patient_ids = valid_patient_ids
    
    def _load_nifti(self, file_path):
        """加载NIfTI文件"""
        img = nib.load(file_path)
        data = img.get_fdata()
        return data
    
    def _preprocess(self, data):
        """预处理数据"""
        # 调整大小到目标尺寸
        if data.shape != self.target_size:
            zoom_factors = [t / s for t, s in zip(self.target_size, data.shape)]
            data = ndimage.zoom(data, zoom_factors, order=1)
        
        # 归一化
        if np.max(data) > 0:
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        return data
    
    def _augment(self, data_list):
        """数据增强"""
        if not self.augment:
            return data_list
        
        # 随机翻转
        if random.random() > 0.5:
            data_list = [np.flip(data, axis=0) for data in data_list]
        
        # 随机旋转
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            data_list = [ndimage.rotate(data, angle, axes=(0, 1), reshape=False) for data in data_list]
        
        # 随机缩放
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            data_list = [ndimage.zoom(data, (scale, scale, 1), order=1) for data in data_list]
        
        # 随机亮度和对比度调整（仅对MRI序列，不对掩码）
        if random.random() > 0.5:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            for i in range(len(data_list) - 1):  # 不包括最后一个（掩码）
                data = data_list[i]
                data = data * contrast + brightness
                data = np.clip(data, 0, 1)
                data_list[i] = data
        
        return data_list
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        """获取数据项"""
        patient_id = self.patient_ids[idx]
        patient_dir = self.data_dir / patient_id
        
        # 加载所有模态
        data_list = []
        for modality in self.modalities:
            modality_files = list(patient_dir.glob(f"*{modality}*.nii.gz"))
            if modality_files:
                data = self._load_nifti(modality_files[0])
                data = self._preprocess(data)
                data_list.append(data)
            else:
                raise ValueError(f"Modality {modality} not found for patient {patient_id}")
        
        # 数据增强
        if self.augment:
            data_list = self._augment(data_list)
        
        # 转换为张量
        data_tensors = []
        for data in data_list:
            # 添加通道维度
            data = np.expand_dims(data, axis=0)
            tensor = torch.from_numpy(data).float()
            data_tensors.append(tensor)
        
        # 获取标签
        if self.label_type == "both":
            # 返回两个标签
            who_label = self.labels[patient_id]["who_grade"]
            ki67_label = self.labels[patient_id]["ki67_index"]
            
            # 在训练脚本中，我们假设两个标签是相同的
            # 实际应用中，您可能需要返回两个不同的标签
            label = torch.tensor(who_label, dtype=torch.long)
            
            # 如果需要分别返回两个标签，可以取消下面的注释
            return data_tensors, (torch.tensor(who_label, dtype=torch.long), 
                                  torch.tensor(ki67_label, dtype=torch.long))
        else:
            label = self.labels[patient_id]
            label = torch.tensor(label, dtype=torch.long)
        
        return data_tensors, label

if __name__ == "__main__":
    # 测试数据集
    data_dir = 'd:/work/zhongzhong/MRI_swintransformer_classify/data'
    label_file = 'labels.csv'
    
    # 检查数据目录和标签文件是否存在
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")
    
    # 检查几个可能的标签文件位置

    
    # 测试Ki67标签
    dataset = MultiModalMRIDataset(
        data_dir=data_dir,
        label_type="ki67",
        label_file=label_file
    )
    
    # 打印数据集信息
    print(f"数据集大小: {len(dataset)}")
    
    # 获取第一个样本
    if len(dataset) > 0:
        data_tensors, label = dataset[0]
        print(f"样本形状: {[tensor.shape for tensor in data_tensors]}")
        print(f"标签: {label}")
        
        # 统计标签分布
        label_counts = {}
        for i in range(len(dataset)):
            _, label = dataset[i]
            label_val = label.item()
            label_counts[label_val] = label_counts.get(label_val, 0) + 1
        
        print(f"标签分布: {label_counts}")
    else:
        print("数据集为空，请检查数据目录和标签文件")
    
    # 测试both模式
    dataset_both = MultiModalMRIDataset(
        data_dir='d:/work/zhongzhong/MRI_swintransformer_classify/data',
        label_type="both"
    )
    
    if len(dataset_both) > 0:
        data_tensors, (who_label, ki67_label) = dataset_both[0]
        print(f"\n测试both模式:")
        print(f"WHO标签: {who_label}, Ki67标签: {ki67_label}")