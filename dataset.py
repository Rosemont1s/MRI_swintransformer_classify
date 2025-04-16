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
            modalities=["T1W", "T2W", "T1C", "T2F", "seg"],
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
                    modality_files = list(patient_dir.glob(f"*{modality}*"))
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
                who_class = row['who']
                # 处理Ki67指数 - 直接使用ki67_binary列
                ki67_class = row['ki67']
                
                # 根据标签类型加载不同的标签
                if self.label_type == "who" and who_class is not None:
                    labels[patient_id] = {
                        "who": who_class
                    }
                elif self.label_type == "ki67" and ki67_class is not None:
                    labels[patient_id] = {
                        "ki67": ki67_class
                    }
                elif self.label_type == "both" and who_class is not None and ki67_class is not None:
                    labels[patient_id] = {
                        "who": who_class,
                        "ki67": ki67_class
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
            # 计算缩放因子
            zoom_factors = [t / s for t, s in zip(self.target_size, data.shape)]
            
            # 使用精确的缩放因子进行缩放
            data = ndimage.zoom(data, zoom_factors, order=1, mode='nearest')
            
            # 确保尺寸完全匹配目标尺寸
            if data.shape != self.target_size:
                # 如果缩放后尺寸仍不匹配，进行裁剪或填充
                temp = np.zeros(self.target_size, dtype=data.dtype)
                
                # 计算每个维度的复制范围
                copy_shape = [min(s, t) for s, t in zip(data.shape, self.target_size)]
                
                # 复制数据
                slices_src = tuple(slice(0, s) for s in copy_shape)
                slices_dst = tuple(slice(0, s) for s in copy_shape)
                temp[slices_dst] = data[slices_src]
                
                data = temp
                
            # 确认尺寸
            assert data.shape == self.target_size, f"预处理后的形状 {data.shape} 与目标形状 {self.target_size} 不匹配"
        
        # 归一化
        if np.max(data) > 0:
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        return data
    
    def _augment(self, data_list):
        """数据增强"""
        if not self.augment:
            return data_list
        
        # 不进行随机翻转，而是使用更适合医学图像的增强方法
        
        # 随机旋转（小角度，保持解剖结构基本不变）
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)  # 减小旋转角度范围
            data_list = [ndimage.rotate(data, angle, axes=(0, 1), reshape=False, order=1) for data in data_list]
        
        # 随机缩放（轻微缩放）
        if random.random() > 0.5:
            scale = random.uniform(0.95, 1.05)  # 减小缩放范围
            data_list = [ndimage.zoom(data, (scale, scale, 1), order=1) for data in data_list]
        
        # 随机亮度和对比度调整（仅对MRI序列，不对掩码）
        if random.random() > 0.5:
            brightness = random.uniform(0.95, 1.05)
            contrast = random.uniform(0.95, 1.05)
            for i in range(len(data_list) - 1):  # 不包括最后一个（掩码）
                data = data_list[i]
                data = data * contrast + brightness
                data = np.clip(data, 0, 1)
                data_list[i] = data
        
        # 随机添加高斯噪声（更适合医学图像）
        if random.random() > 0.5:
            for i in range(len(data_list) - 1):  # 不包括掩码
                noise = np.random.normal(0, 0.01, data_list[i].shape)  # 低强度噪声
                data_list[i] = np.clip(data_list[i] + noise, 0, 1)
        
        # 确保所有数据都是目标尺寸
        for i in range(len(data_list)):
            if data_list[i].shape != self.target_size:
                # 如果增强后尺寸变化，重新调整到目标尺寸
                temp = np.zeros(self.target_size, dtype=data_list[i].dtype)
                
                # 计算每个维度的复制范围
                copy_shape = [min(s, t) for s, t in zip(data_list[i].shape, self.target_size)]
                
                # 复制数据
                slices_src = tuple(slice(0, s) for s in copy_shape)
                slices_dst = tuple(slice(0, s) for s in copy_shape)
                temp[slices_dst] = data_list[i][slices_src]
                
                # 如果需要放大，使用插值
                if any(s < t for s, t in zip(data_list[i].shape, self.target_size)):
                    zoom_factors = [t / s for t, s in zip(self.target_size, data_list[i].shape)]
                    temp = ndimage.zoom(data_list[i], zoom_factors, order=1, mode='nearest')
                    
                    # 确保尺寸完全匹配
                    if temp.shape != self.target_size:
                        temp2 = np.zeros(self.target_size, dtype=temp.dtype)
                        copy_shape = [min(s, t) for s, t in zip(temp.shape, self.target_size)]
                        slices_src = tuple(slice(0, s) for s in copy_shape)
                        slices_dst = tuple(slice(0, s) for s in copy_shape)
                        temp2[slices_dst] = temp[slices_src]
                        temp = temp2
                
                data_list[i] = temp
                
                # 确认尺寸
                assert data_list[i].shape == self.target_size, f"增强后的形状 {data_list[i].shape} 与目标形状 {self.target_size} 不匹配"
        
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
            modality_files = list(patient_dir.glob(f"*{modality}*"))
            if modality_files:
                data = self._load_nifti(modality_files[0])
                data = self._preprocess(data)
                data_list.append(data)
            else:
                raise ValueError(f"Modality {modality} not found for patient {patient_id}")
        
        # 数据增强
        if self.augment:
            data_list = self._augment(data_list)
        
        # 最终检查所有数据的尺寸
        for i, data in enumerate(data_list):
            if data.shape != self.target_size:
                print(f"警告: 模态 {i} 的形状 {data.shape} 与目标形状 {self.target_size} 不匹配，正在调整...")
                # 强制调整到目标尺寸
                temp = np.zeros(self.target_size, dtype=data.dtype)
                copy_shape = [min(s, t) for s, t in zip(data.shape, self.target_size)]
                slices_src = tuple(slice(0, s) for s in copy_shape)
                slices_dst = tuple(slice(0, s) for s in copy_shape)
                temp[slices_dst] = data[slices_src]
                data_list[i] = temp
        
        # 转换为张量
        data_tensors = []
        for data in data_list:
            # 添加通道维度
            data = np.expand_dims(data, axis=0)
            tensor = torch.from_numpy(data).float()
            data_tensors.append(tensor)
            # 最终检查张量形状
            expected_shape = (1,) + self.target_size
            assert tensor.shape == expected_shape, f"张量形状 {tensor.shape} 与预期形状 {expected_shape} 不匹配"
        
        # 获取标签
        if self.label_type == "both":
            # 返回两个标签
            who_label = self.labels[patient_id]["who"]
            ki67_label = self.labels[patient_id]["ki67"]
            return data_tensors, (torch.tensor(who_label, dtype=torch.long), 
                                  torch.tensor(ki67_label, dtype=torch.long))
        elif self.label_type == "ki67":
            label = self.labels[patient_id]["ki67"]
            label = torch.tensor(label, dtype=torch.long)
            return data_tensors, label
        elif self.label_type == "who":
            label = self.labels[patient_id]["who"]
            label = torch.tensor(label, dtype=torch.long)
            return data_tensors, label

if __name__ == "__main__":
    # 测试数据集
    data_dir = '/home/yankai/MRI_swintransformer_classify-main/data/data_with_seg'
    label_file = 'labels.csv'
    
    # 检查数据目录和标签文件是否存在
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")

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
            try:
                _, label = dataset[i]
                label_val = label.item()
                label_counts[label_val] = label_counts.get(label_val, 0) + 1
            except Exception as e:
                print("error when getting label using index{}".format(i))
        
        print(f"标签分布: {label_counts}")
    else:
        print("数据集为空，请检查数据目录和标签文件")
    
    # 测试both模式
    dataset_both = MultiModalMRIDataset(
        data_dir='/home/yankai/MRI_swintransformer_classify-main/data/data_with_seg',
        label_type="both"
    )
    
    if len(dataset_both) > 0:
        data_tensors, (who_label, ki67_label) = dataset_both[0]
        print(f"\n测试both模式:")
        print(f"WHO标签: {who_label}, Ki67标签: {ki67_label}")