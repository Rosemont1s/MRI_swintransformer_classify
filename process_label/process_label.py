import os
import pandas as pd
import re


class MeningiomaDataProcessor:
    """处理脑膜瘤相关数据，包括 Ki67 指标和 WHO 级别转换"""

    def __init__(self, df, id_col, ki67_col, who_col):
        self.df = df
        self.id_col = id_col
        self.ki67_col = ki67_col
        self.who_col = who_col

    @staticmethod
    def extract_min_number(s):
        """提取字符串中的最小数字"""
        s = str(s)
        numbers = re.findall(r'\d+', s)
        return min(map(int, numbers)) if numbers else None

    @staticmethod
    def standardize_roman(text):
        """标准化罗马数字（将1、Ⅰ、ⅰ等转换为 I）"""
        replace_map = {'l': 'I', '1': 'I', 'Ⅰ': 'I', 'ⅰ': 'I'}
        for old, new in replace_map.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def convert_who_grade(grade):
        """将 WHO 级别统一转换为阿拉伯数字"""
        roman_mapping = {'I': 1, 'II': 2, 'III': 3}
        grade = str(grade).strip().upper()
        grade = MeningiomaDataProcessor.standardize_roman(grade)
        return roman_mapping.get(grade, int(grade) if grade.isdigit() else None)

    def process(self):
        """处理 Ki67 和 WHO 分级"""
        self.df['Ki67_check'] = self.df[self.ki67_col].apply(self.extract_min_number)
        self.df['WHO_check'] = self.df[self.who_col].apply(self.convert_who_grade)

        # 过滤掉 Ki67 或 WHO 为空的样本
        clean_df = self.df.dropna(subset=['Ki67_check', 'WHO_check'])

        return {
            row[self.id_col]: {'Ki67': row['Ki67_check'], 'WHO': row['WHO_check']}
            for _, row in clean_df.iterrows()
        }


def load_index_mapping(file_path):
    """加载 index.txt 映射关系"""
    index_map = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(' -> ')
            if len(parts) == 2:
                index_map[parts[1]] = parts[0]
    for key in index_map:
        index_map[key] = index_map[key].split('_')[0]
        index_map[key] = index_map[key].split(' ')[0]
    return index_map

def calculate_ki67_stats(data):
    # 提取所有Ki67值
    ki67_values = [subject['Ki67'] for subject in data.values()]

    # 计算均数
    mean = sum(ki67_values) / len(ki67_values)

    # 计算中位数
    sorted_values = sorted(ki67_values)
    n = len(sorted_values)
    mid = n // 2

    if n % 2 == 1:
        median = sorted_values[mid]
    else:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2

    return mean, median

def get_processed_label_dictlist():
    # 读取 Excel 数据
    list_gx = pd.read_excel("./gx.xlsx", sheet_name='Sheet1')
    list_mr8 = pd.read_excel("./mr8.xlsx", sheet_name='glioma')
    list_mr12 = pd.read_excel("./mr12.xlsx", sheet_name='glioma')
    # 仅保留脑膜瘤和脑膜瘤术后复发病例
    list_mr8 = list_mr8[list_mr8['病理分类'].isin(['脑膜瘤', '脑膜瘤术后复发', '脑膜瘤术后残留', '脑膜瘤术后'])]
    list_mr12 = list_mr12[list_mr12['病理分类'].isin(['脑膜瘤', '脑膜瘤术后复发', '脑膜瘤术后残留', '脑膜瘤术后'])]


    # 处理各数据集
    processor_gx = MeningiomaDataProcessor(list_gx, '住院号', 'Ki67', 'WHO分级')
    dict_gx = processor_gx.process()

    processor_mr8 = MeningiomaDataProcessor(list_mr8, '编号', 'Ki67', 'WHO分级')
    dict_mr8 = processor_mr8.process()

    processor_mr12 = MeningiomaDataProcessor(list_mr12, '编号', 'Ki67（%）', 'WHO分级')
    dict_mr12 = processor_mr12.process()
    dict_gx = {str(key): value for key, value in dict_gx.items()}

    # 读取 index 映射
    index_map = load_index_mapping("./index.txt")

    # print("mr12: 均数:{:.1f} 中位数:{:.1f}, mr8: 均数:{:.1f} 中位数:{:.1f}, gx: 均数:{:.1f} 中位数:{:.1f}".format(
    #     *calculate_ki67_stats(dict_mr12),
    #     *calculate_ki67_stats(dict_mr8),
    #     *calculate_ki67_stats(dict_gx)
    # ))
    #
    # all_ki67 = [v['Ki67'] for data in [dict_mr12, dict_mr8, dict_gx] for v in data.values()]
    # total_mean = sum(all_ki67) / len(all_ki67)
    # print(f"Ki67总均值：{total_mean:.1f}")
    #mr12: 均数:3.4 中位数:3.0, mr8: 均数:4.3 中位数:3.0, gx: 均数:4.6 中位数:2.0
    #Ki67总均值：4.2

    def binarilized_dict(data):
        threshold = 5
        # 将 Ki67 进行二分类
        for key in data:
            data[key]['Ki67_binary'] = 1 if data[key]['Ki67'] >= threshold else 0
        threshold = 2
        for key in data:
            data[key]['WHO_binary'] = 1 if data[key]['WHO'] >= threshold else 0

    binarilized_dict(dict_mr12)
    binarilized_dict(dict_mr8)
    binarilized_dict(dict_gx)
    merged_dict = dict_mr12 | dict_mr8 | dict_gx
    print(merged_dict)
    print(index_map)
    new_dict = {k: merged_dict[v] for k, v in index_map.items() if v in merged_dict}
    print(new_dict)
    print(len(new_dict))
    #统计who
    who_dict={'who1':0, 'who2':0, 'who3':0}
    for key, value in new_dict.items():
        who_dict['who{0}'.format(int(value['WHO']))] += 1
    print(who_dict)
    return new_dict

def export_to_csv(data_dict, output_file):
    """将处理好的标签数据导出为CSV文件
    
    Args:
        data_dict: 处理好的标签数据字典，格式为 {patient_id: {'WHO': who_grade, 'Ki67': ki67_value, ...}}
        output_file: 输出CSV文件路径
    """
    # 准备CSV数据
    csv_data = []
    for patient_id, label_data in data_dict.items():
        csv_data.append({
            'patient_id': patient_id,
            'who': label_data['WHO_binary'],
            'ki67': label_data['Ki67_binary']
        })
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"已将标签数据保存到 {output_file}，共 {len(df)} 条记录")
    return df

if __name__ == '__main__':
    # 获取处理好的标签数据
    processed_labels = get_processed_label_dictlist()
    
    # 导出为CSV文件
    output_dir = "d:/work/zhongzhong/MRI_swintransformer_classify/data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "labels.csv")
    
    export_to_csv(processed_labels, output_file)
