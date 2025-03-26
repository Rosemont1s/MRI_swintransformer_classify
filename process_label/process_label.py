# import os
# import pandas as pd
# import re
#
#
# def extract_min_number(s):
#     # 提取所有连续数字部分
#     s=str(s)
#     numbers = re.findall(r'\d+', s)
#     if not numbers:
#         return None  # 或 raise ValueError("字符串中未找到数字")
#     # 转换为整数并取最小值
#     return min(map(int, numbers))
#
#
# def standardize_roman(text):
#     # 定义需替换的字符映射
#     replace_map = {
#         'l': 'I',    # 小写字母L → I
#         '1': 'I',    # 数字1 → I
#         'Ⅰ': 'I',    # Unicode罗马数字一 → I
#         'ⅰ': 'I'     # Unicode小写罗马数字一 → I
#     }
#     for old_char, new_char in replace_map.items():
#         text = text.replace(old_char, new_char)
#     return text
# def convert_who_grade(grade):
#     """
#     将 WHO 分级中的罗马数字或阿拉伯数字统一转换为阿拉伯数字。
#     支持输入类型：整数、浮点数、字符串（如 "III", "2"）。
#     """
#     roman_mapping = {'I': 1, 'II': 2, 'III': 3}
#     grade=str(grade).strip().upper()
#     grade = standardize_roman(grade)
#     if grade in roman_mapping:
#         return roman_mapping[grade]
#     else:
#         try:
#             return int(grade)
#         except ValueError:
#             return None
#
# def clean_dict(data_dict):
#     clean_data = {
#         k: v for k, v in data_dict.items()
#         if not (pd.isna(v['Ki67']) or pd.isna(v['WHO']))
#     }
#     return clean_data
# list_gx = pd.DataFrame(pd.read_excel("./gx.xlsx",sheet_name='Sheet1'))
# list_mr8 = pd.DataFrame(pd.read_excel("./mr8.xlsx",sheet_name='glioma'))
# list_mr8 = list_mr8[list_mr8['病理分类'] == ('脑膜瘤' or '脑膜瘤术后复发')]
# list_mr12 = pd.DataFrame(pd.read_excel("./mr12.xlsx",sheet_name='glioma'))
# list_mr12 = list_mr12[list_mr12['病理分类'] == ('脑膜瘤' or '脑膜瘤术后复发')]
#
#
# list_gx['Ki67_check'] = [extract_min_number(i) for i in list_gx['Ki67']]
# list_gx['WHO_check'] = [convert_who_grade(i) for i in list_gx['WHO分级']]
# dict_gx = {
#     num1: {'Ki67': ki67, 'WHO': who}
#     for num1, ki67, who in zip(
#         list_gx['住院号'],
#         list_gx['Ki67_check'],
#         list_gx['WHO_check']
#     )
# }
# dict_gx=clean_dict(dict_gx)
# list_mr8['Ki67_check'] = [extract_min_number(i) for i in list_mr8['Ki67']]
# list_mr8['WHO_check'] = [convert_who_grade(i) for i in list_mr8['WHO分级']]
# dict_mr8 = {
#     num1: {'Ki67': ki67, 'WHO': who}
#     for num1, ki67, who in zip(
#         list_mr8['编号'],
#         list_mr8['Ki67_check'],
#         list_mr8['WHO_check']
#     )
# }
# dict_mr8=clean_dict(dict_mr8)
# list_mr12['Ki67_check'] = [extract_min_number(i) for i in list_mr12['Ki67（%）']]
# list_mr12['WHO_check'] = [convert_who_grade(i) for i in list_mr12['WHO分级']]
# dict_mr12 = {
#     num1: {'Ki67': ki67, 'WHO': who}
#     for num1, ki67, who in zip(
#         list_mr12['编号'],
#         list_mr12['Ki67_check'],
#         list_mr12['WHO_check']
#     )
# }
# dict_mr12=clean_dict(dict_mr12)
#
# print(dict_mr12)
# print(dict_mr8)
# print(dict_gx)
#
#
#
# index={}
# with open("./index.txt", encoding="utf-8") as f:
#     for i in f.readlines():
#         origin=i.split(' -> ')[0]
#         new=i.split(' -> ')[1].strip()
#         index[new]=origin
#     f.close()

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
    return index_map


# 读取 Excel 数据
list_gx = pd.read_excel("./gx.xlsx", sheet_name='Sheet1')
list_mr8 = pd.read_excel("./mr8.xlsx", sheet_name='glioma')
list_mr12 = pd.read_excel("./mr12.xlsx", sheet_name='glioma')

# 仅保留脑膜瘤和脑膜瘤术后复发病例
list_mr8 = list_mr8[list_mr8['病理分类'].isin(['脑膜瘤', '脑膜瘤术后复发'])]
list_mr12 = list_mr12[list_mr12['病理分类'].isin(['脑膜瘤', '脑膜瘤术后复发'])]

# 处理各数据集
processor_gx = MeningiomaDataProcessor(list_gx, '住院号', 'Ki67', 'WHO分级')
dict_gx = processor_gx.process()

processor_mr8 = MeningiomaDataProcessor(list_mr8, '编号', 'Ki67', 'WHO分级')
dict_mr8 = processor_mr8.process()

processor_mr12 = MeningiomaDataProcessor(list_mr12, '编号', 'Ki67（%）', 'WHO分级')
dict_mr12 = processor_mr12.process()

print(dict_mr12)
print(dict_mr8)
print(dict_gx)

# 读取 index 映射
index_map = load_index_mapping("./index.txt")
print(index_map)

