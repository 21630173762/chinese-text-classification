import os
import jieba
import re
from tqdm import tqdm
from utils.data_processor import clean_text, preprocess_text

def preprocess_file(input_file, output_file):
    """预处理单个文件"""
    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in):
            try:
                label, text = line.strip().split('\t')
                # 清洗文本
                text = clean_text(text)
                # 分词
                text = preprocess_text(text)
                # 写入处理后的数据
                f_out.write(f"{label}\t{text}\n")
            except:
                print(f"Error processing line: {line}")
                continue

def main():
    # 创建输出目录
    os.makedirs('D:/c/AICode/work/data/processed', exist_ok=True)
    
    # 预处理训练集
    preprocess_file(
        'D:/c/AICode/work/data/cnews.train.txt',
        'D:/c/AICode/work/data/processed/cnews.train.processed.txt'
    )
    
    # 预处理验证集
    preprocess_file(
        'D:/c/AICode/work/data/cnews.val.txt',
        'D:/c/AICode/work/data/processed/cnews.val.processed.txt'
    )
    
    # 预处理测试集
    preprocess_file(
        'D:/c/AICode/work/data/cnews.test.txt',
        'D:/c/AICode/work/data/processed/cnews.test.processed.txt'
    )
    
    print("Preprocessing completed!")

if __name__ == '__main__':
    main() 