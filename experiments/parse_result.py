import os
import re
import csv

def extract_accuracies_from_file(file_path, dataset_names):
    """
    从输出文件中提取准确率信息
    
    Args:
        file_path: 输出文件的完整路径
        dataset_names: 数据集名称列表
    
    Returns:
        list: 包含准确率数值的列表
    """
    accuracies = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 使用正则表达式匹配所有符合条件的结果行
            pattern = r'episode 1000/1000, Acc \[([0-9.]+)\]'
            matches = re.findall(pattern, content)
            
            # 只取前23个匹配结果（对应23个数据集）
            matches = matches[:23]
            
            if len(matches) != len(dataset_names):
                print(f"警告: 在文件 {file_path} 中找到 {len(matches)} 个结果，但期望 {len(dataset_names)} 个数据集")
            
            # 处理每个匹配结果
            for i, match in enumerate(matches):
                if i < len(dataset_names):
                    acc_value = float(match)
                    # 转换为百分比，保留两位小数，四舍五入
                    acc_percent = round(acc_value * 100, 2)
                    accuracies.append(acc_percent)
                else:
                    break
                    
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
    
    return accuracies

def save_results_to_csv(all_results, output_file_path, dataset_names, filenames):
    """
    将所有结果保存为CSV文件（横向排版）
    
    Args:
        all_results: 所有文件的结果列表，每个元素是一个准确率列表
        output_file_path: 输出文件路径
        dataset_names: 数据集名称列表
        filenames: 文件名列表
    """
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头：第一列是文件名，后面是数据集名称
            header = ['Filename'] + dataset_names
            writer.writerow(header)
            
            # 写入数据行
            for i, (filename, accuracies) in enumerate(all_results):
                if len(accuracies) == len(dataset_names):
                    row = [filename] + accuracies
                    writer.writerow(row)
                else:
                    print(f"警告: 文件 {filename} 的结果数量不匹配，跳过")
        
        print(f"CSV结果已保存到: {output_file_path}")
        
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")

def process_eval_files(root_dir, filenames, dataset_names):
    """
    处理所有评估文件并生成横向排版的CSV
    
    Args:
        root_dir: 根目录路径
        filenames: 文件名列表
        dataset_names: 数据集名称列表
    """
    all_results = []  # 存储所有文件的结果
    
    for filename in filenames:
        file_path = os.path.join(root_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        print(f"正在处理文件: {filename}")
        
        # 提取准确率信息
        accuracies = extract_accuracies_from_file(file_path, dataset_names)
        
        if accuracies:
            all_results.append((filename, accuracies))
            
            # 在控制台显示单个文件的结果
            print(f"{filename} 的处理结果:")
            for j, dataset_name in enumerate(dataset_names):
                if j < len(accuracies):
                    print(f"  {dataset_name}: {accuracies[j]:.2f}%")
            print()
        else:
            print(f"在文件 {filename} 中未找到有效结果\n")
    
    # 如果有结果，保存到CSV文件
    if all_results:
        output_file_path = os.path.join(root_dir, "all_results.csv")
        save_results_to_csv(all_results, output_file_path, dataset_names, filenames)
        
        # 显示CSV内容预览
        print("\nCSV文件内容预览:")
        print("Filename", end="")
        for name in dataset_names:
            print(f",{name}", end="")
        print()
        
        for filename, accuracies in all_results:
            print(filename, end="")
            for acc in accuracies:
                print(f",{acc:.2f}", end="")
            print()
    else:
        print("没有找到任何有效结果")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    roots = [
        "/project/vonneumann1/cl2025/GKD/experiments/gkd_ablation/study_kg/eval_kg_blk2_retest",
    ]
    
    filenames = [ "eval_1shot.out", "eval_0shot.out", "eval_1shot+text.out"]
    # filenames = [ "eval_1shot+text.out"]
    
    # 数据集名称列表作为参数传入
    dataset_names = [
        "Animal pose", "AwA", "CUB", "NAB", "ap10k test", 
        "vinegar fly", "locust", "topviewmouse5k", "macaque", 
        "atrw tiger", "ak test", "coco val", "human art", 
        "human face 300w", "animalweb", "onehand10k", "HInt", 
        "keypoint-5", "carfusion", "df2 val", "cephalometric", 
        "hand xray (base kp)", "hand xray (novel kp)"
    ]
    
    # 处理文件，传入数据集名称列表
    for i, each_root in enumerate(roots):
        print(f"==>root {i}: {each_root}")
        process_eval_files(each_root, filenames, dataset_names)