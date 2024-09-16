import os
import pandas as pd
import shutil
import random

def split_csv_by_ones(input_csv_path, output_folder):

    output_folder = os.path.join(output_folder, 'splits_csv')
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(input_csv_path, header=None)

    last_column = df.iloc[:, -1]

    start = None
    splits = []

    # 遍历每一行，寻找连续的1区间
    for i, value in enumerate(last_column):
        if value == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                splits.append((start, end))
                start = None

    # 处理最后一个区间
    if start is not None:
        end = len(last_column) - 1
        splits.append((start, end))

    # 提取每个区间，并保存为新文件
    frame_rate = 30 
    for idx, (start, end) in enumerate(splits):
        start_adjusted = max(0, start - 5 * frame_rate)
        end_adjusted = min(len(df) - 1, end + 3 * frame_rate)

        split_df = df.iloc[start_adjusted:end_adjusted+1]
        
        output_path = os.path.join(output_folder, f"falldown_test_split_{idx+1}.csv")
        split_df.to_csv(output_path, index=False, header=False)
        print(f"Saved split {idx+1} to {output_path}")


def split_csv_files(source_folder, target_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of ratios must be 1.0"

    train_folder = os.path.join(target_folder, 'train')
    val_folder = os.path.join(target_folder, 'val')
    test_folder = os.path.join(target_folder, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    clear_folder(train_folder)
    clear_folder(val_folder)
    clear_folder(test_folder)

    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

    # 随机排序文件列表
    random.shuffle(csv_files)

    # 计算各部分的数量
    total_files = len(csv_files)
    train_count = int(total_files * train_ratio)
    test_count = int(total_files * test_ratio)
    val_count = total_files - train_count - test_count

    # 分割文件
    train_files = csv_files[:train_count]
    val_files = csv_files[train_count:train_count + val_count]
    test_files = csv_files[train_count + val_count:]

    for file in train_files:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(train_folder, file)
        shutil.copy(src_path, dst_path)

    for file in val_files:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(val_folder, file)
        shutil.copy(src_path, dst_path)

    for file in test_files:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(test_folder, file)
        shutil.copy(src_path, dst_path)

    print(f"Split completed: {len(train_files)} files in train, {len(val_files)} files in val, {len(test_files)} files in test.")


def remove_first_column_from_csvs(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        
        try:
            df = pd.read_csv(file_path)
            
            if df.shape[1] > 1: 
                df.drop(df.columns[0], axis=1, inplace=True)
                
                df.to_csv(file_path, index=False)
                print(f"Processed and saved: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def merge_folders(src_folder1, src_folder2, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

    clear_folder(dest_folder)
    for filename in os.listdir(src_folder1):
        src_file = os.path.join(src_folder1, filename)
        dest_file = os.path.join(dest_folder, filename)
        if not os.path.exists(dest_file) or os.path.getsize(src_file) != os.path.getsize(dest_file):
            shutil.copy2(src_file, dest_folder)

    for filename in os.listdir(src_folder2):
        src_file = os.path.join(src_folder2, filename)
        dest_file = os.path.join(dest_folder, filename)
        if not os.path.exists(dest_file) or os.path.getsize(src_file) != os.path.getsize(dest_file):
            shutil.copy2(src_file, dest_folder)

if __name__ == "__main__":
    source_folder = '../../ts_stage/ts_data/splits_csv'  
    target_folder = '../../ts_stage/ts_data' 
    # remove_first_column_from_csvs(source_folder)
    split_csv_files(source_folder, target_folder, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10)


    # input_csv_path = '../data/output_csv_label/test_falldown_one_label.csv'
    # output_folder = '../../ts_stage/ts_data/'
    # os.makedirs(output_folder, exist_ok=True)

    # split_csv_by_ones(input_csv_path, output_folder)