import pandas as pd
import os

def count_zeros_and_ones_in_last_column(csv_path):
    try:
        df = pd.read_csv(csv_path, header=None)
        last_column = df.iloc[:, -1]
        zeros_count = (last_column == 0).sum()
        ones_count = (last_column == 1).sum()
        return zeros_count, ones_count
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None, None

def count_zeros_and_ones_in_folder(folder_path):
 
    results = {}
    total_zeros = 0
    total_ones = 0
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        zeros_count, ones_count = count_zeros_and_ones_in_last_column(file_path)
        if zeros_count is not None and ones_count is not None:
            results[csv_file] = (zeros_count, ones_count)
            total_zeros += zeros_count
            total_ones += ones_count
    
    return results, (total_zeros, total_ones)

def count_columns_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                num_columns = len(df.columns)
                print(f"{filename}: {num_columns} columns")
            except Exception as e:
                print(f"Failed to read {filename}. Reason: {e}")

if __name__ == "__main__":

    folder_path = '../../ts_stage/ts_data/train' 
    results, (total_zeros, total_ones) = count_zeros_and_ones_in_folder(folder_path)

    # for csv_file, (zeros_count, ones_count) in results.items():
    #     print(f"{csv_file}: Zeros={zeros_count}, Ones={ones_count}")

    print(f"Total: Zeros={total_zeros}, Ones={total_ones}")
    # folder_path = '../data/output_csv_label'
    # folder_path = '../../ts_stage/ts_data/train'
    # count_columns_in_folder(folder_path)