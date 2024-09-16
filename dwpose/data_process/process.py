from splits_csv import split_csv_files, remove_first_column_from_csvs, merge_folders
from from_mark_to_label import process_videos
from video_fall_mark import video_data_list, video_data_list_0914, video_data_list_0916
from count_folder_label_num import count_zeros_and_ones_in_folder

if __name__ == '__main__':
    output_folder = '../data/output_csv_label'
    process_videos(video_data_list_0914, output_folder)  

    source1_folder = '../data/output_csv_label'
    source2_folder = '../data/origin_csv_split'
    split_source_folder = '../../ts_stage/ts_data/splits_csv' 
    
    # remove_first_column_from_csvs(source1_folder)
    merge_folders(source1_folder, source2_folder, split_source_folder)

    target_folder = '../../ts_stage/ts_data'  
    split_csv_files(split_source_folder, target_folder, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10)

    folder_path = '../../ts_stage/ts_data/train' 
    results, (total_zeros, total_ones) = count_zeros_and_ones_in_folder(folder_path)

    # for csv_file, (zeros_count, ones_count) in results.items():
    #     print(f"{csv_file}: Zeros={zeros_count}, Ones={ones_count}")
    print(f"Total: Zeros={total_zeros}, Ones={total_ones}")

