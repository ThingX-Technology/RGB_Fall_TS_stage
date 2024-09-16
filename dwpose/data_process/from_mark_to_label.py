import pandas as pd
import os
import csv 
from video_fall_mark import video_data_list, video_data_list_0914
def time_to_frame(time_str, frame_rate):
    minutes, seconds = map(float, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    frame_number = round(total_seconds * frame_rate)
    # print(f"frame_number is {frame_number}")
    return frame_number

def mark_frames(csv_path, output_path, time_ranges, frame_rate, frame_interval=5):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path, header=None)
    
    last_column_index = len(df.columns)
    # print(last_column_index)
    df[last_column_index] = 0
    
    if not time_ranges:
        df.drop(columns=[0], inplace=True)
        df.to_csv(output_path, index=False, header=False)
        # df.to_csv(output_temp_csv_path, index=False, header=False)
        print(f"No fall detected. Marked all frames as 0 and saved to {output_path} .")
        return
    
    active_ranges = [(time_to_frame(start, frame_rate), time_to_frame(end, frame_rate)) for start, end in time_ranges]
    # print(active_ranges)

    adjusted_ranges = [(start // frame_interval, end // frame_interval) for start, end in active_ranges]

    for start_row, end_row in adjusted_ranges:
        df.iloc[start_row:end_row, last_column_index] = 1

    df.drop(columns=[0], inplace=True)
    df.to_csv(output_path, index=False, header=False)
    # df.to_csv(output_temp_csv_path, index=False, header=False)
    print(f"Marked frames and saved to {output_path}.")


def prune_to_final_csv(input_file, output_folder, output_file_suffix):
    points_to_remove = [14, 15, 16, 17]
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
    # del rows[0]
    for row in rows:
        del row[0]

    for row in rows:
        for point_index in sorted(points_to_remove, reverse=True):
            start_index = point_index * 3
            del row[start_index:start_index + 3]

    base_name, ext = os.path.splitext(os.path.basename(input_file))
    output_file = f"{os.path.join(output_folder, base_name)}_{output_file_suffix}{ext}"

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    print(f"Processed data saved to {output_file}")
    return output_file

def process_videos(video_data_list, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # os.makedirs(output_temp_folder, exist_ok=True)
    for video_info in video_data_list:
        input_csv_path = video_info['csv_path']
        time_ranges = video_info['time_ranges']
        frame_rate = video_info.get('frame_rate', 30)
        frame_interval = video_info.get('frame_interval', 5)
        base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
        output_csv_path = os.path.join(output_folder, f"{base_name}_label.csv")
        # output_temp_csv_path = os.path.join(output_temp_folder, f"{base_name}_temp.csv")

        mark_frames(input_csv_path, output_csv_path, time_ranges, frame_rate, frame_interval)

    
if __name__ == "__main__":
    output_folder = '../data/output_csv_label'
    process_videos(video_data_list, output_folder)    
    