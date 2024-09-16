import os
import csv
import torch 
import imageio
import warnings
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from controlnet_aux import DWposeDetector

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

def process_video(video_name):
    video_path = os.path.join(in_video_folder, video_name)
    video = VideoReader(video_path, ctx=cpu(0))
    fps = video.get_avg_fps()

    out_video_path = os.path.join(out_video_folder, video_name)
    out_csv_path = os.path.join(out_csv_folder, os.path.splitext(video_name)[0] + '.csv')

    with open(out_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        poses = []
        for i in tqdm(range(len(video))):
            frame = Image.fromarray(video[i].asnumpy())
            pose, persons_data = pose_det(frame)
            row = [i] + persons_data[0]
            writer.writerow(row)
            poses.append(pose)

    print(f"开始保存 {out_video_path}，帧率为 {fps}")
    imageio.mimsave(out_video_path, poses, fps=fps)

if __name__ == "__main__":
    
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
	pose_det = DWposeDetector(
		det_config='D:/lytvton/TTFproject/control_dwpose/dwpose/config/yolox_l_8xb8-300e_coco.py',
		pose_config='D:/lytvton/TTFproject/control_dwpose/dwpose/config/dwpose-l_384x288.py',
        det_ckpt='D:/lytvton/TTFproject/control_dwpose/weights/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        pose_ckpt='D:/lytvton/TTFproject/control_dwpose/weights/dw-ll_ucoco_384.pth',
		device=device
	)
		
	# in_video_folder = '../data/input_video/'
	in_video_folder = 'data/input_temp/'
	out_video_folder = 'data/output_video/'
	out_csv_folder = 'data/output_csv_origin/'
	# video_names = ['two_people.mp4', 'another_video.mp4']
	video_names = [f for f in os.listdir(in_video_folder) if f.endswith('.mp4')]
      
	for video_name in video_names:
		process_video(video_name)