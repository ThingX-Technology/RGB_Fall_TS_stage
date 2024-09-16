import os
import cv2
import torch
import warnings
import argparse
import numpy as np
from collections import deque
from ts_stage.ts_model import TCN 
from controlnet_aux import DWposeDetector

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")


def main(weights_path, save_video, fps, camera_source):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 42
    output_size = 1
    num_channels = [256, 256, 256]
    kernel_size = 3
    model = TCN(input_size, output_size, num_channels, kernel_size)
    # model_path = "weights/best_model_0916_3.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)

    pose_detector = DWposeDetector(
        det_config='D:/lytvton/TTFproject/control_dwpose/dwpose/config/yolox_l_8xb8-300e_coco.py',
        pose_config='D:/lytvton/TTFproject/control_dwpose/dwpose/config/dwpose-l_384x288.py',
        det_ckpt='D:/lytvton/TTFproject/control_dwpose/weights/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        pose_ckpt='D:/lytvton/TTFproject/control_dwpose/weights/dw-ll_ucoco_384.pth',
        device=device
    )

    seq_length = 40
    pose_data_queue = deque(maxlen=seq_length)

    weights_name = os.path.basename(weights_path).replace(".pth", "")
    video_name = f"{weights_name}_output.mp4"
    # save_video = True
    video_writer = None
    if save_video:
        print("Recording started.")
        output_dir = 'recorded_videos'
        os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cap = cv2.VideoCapture(camera_source)
    print("Starting video stream... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        H, W = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose, persons_data = pose_detector(frame)

        if len(persons_data) > 0:
            keypoints = np.array(persons_data[0]).reshape(-1, 3)
            keypoints = [(x * W, y * H, conf) for x, y, conf in keypoints]
            pose_keypoints = [val for kp in keypoints for val in kp]
        else:
            pose_keypoints = [0] * 42

        pose_data_queue.append(pose_keypoints)

        for (x, y, conf) in keypoints:
            if conf > 0.5: 
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        if len(pose_data_queue) == seq_length:
            input_data = torch.tensor([list(pose_data_queue)], dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(input_data.unsqueeze(0))
                prediction = (output >= 0.5).float().item()

            status_text = 'Fall' if prediction == 1 else 'Normal'
            color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
            cv2.putText(frame, f'State: {status_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        if save_video and video_writer is None:
            video_writer = cv2.VideoWriter(os.path.join(output_dir, video_name), fourcc, fps, (frame.shape[1], frame.shape[0]))
            

        if save_video and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print("Recording ended.")
    cv2.destroyAllWindows()
    print("Video stream ended.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detection with TCN model')
    parser.add_argument('--weights_path', type=str, default="weights/best_model_0916_3.pth", help='Path to the TCN model weights')
    parser.add_argument('--save_video', action='store_true', help='Whether to save the output video')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second for the saved video')
    parser.add_argument('--source', type=int, default=1, help='Camera source (0 for local, 1 for USB)')
    
    args = parser.parse_args()
    main(args.weights_path, args.save_video, args.fps, args.source)
