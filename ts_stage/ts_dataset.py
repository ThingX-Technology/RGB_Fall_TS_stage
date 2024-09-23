import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NumberSequenceDataset(Dataset):
    def __init__(self, seq_length, data_size, range_max):
        self.seq_length = seq_length
        self.data_size = data_size
        self.range_max = range_max
        self.data = self._generate_data()

    def _generate_data(self):
        sequences = np.array([np.arange(i, i + self.seq_length + 1) % self.range_max for i in range(self.data_size)])
        sequences = sequences / (self.range_max - 1)  # Normalize to 0-1  # Scale to 0-255
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index, :-1]
        y = self.data[index, -1]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

def create_dataloader(seq_length, data_size, range_max, batch_size):
    dataset = NumberSequenceDataset(seq_length, data_size, range_max)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def generate_test_data(seq_length, num_samples, range_max):
    sequences = np.array([np.arange(i, i + seq_length + 1) % range_max for i in range(num_samples)])
    sequences = sequences / (range_max - 1)  # Normalize to 0-1
    sequences = sequences * 255  # Scale to 0-255
    x_test = sequences[:, :-1]
    y_test = sequences[:, -1]
    return x_test, y_test

def save_test_data(x_test, y_test, x_file, y_file):
    np.save(x_file, x_test)
    np.save(y_file, y_test)





import os
import pandas as pd
import numpy as np
import torch
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, IterableDataset
class CSVSequenceDataset(Dataset):
    def __init__(self, folder_path, seq_length, step_size=1):
        self.seq_length = seq_length
        self.step_size = step_size
        self.data = []
        self.labels = []
        self._read_csv_files(folder_path)
        self.feature_num = self.data[0].shape[1]
        # print(f"feature num is {self.feature_num}")
        # print(f"the first data shape is {self.data[0].shape}")
        

    def _read_csv_files(self, folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path, dtype=np.float32)
                self._create_simple_sequences(df)

    def _create_simple_sequences(self, df):
        num_rows = df.shape[0]
        step_length = self.seq_length * self.step_size
        if num_rows < step_length:
            return  # Skip if the sequence is too short
        
        for start_idx in range(0, num_rows - step_length + 1):
            end_idx = start_idx + step_length
            sequence = df.iloc[start_idx:end_idx:self.step_size, :-1].values  # Features 
            # print(f"sequence shape : {sequence.shape}")
            falling_perctange = sum(df.iloc[start_idx:end_idx, -1]) / step_length
            falling_all_in_window = df.iloc[end_idx-1, -1] == 0 and df.iloc[start_idx, -1] == 0

            if 0.5 < falling_perctange or (0 < falling_perctange and falling_all_in_window):
                label = 1
            # elif falling_perctange > 0:
            #     continue
            else:
                label = 0
            
            self.data.append(sequence)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # x_coords = x[:, 0:self.feature_num:3]
        # y_coords = x[:, 1:self.feature_num:3]
        # confidence = x[:, 2:self.feature_num:3]
        # epsilon = 1e-8
        # x_min = x_coords.min(dim=1, keepdim=True)[0]
        # x_max = x_coords.max(dim=1, keepdim=True)[0]
        # y_min = y_coords.min(dim=1, keepdim=True)[0]
        # y_max = y_coords.max(dim=1, keepdim=True)[0]
        # x_coords_normalized = (x_coords - x_min) / (x_max - x_min + epsilon)
        # y_coords_normalized = (y_coords - y_min) / (y_max - y_min + epsilon)
        # x_normalized = torch.stack((x_coords_normalized, y_coords_normalized, confidence), dim=2)
        # x_normalized = x_normalized.view(x_normalized.size(0), -1)
        # return x_normalized, y
        
        return x, y

def create_csv_dataloaders(folder_path, seq_length, batch_size, train_ratio=0.8, step_size=1, shuffle=True):
    train_loader = DataLoader(CSVSequenceDataset(os.path.join(folder_path, 'train'), seq_length, step_size), batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(CSVSequenceDataset(os.path.join(folder_path, 'val'), seq_length, step_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
def create_csv_test_dataloaders(folder_path, seq_length, batch_size, train_ratio=0.8, step_size=1, shuffle=False):
    test_loader = DataLoader(CSVSequenceDataset(os.path.join(folder_path, 'test'), seq_length, step_size), batch_size=batch_size, shuffle=False)
    return test_loader

class FeatureDataset(Dataset):
    def __init__(self, feature_dir, label_dir, transform=None):
        """
        Args:
            feature_dir (str): Directory containing all .npy feature files.
            label_dir (str): Directory containing all .txt label files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # List all feature files
        self.feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
        
        # Ensure each feature file has a corresponding label file
        self.label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files must match."
        
        # Sort files to ensure matching
        self.feature_files.sort()
        self.label_files.sort()

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        feature_path = os.path.join(self.feature_dir, feature_file)
        label_path = os.path.join(self.label_dir, label_file)

        features = np.load(feature_path)
        with open(label_path, 'r') as f:
            label = int(f.read().strip()[0])

        if self.transform:
            features = self.transform(features)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def create_dataloaders(train_feature_dir, train_label_dir, val_feature_dir, val_label_dir, batch_size, transform=None):
    # Create FeatureDataset for training and validation
    train_dataset = FeatureDataset(train_feature_dir, train_label_dir, transform=transform)
    val_dataset = FeatureDataset(val_feature_dir, val_label_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_number():

    # The provided numbers as a single list
    numbers = [
        [119, 166, 130, 94, 0, 0, 0, 0, 0, 242, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [119, 164, 130, 90, 0, 0, 0, 0, 0, 244, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [116, 164, 130, 90, 0, 0, 0, 0, 0, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [114, 166, 133, 94, 0, 0, 0, 0, 0, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [117, 164, 127, 90, 0, 0, 0, 0, 0, 246, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [116, 164, 130, 90, 0, 0, 0, 0, 0, 242, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [113, 164, 137, 90, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [113, 164, 137, 90, 0, 0, 0, 0, 0, 244, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [113, 164, 137, 90, 0, 0, 0, 0, 0, 242, 30, 34, 60, 69, 1, 25, 0, 0, 0, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [113, 164, 137, 90, 0, 0, 0, 0, 0, 241, 30, 34, 60, 69, 1, 16, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [113, 164, 130, 90, 0, 0, 0, 0, 0, 239, 30, 34, 60, 69, 0, 41, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [113, 162, 137, 86, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [111, 164, 133, 90, 0, 0, 0, 0, 0, 244, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [111, 164, 133, 90, 0, 0, 0, 0, 0, 243, 25, 37, 51, 74, 79, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [113, 164, 137, 90, 0, 0, 0, 0, 0, 241, 30, 34, 60, 69, 3, 6, 0, 0, 0, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    # Convert the list to a NumPy array and reshape it
    array = np.array(numbers).reshape(1, 1, 15, 50)
    return array



def just_test(sequence, ort_session):
       
        # # Prepare the input for the ONNX model
        input_name = ort_session.get_inputs()[0].name
        input_tensor = sequence.astype(np.float32)

        # Run inference
        output = ort_session.run(None, {input_name: input_tensor})[0]
        
        input_tensor = input_tensor.flatten().astype(np.uint8)
        print(f"Model output: {output}")

# if __name__ == "__main__":
#     train_feature_dir = '/home/yolo/yolov5_ch/yolov5/dataset/classificationC030_nl_with_fall_case/train/features'
#     train_label_dir = '/home/yolo/yolov5_ch/yolov5/dataset/classificationC030_nl_with_fall_case/train/labels'
#     val_feature_dir = '/home/yolo/yolov5_ch/yolov5/dataset/classificationC030_nl_with_fall_case/val/features'
#     val_label_dir = '/home/yolo/yolov5_ch/yolov5/dataset/classificationC030_nl_with_fall_case/val/labels'
#     batch_size = 32

#     train_loader, val_loader = create_dataloaders(
#         train_feature_dir, train_label_dir, val_feature_dir, val_label_dir, batch_size
#     )

#     # Iterate through the training data loader
#     for x_batch, y_batch in train_loader:
#         print("Train batch:")
#         print("Features:", x_batch.shape)
#         print("Labels:", y_batch.shape)
    
#     # Iterate through the validation data loader
#     for x_batch, y_batch in val_loader:
#         print("Validation batch:")
#         print("Features:", x_batch.shape)
#         print("Labels:", y_batch.shape)

if __name__ == "__main__":
    import onnxruntime as ort
    
    # Load the ONNX model
    onnx_model_path = '/home/yolo/yolov5_ch/stage2/runs/exp176/tcn_best_model.onnx'  # Replace with your actual ONNX model path
    ort_session = ort.InferenceSession(onnx_model_path)

    # sequence = np.full((1, 1, 15, 50), 1)
    # # sequence[0, 0, 0, :50] = 1
    # #sequence = get_number()/255
    # # np.save("examine.npy", sequence.astype(np.uint8))
    # just_test(sequence, ort_session)
    
    folder_path = r"/home/yolo/yolov5_ch/stage2/data/3040guanyu_cos"
    seq_length = 30
    batch_size = 32

    # Assuming the create_csv_dataloaders function is defined elsewhere and works correctly
    train_loader, val_loader = create_csv_dataloaders(folder_path, seq_length, batch_size)

    print(len(train_loader))

    # Create a directory to save the numpy files if it doesn't exist
    save_dir = 'saved_sequences11'
    os.makedirs(save_dir, exist_ok=True)

    # Counter for naming the .npy files
    file_counter = 0
    
    # Function to test the sequence with the ONNX model and save the result to txt
    def test_and_save(sequence, file_counter):
        file_name_npy = os.path.join(save_dir, f'{file_counter}.npy')
        file_name_txt = os.path.join(save_dir, f'{file_counter}.txt')

        # Save the numpy array
        # sequence = np.full((1, 1, 15, 50), 1)
        # if file_counter == 76:
        #     print(sequence.numpy().astype(np.float32))
        #     pass
        np.save(file_name_npy, sequence.numpy().astype(np.float32))

        # # Prepare the input for the ONNX model
        input_name = ort_session.get_inputs()[0].name
        input_tensor = sequence.numpy().astype(np.float32)

        # Run inference
        output = ort_session.run(None, {input_name: input_tensor/255})[0]
        
        input_tensor = input_tensor.flatten().astype(np.uint8)
        # Save the model output to txt
        # np.savetxt(file_name_txt, input_tensor, fmt='%i')
        
        print(f"Saved {file_name_npy} and {file_name_txt}")
        print(f"Model output: {output}")

    

    # Iterate through the train_loader and save each sequence
    for x_batch, y_batch in train_loader:
        if file_counter > 50:
            break
        for i in range(0, x_batch.shape[0]):  # Iterate through the batch
            if y_batch[i] > 0:
                sequence = x_batch[i].reshape(1, 1, seq_length, 96)  # Reshape to (1, 1, 15, 50)
                test_and_save(sequence, file_counter)
                file_counter += 1
    
    print("first half")
    for x_batch, y_batch in train_loader:
        if file_counter > 100:
            break
        for i in range(0, x_batch.shape[0]):  # Iterate through the batch
            if y_batch[i] == 0:
                sequence = x_batch[i].reshape(1, 1, seq_length, 96)  # Reshape to (1, 1, 15, 50)
                test_and_save(sequence, file_counter)
                file_counter += 1


    print(f"Total files saved: {file_counter}")

