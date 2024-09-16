import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Assuming the DeepLSTM model is defined in model.py
from ts_model import ResidualLSTMModel, ResidualLSTMModel2, DeepLSTM, TCN
from ts_dataset import create_csv_dataloaders, create_csv_test_dataloaders # Assuming this function is defined in csv_sequence_dataloader.py

# 定义TCN模型
class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 每一层的膨胀系数
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * dilation_size,
                          dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 2)  # 最后接一个全连接层，输出类别数为2

    def forward(self, x):
        # TCN expects input of shape (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)  # 将输入转换为 (batch_size, input_size, sequence_length)
        y = self.network(x)
        # 取最后一帧的特征向量
        out = y[:, :, -1]
        return self.fc(out)
    
def calculate_metrics(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def create_dir_with_incrementing_name(base_path, base_name):
    index = 0
    path = os.path.join(base_path, base_name)
    while os.path.exists(path):
        index += 1
        path = os.path.join(base_path, f"{base_name}{index}")
    os.makedirs(path)
    return path

def plot_train_metrics(train_losses, precisions, recalls, f1_scores, result_dir):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # Training and Validation Loss
    axs[0, 0].plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    # axs[0, 0].plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].legend()

    # Precision
    axs[0, 1].plot(range(1, len(precisions) + 1), precisions, label='Precision', color='green')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].set_title('Precision Over Epochs')

    # Recall
    axs[1, 0].plot(range(1, len(recalls) + 1), recalls, label='Recall', color='red')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].set_title('Recall Over Epochs')

    # F1 Score
    axs[1, 1].plot(range(1, len(f1_scores) + 1), f1_scores, label='F1 Score', color='purple')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('F1 Score')
    axs[1, 1].set_title('F1 Score Over Epochs')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'metrics_train.png'))
    plt.close()
def plot_val_metrics(val_losses, precisions, recalls, f1_scores, result_dir):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # Training and Validation Loss
    # axs[0, 0].plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    axs[0, 0].plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].legend()

    # Precision
    axs[0, 1].plot(range(1, len(precisions) + 1), precisions, label='Precision', color='green')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].set_title('Precision Over Epochs')

    # Recall
    axs[1, 0].plot(range(1, len(recalls) + 1), recalls, label='Recall', color='red')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].set_title('Recall Over Epochs')

    # F1 Score
    axs[1, 1].plot(range(1, len(f1_scores) + 1), f1_scores, label='F1 Score', color='purple')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('F1 Score')
    axs[1, 1].set_title('F1 Score Over Epochs')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'metrics_val.png'))
    plt.close()

def dynamic_adjust_pos_weight(pos_weight, true_positive_val, false_positive_val, false_negative_val):
    
    recall_val = true_positive_val / (true_positive_val + false_negative_val + 1e-8)
    precision_val = true_positive_val / (true_positive_val + false_positive_val + 1e-8)

    if recall_val < 0.3:
        new_pos_weight = min(pos_weight * 1.1, 10.0)  
    elif precision_val < 0.5:
        new_pos_weight = max(pos_weight * 0.9, 0.1) 
    else:
        new_pos_weight = pos_weight 

    return new_pos_weight, torch.tensor([new_pos_weight]).to(device)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 超参数设置
input_size = 42
# num_layers = 30
output_size = 1
seq_length = 40
batch_size = 20
num_epochs = 120
learning_rate = 0.0001
step_size = 1
kernel_size = 3
num_channels = [256, 256, 256]
# num_channels = [256, 64, 64, 64]
# num_channels = [64, 64]
dropout = 0.3
hidden_size = 64
num_layers = 3
# 模型实例化
model = TCN(input_size, output_size, num_channels, kernel_size, dropout)
# model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 数据加载
folder_path = "ts_data"
train_loader, val_loader = create_csv_dataloaders(folder_path, seq_length, batch_size, step_size=step_size)

# Loss 和优化器
num_positives = 5493
num_negatives = 10869
pos_weight = num_negatives / num_positives
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
# criterion = torch.nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

# 创建保存模型和结果的目录
model_dir = create_dir_with_incrementing_name("./stage2_runs", "exp")
result_dir = create_dir_with_incrementing_name("./stage2_runs", "result")

best_f1_score = 0.0
best_model_path = ""

# 记录各项指标的列表
train_losses = []
val_losses = []
batch_count = 0
precisions_train = []
recalls_train = []
f1_scores_train = []
precisions_val = []
recalls_val = []
f1_scores_val = []
from tqdm import tqdm
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    true_positive_train = 0
    false_positive_train = 0
    false_negative_train = 0
    for i, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
    # for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # print(x_batch.shape, y_batch.shape)
        optimizer.zero_grad()
        outputs = model(x_batch.unsqueeze(0))
        # outputs = model(x_batch)
        loss = criterion(outputs.squeeze(-1), y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_count += 1
        predictions = (outputs.squeeze(-1) >= 0.5).float()
        true_positive_train += ((predictions == 1) & (y_batch == 1)).sum().item()
        false_positive_train += ((predictions == 1) & (y_batch == 0)).sum().item()
        false_negative_train += ((predictions == 0) & (y_batch == 1)).sum().item()

    avg_loss = running_loss / len(train_loader)
    # if batch_count > 0:
    #     avg_loss = running_loss / batch_count  # 改为用 batch_count 计算平均损失
    # else:
    #     avg_loss = 0
    train_losses.append(avg_loss)  # 记录训练损失

    precision_train, recall_train, f1_score_train = calculate_metrics(true_positive_train, false_positive_train, false_negative_train)
    precisions_train.append(precision_train)
    recalls_train.append(recall_train)
    f1_scores_train.append(f1_score_train)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    print(f'Train Precision: {precision_train:.4f}, Train Recall: {recall_train:.4f}, Train F1 Score: {f1_score_train:.4f}')

    # plot_metrics(train_losses, precisions_train, recalls_train, f1_scores_train, result_dir)


    # 验证阶段
    model.eval()
    val_loss = 0.0
    batch_count = 0
    true_positive_val = 0
    false_positive_val = 0
    false_negative_val = 0

    all_ground_truth = []
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
            val_outputs = model(x_val_batch.unsqueeze(0))
            # val_outputs = model(x_val_batch)
            val_predictions = val_outputs.squeeze()
            val_predictions_rounded = (val_predictions >= 0.5).float()

            v_loss = criterion(val_outputs.squeeze(-1), y_val_batch)
            val_loss += v_loss.item()
            batch_count += 1
            true_positive_val += ((val_predictions_rounded == 1) & (y_val_batch == 1)).sum().item()
            false_positive_val += ((val_predictions_rounded == 1) & (y_val_batch == 0)).sum().item()
            false_negative_val += ((val_predictions_rounded == 0) & (y_val_batch == 1)).sum().item()

            # all_ground_truth.extend(y_val_batch.cpu().numpy())
            # all_predictions.extend(val_predictions_rounded.cpu().numpy())
            # all_confidences.extend(val_predictions.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    # if batch_count > 0:
    #     avg_val_loss = running_loss / batch_count  # 改为用 batch_count 计算平均损失
    # else:
    #     avg_val_loss = 0
    val_losses.append(avg_val_loss)  # 记录验证损失
    precision_val, recall_val, f1_score_val = calculate_metrics(true_positive_val, false_positive_val, false_negative_val)
    precisions_val.append(precision_val)
    recalls_val.append(recall_val)
    f1_scores_val.append(f1_score_val)

    print(f'Validation Loss: {avg_val_loss:.4f}')
    print(f'Validation Precision: {precision_val:.4f}, Validation Recall: {recall_val:.4f}, Validation F1 Score: {f1_score_val:.4f}')

    # 动态调整 pos_weight
    # pos_weight, pos_weight_tensor = dynamic_adjust_pos_weight(pos_weight, true_positive_val, false_positive_val, false_negative_val)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # 保存最佳模型
    if f1_score_val > best_f1_score:
        best_f1_score = f1_score_val
        best_model_path = os.path.join(model_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved at epoch {epoch+1} with F1 Score: {f1_score_val:.4f}')

    # 每5个epoch保存一次模型
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch+1}.pth'))
        print(f'Model saved at epoch {epoch+1}')

# 训练完成后绘制和保存指标图
plot_train_metrics(train_losses, precisions_train, recalls_train, f1_scores_train, result_dir)
plot_val_metrics(val_losses, precisions_val, recalls_val, f1_scores_val, result_dir)

print('Training complete.')

# # Convert the best model to ONNX
# onnx_file_path = os.path.join(model_dir, "tcn_best_model.onnx")
# dummy_input = torch.zeros(1, 1, seq_length, input_size).to(device)

# model.load_state_dict(torch.load(best_model_path))
# torch.onnx.export(model,
#                   dummy_input,
#                   onnx_file_path,
#                   export_params=True,
#                   opset_version=12,
#                   do_constant_folding=True,
#                   input_names=['input'],
#                   output_names=['output'],
#                   dynamic_axes=None)

# print(f"Best model has been converted to ONNX and saved to {onnx_file_path}")

# model_path = "./stage2_runs/exp5/best_model.pth" 
model_path = best_model_path
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)
batch_size = 1
folder_path = "ts_data"
test_loader = create_csv_test_dataloaders(folder_path, seq_length, batch_size, step_size=step_size)

# 初始化计算指标的变量
true_positive = 0
false_positive = 0
false_negative = 0
all_predictions = []
all_confidences = []
all_ground_truth = []

with torch.no_grad():
    for x_test_batch, y_test_batch in test_loader:
        x_test_batch, y_test_batch = x_test_batch.to(device), y_test_batch.to(device)

        # 前向传播
        test_outputs = model(x_test_batch.unsqueeze(0))
        # test_outputs = model(x_test_batch)
        test_predictions = test_outputs.squeeze()
        test_predictions_rounded = (test_predictions >= 0.5).float()
        if test_predictions_rounded.dim() == 0:
            test_predictions_rounded = torch.tensor([test_predictions_rounded.item()], device=device)
        else:
            test_predictions_rounded = test_predictions_rounded

        y_test_batch = y_test_batch.float()
        # 计算 TP, FP, FN
        true_positive += ((test_predictions_rounded == 1) & (y_test_batch == 1)).sum().item()
        false_positive += ((test_predictions_rounded == 1) & (y_test_batch == 0)).sum().item()
        false_negative += ((test_predictions_rounded == 0) & (y_test_batch == 1)).sum().item()

        # 收集数据用于进一步分析或绘图
        # all_ground_truth.extend(y_test_batch.cpu().numpy())
        # all_predictions.extend(test_predictions_rounded.cpu().numpy())
        # all_confidences.extend(test_predictions.cpu().numpy())

# 计算精度、召回率和 F1 分数
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 输出结果
print(f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1_score:.4f}')
