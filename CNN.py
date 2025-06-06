import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from collections import defaultdict

##onehot encoding
def onehot_encoding(file):
    peptide_seqs = pd.read_csv(file)
    seqs = peptide_seqs['seq'].tolist()
    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    amino_acid_to_index = {amino_acid: index for index, amino_acid in enumerate(amino_acids)}
    peptide_seqs_encoded_onehot = []
    for i in range(len(seqs)):
        seq = seqs[i]
        seq_index = [amino_acid_to_index[amino_acid] for amino_acid in seq]
        onehot_encoded = np.zeros((len(seq_index), len(amino_acids)))
        onehot_encoded[np.arange(len(seq_index)), seq_index] = 1
        seq_length = len(seq)
        max_length = 30
        if seq_length < 5 or seq_length > 30:
            continue
        if seq_length < max_length:
            padding_length = max_length - seq_length
            padding_matrix = np.zeros((padding_length, len(amino_acids)))
            seq_encoded = np.concatenate((onehot_encoded, padding_matrix), axis=0)
        else:
            seq_encoded = onehot_encoded
        peptide_seqs_encoded_onehot.append(seq_encoded)
    return peptide_seqs_encoded_onehot


# 创建PyTorch数据集
class SequenceDataset(Dataset):
    def __init__(self, seqs, labels, ids):
        self.seqs = seqs
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = torch.tensor(self.seqs[idx], dtype=torch.float)
        seq = seq.transpose(0, 1)  
        id = self.ids[idx]
        return seq, torch.tensor(self.labels[idx], dtype=torch.long),id


BBPs_real_seqs = onehot_encoding('./data_BBBPs.csv')
BBPs_generated_seqs = onehot_encoding('./unique_seqs.csv')
nonBBPs_seqs = onehot_encoding("./non_BBBPs_unique.csv")

ids = []
for prefix, seqs in zip(['R', 'G', 'N'], [BBPs_real_seqs, BBPs_generated_seqs, nonBBPs_seqs]):
    ids.extend([prefix + str(i+1) for i in range(len(seqs))])

all_data = [(seq, label, id_) for seq, label, id_ in zip(BBPs_real_seqs + BBPs_generated_seqs + nonBBPs_seqs,
                                                         [1]*len(BBPs_real_seqs) + [1]*len(BBPs_generated_seqs) + [0]*len(nonBBPs_seqs),
                                                         ids)]

data_by_type = {'R': [], 'G': [], 'N': []}
for seq, label, id_ in all_data:
    data_by_type[id_[0]].append((seq, label, id_))



sampled_data = data_by_type['R'] + random.sample(data_by_type['G'], len(nonBBPs_seqs)-len(BBPs_real_seqs)) + data_by_type['N']

def save_sampled_ids(sampled_ids, file_path):
    with open(file_path, 'w') as file:
        for id in sampled_ids:
            file.write(id + '\n')

sampled_ids = [id_ for _, _, id_ in sampled_data]


sequences_sampled, labels_sampled, ids_sampled = zip(*sampled_data)
dataset = SequenceDataset(list(sequences_sampled), list(labels_sampled), list(ids_sampled))

torch.manual_seed(123)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(20, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.8)

        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, 1)
        x = self.dropout(x)  
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
model = CNN()


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1, weight_decay=0.15)


def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

def add_noise(sequences, noise_factor=0.01):
    noisy_sequences = sequences + noise_factor * torch.randn_like(sequences)
    return torch.clamp(noisy_sequences, 0., 1.)  

def train_model(model, train_loader, val_loader, num_epochs):
    best_val_acc = 0.0
    best_model_state = None

    train_losses = []
    val_losses = []
    train_ACC = []
    val_ACC = []
    for epoch in range(num_epochs):
    
        model.train()
        train_loss, train_acc, train_precision, train_recall, train_f1 =0.0, [], [], [], []
        for inputs, labels, _ in train_loader:
            inputs = add_noise(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            acc, prec, rec, f1 = evaluate_metrics(labels.cpu().numpy(), predicted.cpu().numpy())
            train_acc.append(acc)
            train_precision.append(prec)
            train_recall.append(rec)
            train_f1.append(f1)

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_train_precision = sum(train_precision) / len(train_precision)
        avg_train_recall = sum(train_recall) / len(train_recall)
        avg_train_f1 = sum(train_f1) / len(train_f1)
        avg_train_loss = train_loss / len(train_loader)  
        train_losses.append(avg_train_loss) 
        train_ACC.append(avg_train_acc)
 
        model.eval()
        val_loss, val_acc, val_precision, val_recall, val_f1 = 0.0, [], [], [], []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                acc, prec, rec, f1 = evaluate_metrics(labels.cpu().numpy(), predicted.cpu().numpy())
                val_acc.append(acc)
                val_precision.append(prec)
                val_recall.append(rec)
                val_f1.append(f1)

        avg_val_acc = sum(val_acc) / len(val_acc)
        avg_val_precision = sum(val_precision) / len(val_precision)
        avg_val_recall = sum(val_recall) / len(val_recall)
        avg_val_f1 = sum(val_f1) / len(val_f1)
        avg_val_loss = val_loss / len(val_loader)  
        val_losses.append(avg_val_loss)
        val_ACC.append(avg_val_acc)
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, './best_model_CNN.pth')

        print(f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss}, "
            f"Train Accuracy: {avg_train_acc}, Precision: {avg_train_precision}, "
            f"Recall: {avg_train_recall}, F1: {avg_train_f1}, "
            f"Val Loss: {avg_val_loss}, "
            f"Val Accuracy: {avg_val_acc}, Precision: {avg_val_precision}, "
            f"Recall: {avg_val_recall}, F1: {avg_val_f1}")
            
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_ACC, label='Train Accuracy')
    ax2.plot(val_ACC, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def evaluate_and_plot(model_path, test_loader):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.eval()
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix_CNN',fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['nonBBPs', 'BBPs'])
    plt.yticks(tick_marks, ['nonBBPs', 'BBPs'])
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    
    plt.subplots_adjust(left=0.2)
    plt.savefig('confusion_matrix_test_CNN.pdf')
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=400, facecolor='white')
    plt.plot(fpr, tpr, label="ROC curve (AUC = {:.3f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('1-Specificity(FPR)', fontsize=12)
    plt.ylabel('Sensitivity(TPR)', fontsize=12)
    plt.title('ROC curve_CNN', fontsize=14,fontweight='bold')
    plt.legend(loc="lower right")
    plt.subplots_adjust(left=0.15, bottom = 0.12)
    plt.savefig('roc_curve_CNN.pdf')
    plt.close()
    return y_score


if __name__ == '__main__':
    train_model(model, train_loader, val_loader, num_epochs=200)