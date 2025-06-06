import os
import math
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from Bio.Align import substitution_matrices
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

BBBPs = "./data_BBBPs.xlsx"

def onehot_encoding(file):
    peptide_seqs = pd.read_excel(file)
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

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width = x.size()
        
        query = self.query_conv(x).view(batch_size, -1, width)  # B x C/8 x N
        key = self.key_conv(x).view(batch_size, -1, width)  # B x C/8 x N
        energy = torch.bmm(query.permute(0, 2, 1), key)  # B x N x N
        attention = F.softmax(energy, dim=-1)  # B x N x N
        value = self.value_conv(x).view(batch_size, -1, width)  # B x C x N

        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, channels, width)
        
        out = self.gamma * out + x  
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.seq_length = 30  
        self.num_classes = 20  
        self.latent_dim = 200  

        self.linear1 = nn.Linear(self.latent_dim, 128) 
        self.linear2 = nn.Linear(128, 256)  
        self.bn2 = nn.BatchNorm1d(256)  
        self.linear3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear4 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.linear5 = nn.Linear(1024, self.seq_length * self.num_classes)

        self.tanh = nn.Tanh()  
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

        
        self.conv1 = nn.Conv1d(self.num_classes, 256, kernel_size=5, stride=1, padding=2)  
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2)  
        self.conv3 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)  
        self.conv4 = nn.Conv1d(64, self.num_classes, kernel_size=5, stride=1, padding=2)  

        
        self.bn_conv1 = nn.BatchNorm1d(256)
        self.bn_conv2 = nn.BatchNorm1d(128)
        self.bn_conv3 = nn.BatchNorm1d(64)

        
        self.attn1 = SelfAttention(256)
        self.attn2 = SelfAttention(128)

    def forward(self, z):
        
        z = self.linear1(z)
        z = self.leakyrelu(z)

        z = self.linear2(z)
        z = self.bn2(z)
        z = self.leakyrelu(z)

        z = self.linear3(z)
        z = self.bn3(z)
        z = self.leakyrelu(z)

        z = self.linear4(z)
        z = self.bn4(z)
        z = self.leakyrelu(z)

        z = self.linear5(z)
        z = self.tanh(z)
        z = z.view(-1, self.num_classes, self.seq_length)

        z = self.conv1(z)
        z = self.bn_conv1(z)
        z = self.leakyrelu(z)
        z = self.attn1(z)  

        z = self.conv2(z)
        z = self.bn_conv2(z)
        z = self.leakyrelu(z)
        z = self.attn2(z)  

        z = self.conv3(z)
        z = self.bn_conv3(z)
        z = self.leakyrelu(z)

        z = self.conv4(z)
        z = self.tanh(z)

        z = z.view(-1, self.seq_length, self.num_classes)
        return z

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.seq_length = 30
        self.num_classes = 20

        self.linear1 = nn.Linear(self.seq_length * self.num_classes, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        

    def forward(self, x):
        x = x.view(-1, self.seq_length * self.num_classes)
        x = self.linear1(x)
        x = self.leakyrelu(x)
        x = self.linear2(x)
        x = self.leakyrelu(x)
        x = self.linear3(x)
        return x

def cal_gp(D, real_peptides, fake_peptides):
    batch_size = real_peptides.size(0)  
    alpha = torch.rand(batch_size, 1, 1, device=real_peptides.device)  
    alpha = alpha.expand_as(real_peptides)
    interpolated_peptides = alpha * real_peptides + (1 - alpha) * fake_peptides  
    interpolated_peptides.requires_grad_(True)  
    d_interpolated = D(interpolated_peptides)  
    gradients = torch.autograd.grad(
        outputs=d_interpolated,  
        inputs=interpolated_peptides,  
        grad_outputs=torch.ones_like(d_interpolated),  
        create_graph=True,  
        retain_graph=True  
    )[0]  
    gradients = gradients.view(batch_size, -1)  
    gradients_norm = gradients.norm(2, dim=1)  
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()  

    return gradient_penalty  

torch.manual_seed(123)

def save_metrics_to_csv(epoch, generator_losses, discriminator_losses, gradient_penalties, wasserstein_distance):
    
    data = {
        'Epoch': list(range(1, epoch + 1)),
        'Generator Loss': generator_losses,
        'Discriminator Loss': discriminator_losses,
        'Gradient Penalty': gradient_penalties,
        'Wasserstein Distance': wasserstein_distance
    }
    
    
    df = pd.DataFrame(data)
    
    
    df.to_csv('./Results/training_metrics.csv', index=False)

def train_model(sequence, epochs):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    encoded_seqs = onehot_encoding(sequence)
    encoded_seqs = np.array(encoded_seqs)
    encoded_seqs = torch.tensor(encoded_seqs, dtype=torch.float).to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.9))

    
    batch_size = 64
    num_workers = 3
    n_critic = 5

    generator_losses = []
    discriminator_losses = []
    gradient_penalties = []
    wasserstein_distance = []

    dataloader = DataLoader(
        dataset=encoded_seqs,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    for epoch in range(epochs):
        for i, real_data in enumerate(dataloader):
            bs = real_data.shape[0]

            for _ in range(n_critic):
                discriminator.zero_grad()
                real_output = discriminator(real_data)
                noise = torch.randn(bs, 200, device=device)
                fake_data = generator(noise)
                fake_output = discriminator(fake_data)
                
                
                gradient_penalty = cal_gp(discriminator, real_data, fake_data)
                gradient_penalty_weight = 10  
                loss_D = -torch.mean(real_output) + torch.mean(fake_output) + gradient_penalty_weight * gradient_penalty
                wd = torch.mean(real_output) - torch.mean(fake_output)
                loss_D.backward()
                optimizer_d.step()
                

           
            generator.zero_grad()
            noise = torch.randn(bs, 200, device=device)
            fake_data = generator(noise)
            loss_G = -torch.mean(discriminator(fake_data))
            loss_G.backward()
            optimizer_g.step()


        generator_losses.append(loss_G.item())
        discriminator_losses.append(loss_D.item())
        gradient_penalties.append(gradient_penalty.item())
        wasserstein_distance.append(wd.item())

        
        print(f"Epoch {epoch+1}/{epochs}  Generator Loss: {loss_G.item()}  Discriminator Loss: {loss_D.item()}  Gradient Penalty: {gradient_penalty.item()}  Wasserstein Distance: {wd.item()}")

        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

        
        ax1.plot(discriminator_losses, label='Discriminator Loss')
        ax1.plot(generator_losses, label='Generator Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        
        ax2.plot(gradient_penalties, label='Gradient Penalty')
        ax2.plot(wasserstein_distance, label='Wasserstein Distance')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        
        plt.tight_layout()
        plt.savefig(f'./Results/Loss_{epoch+1}.pdf')
        plt.close()
        
        
        torch.save(generator, f'./Results/generator_epoch_{epoch+1}.pth')
    
        save_metrics_to_csv(epoch+1, generator_losses, discriminator_losses, gradient_penalties, wasserstein_distance)
    


def data_generator(number, model_path):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    noise = torch.randn(number, 200, device=device)
    generator = torch.load(model_path, map_location=device)
    generator.eval()
    with torch.no_grad():
        generated_data = generator(noise).detach().cpu().numpy()
    return generated_data

def decode_sequences_cosine_similarity(input):
    BLOSUM62 = pd.DataFrame(substitution_matrices.load("BLOSUM62")).iloc[:20, :20]
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    BLOSUM62.index = amino_acids
    BLOSUM62.columns = amino_acids

    decoded_sequences = []
    
    for i in range(len(input)):
        pep = []
        for j in range(len(input[i])):
            cs = cosine_similarity(np.array(input[i][j]).reshape(1, -1), BLOSUM62)
            if cs.max() > 0.3:
                aa = amino_acids[np.argmax(cs)]
                pep += aa
            else:
                break
        if len(pep) >= 5:
            decoded_sequences.append(''.join(pep))
    return decoded_sequences

def tsne(sequence, number, model_path, n_components):
    BBBPs_real = onehot_encoding(sequence)

    BBBPs_generated = data_generator(number, model_path)
    BBBPs_generated = decode_sequences_cosine_similarity(BBBPs_generated)
    BBBPs_generated = pd.DataFrame(BBBPs_generated, columns=['seq'])
    BBBPs_generated = BBBPs_generated.drop_duplicates()
    BBBPs_generated.to_excel('./BBBPs_generated.xlsx', index=False)
    
    BBBPs_generated = onehot_encoding('./BBBPs_generated.xlsx')

    matrix_seq_real = []
    for i in range(len(BBBPs_real)):
        trans_seq_real = np.array (BBBPs_real[i]).flatten()
        matrix_seq_real.append(trans_seq_real)
    trans_seqs_real = np.array(matrix_seq_real)
    trans_seqs_real = pd.DataFrame(trans_seqs_real)

    matrix_seq_generated = []
    for i in range(len(BBBPs_generated)):
        trans_seq_generated = np.array (BBBPs_generated[i]).flatten()
        matrix_seq_generated.append(trans_seq_generated)
    trans_seqs_generated = np.array(matrix_seq_generated)
    trans_seqs_generated = pd.DataFrame(trans_seqs_generated)

    
    X = pd.concat([trans_seqs_real, trans_seqs_generated], axis=0, ignore_index=True)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components = n_components)
    result_tsne = tsne.fit_transform(X)
    vector = ['Real'] * len(BBBPs_real) + ['GAN'] * len(BBBPs_generated)
    result_tsne = pd.concat([pd.DataFrame(result_tsne), pd.DataFrame(vector, columns=['type'])], axis=1)

    import matplotlib.pyplot as plt

 
    real_points = result_tsne[result_tsne['type'] == 'Real']
    generated_points = result_tsne[result_tsne['type'] == 'GAN']


    plt.scatter(real_points.iloc[:,0], real_points.iloc[:,1], color='blue', label='Real', s=20)
    plt.scatter(generated_points.iloc[:,0], generated_points.iloc[:,1], color='red', label='GAN', s=2)
    

    plt.title('Scatter Plot')
    plt.xlabel('TSNE_1')
    plt.ylabel('TSNE_2')
  
    plt.legend()
  
    plt.savefig('./tsne.pdf')
    plt.close()


if __name__ == '__main__':
    train_model(BBBPs, 15000)