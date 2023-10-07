import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_base(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=False, batch_norm=False):
        super(VAE_base, self).__init__()

        encoder_layers = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
        encoder_layers.append(nn.ReLU())
        if dropout:
            encoder_layers.append(nn.Dropout(0.2))
        encoder_layers.append(nn.Linear(hidden_dim, latent_dim * 2))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [nn.Linear(latent_dim, hidden_dim)]
        if batch_norm:
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)
        self.z = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # Encode
        enc_output = self.encoder(x)
        mu, logvar = enc_output.chunk(2, dim=1)  # Split into mean and log-variance
        z = self.reparameterize(mu, logvar)
        self.z = z
        # Decode
        dec_output = self.decoder(z)
        return dec_output, mu, logvar
    
    def get_z(self):
        return self.z
    

class VAE256(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(VAE256, self).__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.z = None
        # Encoder
        self.fc1 = nn.Linear(self.input_dim, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        self.fc2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        self.fc3_mu = nn.Linear(2048, 256)    # mean
        self.fc3_logvar = nn.Linear(2048, 256) # log variance
        
        # Decoder
        self.fc4 = nn.Linear(256, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.dropout4 = nn.Dropout(self.dropout_rate)
        
        self.fc5 = nn.Linear(2048, 4096)
        self.bn5 = nn.BatchNorm1d(4096)
        self.dropout5 = nn.Dropout(self.dropout_rate)
        
        self.fc6 = nn.Linear(4096, self.input_dim)

    def encode(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout1(h)
        
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout2(h)
        
        return self.fc3_mu(h), self.fc3_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.bn4(self.fc4(z)))
        h = self.dropout4(h)
        
        h = F.relu(self.bn5(self.fc5(h)))
        h = self.dropout5(h)
        
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        self.z = z
        return self.decode(z), mu, logvar
    
    def get_z(self):
        return self.z


# ------VQ-VAE------
# Define the MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# Define the VQ layer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, x):
        # Compute distances
        distances = (x.pow(2).sum(1, keepdim=True)
                    + self.embedding.weight.pow(2).sum(1)
                    - 2 * x @ self.embedding.weight.t())
        
        # Encoding
        encoding_indices = distances.argmin(1)
        encodings = F.one_hot(encoding_indices, self.embedding.weight.size(0))
        encodings = encodings.type_as(x)
        
        # Use encodings to compute quantized version of the input
        quantized = encodings @ self.embedding.weight
        
        return quantized, encodings, encoding_indices

# VQ-VAE model
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings, dropout_rate=0.2):
        super(VQVAE, self).__init__()
        
        self.encoder = MLP(input_dim, hidden_dim, embedding_dim, dropout_rate)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = MLP(embedding_dim, hidden_dim, input_dim, dropout_rate)
        self.z = None
        self.z_q = None
    
    def forward(self, x):
        z = self.encoder(x)
        self.z = z
        z_q, _, _ = self.vq_layer(z)
        self.z_q = z_q
        x_recon = self.decoder(z_q)
        return x_recon, z, z_q
    
    def get_z(self):
        return self.z

    def get_z_q(self):
        return self.z_q

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def vqvae_loss(x_recon, x, z, z_q, lambda_weight=0.25):
    recon_loss = F.mse_loss(x_recon, x)
    quant_loss = F.mse_loss(z, z_q.detach())  # Detach z_q so gradients aren't computed for VQ layer
    return recon_loss + lambda_weight * quant_loss