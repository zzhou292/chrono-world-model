import torch
import numpy as np
import os
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

def load_latent_data(latent_path):
    """
    Load latent data and return the tensor along with metadata if available
    """
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent file not found: {latent_path}")
    
    # Load the latent tensor
    latent_tensor = torch.load(latent_path, map_location='cpu')  # Load to CPU first
    print(f"Loaded latent tensor with shape: {latent_tensor.shape}")
    print(f"Latent tensor dtype: {latent_tensor.dtype}")
    
    # Try to load metadata if it exists
    metadata_path = latent_path.replace('_latents_', '_latent_metadata_').replace('.pt', '.json')
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata}")
    
    return latent_tensor, metadata

# Define the STTransformer class
class STTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, latent_dim=16*60*80, seq_len=5):
        super(STTransformer, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim  
        
        # Linear embedding for latents (no spatial patching needed)
        self.embed = nn.Linear(latent_dim, embed_dim)
        
        # Positional encoding for temporal sequence
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder for prediction
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection back to latent space
        self.output_proj = nn.Linear(embed_dim, latent_dim)
        
    def forward(self, x, tgt=None):
        # x: (batch, seq_len, latent_dim) - already flattened latents
        batch_size = x.shape[0]
        x = self.embed(x) + self.pos_embed  # Embed and add pos
        
        memory = self.transformer_encoder(x)  # Encode sequence
        
        if tgt is not None:
            tgt = self.embed(tgt.unsqueeze(1)) + self.pos_embed[:, :1, :]  # Embed target
            output = self.transformer_decoder(tgt, memory)
        else:
            # Inference: Use zero or previous pred
            output = self.transformer_decoder(torch.zeros_like(memory[:, :1, :]), memory)
        
        output = self.output_proj(output).squeeze(1)  # (batch, latent_dim)
        return output

# 3) Define Dataset for Sliding Window Training
class LatentDataset(Dataset):
    def __init__(self, latent_tensor, seq_len=5, latent_dim=16*60*80):
        # latent_tensor should be [126, 16, 60, 80]
        self.data = latent_tensor.reshape(latent_tensor.shape[0], -1)  # Flatten each frame to (126, 76800)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        input_seq = self.data[idx:idx+self.seq_len]  # (seq_len, latent_dim)
        target = self.data[idx+self.seq_len]  # (latent_dim,)
        return input_seq, target

# Define the LSTMPredictor class
class LSTMPredictor(nn.Module):
    def __init__(self, latent_dim=16*60*80, hidden_dim=512, num_layers=4, seq_len=5):
        super(LSTMPredictor, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM expects input (batch, seq_len, latent_dim)
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, latent_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # Take the last time step
        output = self.output_proj(last_hidden)  # (batch, latent_dim)
        return output

# Main script starts here
if __name__ == "__main__":
    # 1) Load the pre-encoded latent data
    latent_file_path = "/home/jason/Desktop/chrono-world-model/PyChronobotics-main/experiment/train_data/test_20250913_222358_1/test_20250913_222358_1_latents_CV8x8x8.pt"
    
    print("Loading latent data...")
    latent_tensor, metadata = load_latent_data(latent_file_path)
    
    # 2) Prepare latent tensor for training
    print("\nPreparing latent tensor for training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Reshape latent_tensor: [1, 16, 126, 60, 80] -> [126, 16, 60, 80]
    latent_tensor = latent_tensor.squeeze(0).permute(1, 0, 2, 3).to(device)
    print(f"Reshaped latent tensor: {latent_tensor.shape}")


    # Model, optimizer, loss
    model = LSTMPredictor(latent_dim=16*60*80).to(device, dtype=torch.bfloat16)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Dataset and DataLoader
    dataset = LatentDataset(latent_tensor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training
    num_epochs = 300
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # inputs: (batch, seq_len, latent_dim), targets: (batch, latent_dim)
            optimizer.zero_grad()
            
            # Teacher forcing: Feed ground truth target as input to decoder
            outputs = model(inputs)  # outputs: (batch, latent_dim)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {epoch_loss / len(dataloader):.4f}")
    
    # 5) Autoregressive Inference: Build result_prediction using ground truth where possible
    with torch.no_grad():
        # Initialize result_prediction with ground truth frames 0-4
        result_prediction = [latent_tensor[i].reshape(-1) for i in range(5)]  # List of (76800,) tensors

        # Predict frames 5 to 125 using teacher forcing (always ground truth as input)
        for i in range(5, 126):
            # Use ground truth frames [i-5:i] as input
            current_input = torch.stack([latent_tensor[j].reshape(-1) for j in range(i-5, i)], dim=0).unsqueeze(0).to(device)  # (1, 5, 76800)
            
            # Predict the next frame
            pred_latent = model(current_input)  # (1, 76800)
            
            # Append the prediction to result_prediction
            result_prediction.append(pred_latent.squeeze(0))

        # Stack into full latent tensor: [126, 76800]
        full_predicted_latent = torch.stack(result_prediction, dim=0).to(device)
        print(f"Full predicted latent shape: {full_predicted_latent.shape}")  # (126, 76800)
        
        # Reshape to [1, 16, 126, 60, 80] for decoding
        full_predicted_latent_reshaped = full_predicted_latent.reshape(126, 16, 60, 80).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        print(f"Reshaped full latent for decoding: {full_predicted_latent_reshaped.shape}")  # Should be [1, 16, 126, 60, 80]
        
        # Initialize the tokenizer
        model_name = 'Cosmos-0.1-Tokenizer-CV8x8x8'
        encoder_ckpt = f"Cosmos-Tokenizer/pretrained_ckpts/{model_name}/encoder.jit"
        decoder_ckpt = f"Cosmos-Tokenizer/pretrained_ckpts/{model_name}/decoder.jit"
        tokenizer = CausalVideoTokenizer(
            checkpoint_enc=encoder_ckpt,
            checkpoint_dec=decoder_ckpt,
            device=device,
            dtype="bfloat16",
        )
        
        # Decode the full latent to video
        reconstructed_video = tokenizer.decode(full_predicted_latent_reshaped)  # Expected: [1, T, H, W, 3] or similar
        reconstructed_video = reconstructed_video[0]  # Remove batch dim: [3, 1001, 480, 640] or [T, H, W, 3]
        print(f"Reconstructed video shape: {reconstructed_video.shape}")  # Debug: Check actual shape

        # If shape is [C, T, H, W], permute to [T, H, W, C]
        if reconstructed_video.shape[0] == 3 and reconstructed_video.ndim == 4:
            reconstructed_video = reconstructed_video.permute(1, 2, 3, 0)  # [T, H, W, C]

        # Rescale from [0, 1] to [0, 255] before converting to uint8
        reconstructed_video = reconstructed_video * 255.0

        # Convert to NumPy uint8 for mediapy
        reconstructed_video = reconstructed_video.detach().cpu().to(torch.float32).numpy().clip(0, 255).astype(np.uint8)
        
        # Save the reconstructed video
        import mediapy as media
        output_video_path = "/home/jason/Desktop/chrono-world-model/PyChronobotics-main/experiment/train_script/autoregressive_reconstructed.mp4"
        media.write_video(output_video_path, reconstructed_video, fps=25)
        print(f"Autoregressive reconstructed video saved to: {output_video_path}")
        
        print(reconstructed_video.min(), reconstructed_video.max(), reconstructed_video.mean())
