# @title In this step, load the required checkpoints, and perform video reconstruction. {"run":"auto"}
import cv2
import numpy as np
import torch
import os
import glob

import importlib
import cosmos_tokenizer.video_lib
import mediapy as media

importlib.reload(cosmos_tokenizer.video_lib)
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

def create_video_from_images(image_dir, fps=25):
    """
    Create a video array from sensor images in the specified directory
    """
    # Get all jpg files and sort them numerically
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    # Updated sorting function to handle filenames like 'frame_980.jpg'
    def extract_frame_number(filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        # Handle different naming patterns
        if '_' in basename:
            # For files like 'frame_980.jpg', extract the number after underscore
            parts = basename.split('_')
            return int(parts[-1])  # Get the last part after underscore
        else:
            # For files like '980.jpg', just convert to ints
            return int(basename)
    
    image_files.sort(key=extract_frame_number)
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    if len(image_files) == 0:
        raise ValueError(f"No .jpg files found in {image_dir}")
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")
    
    height, width, channels = first_img.shape
    print(f"Image dimensions: {width}x{height}x{channels}")
    
    # Create video array (T x H x W x 3)
    video_frames = []
    
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping...")
            continue
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video_frames.append(img_rgb)
    
    # Convert to numpy array
    video_array = np.array(video_frames)
    print(f"Created video array with shape: {video_array.shape}")
    
    return video_array

# 1) Specify the model name, and the paths to the encoder/decoder checkpoints.
model_name = 'Cosmos-0.1-Tokenizer-CV8x8x8' # @param ["Cosmos-0.1-Tokenizer-CV4x8x8", "Cosmos-0.1-Tokenizer-CV8x8x8", "Cosmos-0.1-Tokenizer-CV8x16x16", "Cosmos-0.1-Tokenizer-DV4x8x8", "Cosmos-0.1-Tokenizer-DV8x8x8", "Cosmos-0.1-Tokenizer-DV8x16x16", "Cosmos-1.0-Tokenizer-CV8x8x8", "Cosmos-1.0-Tokenizer-DV8x16x16"]
temporal_window = 49 # @param {type:"slider", min:1, max:121, step:8}

encoder_ckpt = f"Cosmos-Tokenizer/pretrained_ckpts/{model_name}/encoder.jit"
decoder_ckpt = f"Cosmos-Tokenizer/pretrained_ckpts/{model_name}/decoder.jit"

# 2) Load sensor images from the specified directory
sensor_img_dir = "/home/jason/Desktop/chrono-world-model/PyChronobotics-main/experiment/train_data/test_20250913_222358_1/sensor_img"

# 3) Create video from sensor images (shape = T x H x W x 3 in RGB)
input_video = create_video_from_images(sensor_img_dir, fps=25)
assert input_video.ndim == 4 and input_video.shape[-1] == 3, "Frames must have shape T x H x W x 3"

# Save the created video for reference
output_video_path = os.path.join(os.path.dirname(sensor_img_dir), "sensor_video.mp4")
media.write_video(output_video_path, input_video, fps=25)
print(f"Original sensor video saved to: {output_video_path}")

# 4) Expand dimensions to B x T x H x W x C, since the CausalVideoTokenizer expects a batch dimension
#    in the input. (Batch size = 1 in this example.)
batched_input_video = np.expand_dims(input_video, axis=0)

# Convert to tensor and rearrange dimensions from [B, T, H, W, C] to [B, C, T, H, W]
batched_input_tensor = torch.from_numpy(np.array(batched_input_video)).float()
batched_input_tensor = batched_input_tensor / 255.0  # Normalize to [0, 1]

# Rearrange dimensions: [B, T, H, W, C] -> [B, C, T, H, W]
batched_input_tensor = batched_input_tensor.permute(0, 4, 1, 2, 3)

# Move to GPU and convert to bfloat16
batched_input_tensor = batched_input_tensor.to(device="cuda", dtype=torch.bfloat16)

print(f"Input tensor shape: {batched_input_tensor.shape}")  # Should be [1, 3, T, H, W]
print(f"Input tensor dtype: {batched_input_tensor.dtype}")

# 5) Create the CausalVideoTokenizer instance with the encoder & decoder.
tokenizer = CausalVideoTokenizer(
    checkpoint_enc=encoder_ckpt,
    checkpoint_dec=decoder_ckpt,
    device="cuda",
    dtype="bfloat16",
)

# 6) Use the tokenizer to autoencode (encode & decode) the video.
#    The output is a NumPy array with shape = B x T x H x W x C, range [0..255].
batched_output_video = tokenizer(batched_input_video,
                                 temporal_window=temporal_window)

# 7) Extract the single video from the batch (index 0).
output_video = batched_output_video[0]

# 8) Save the reconstructed video to disk.
sensor_dir = os.path.dirname(sensor_img_dir)
output_filepath = os.path.join(sensor_dir, f"sensor_reconstructed_{model_name.split('-')[-1]}.mp4")
media.write_video(output_filepath, output_video, fps=25)
print("Input video created from sensor images")
print("Reconstruction saved:\t", output_filepath)

# 9) Visualization of the input video (left) and the reconstruction (right).
media.show_videos([input_video, output_video], ["Original Sensor Video", "Reconstructed Video"], height=720)


# Now use the properly formatted tensor for encoding
(latent, ) = tokenizer.encode(batched_input_tensor)
print(f"Latent shape: {latent.shape}")

# 10) Save the latents for later use (THIS IS THE IMPORTANT .pt FILE)
sensor_dir = os.path.dirname(sensor_img_dir)
dataset_name = os.path.basename(sensor_dir)  # e.g., "test_20250913_222358_1"
latent_filepath = os.path.join(sensor_dir, f"{dataset_name}_latents_{model_name.split('-')[-1]}.pt")
torch.save(latent, latent_filepath)
print(f"Latents saved to: {latent_filepath}")