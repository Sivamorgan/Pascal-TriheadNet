
import os
import sys
import torch
import numpy as np
import cv2
import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.infer_single import InferencePipeline

@hydra.main(version_base=None, config_path="../configs", config_name="joint_training")
def main(cfg: DictConfig):
    # Usage: python scripts/infer_video.py +video_path="input.mp4"
    
    video_path = cfg.get('video_path', None)
    if not video_path:
        print("Error: Please provide +video_path='/path/to/video.mp4'")
        return

    checkpoint = cfg.training.resume or os.path.join(cfg.training.checkpoint_dir, 'checkpoint_epoch_48.pth')
    
    # Initialize Pipeline
    print("Initializing Model...")
    pipeline = InferencePipeline(checkpoint, cfg, device=cfg.training.device)
    
    # Video Capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    # Get Video Properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing Video: {width}x{height} @ {int(fps)} FPS ({frame_count} frames)")
    
    # Output Writer
    out_name = f"vis_{Path(video_path).stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v or h264
    
    out = None # Initialize later when we know exact output size
    
    try:
        for _ in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR (OpenCV) to RGB (PIL)
            # We can pass numpy array directly to predict, but it expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict
            # Returns (H, W, 3) BGR numpy array ready for saving/vis
            vis_result = pipeline.predict(frame_rgb)
            
            # Initialize VideoWriter once we know the output dimensions
            if out is None:
                h, w = vis_result.shape[:2]
                print(f"Initializing output video: {w}x{h}.")
                out = cv2.VideoWriter(out_name, fourcc, int(fps), (w, h))

            # Write key frame
            out.write(vis_result)
            
    finally:
        cap.release()
        out.release()
        print(f"\nSaved video result to {out_name}")

if __name__ == '__main__':
    main()
