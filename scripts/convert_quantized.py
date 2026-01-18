import os
import sys
import torch
import hydra
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.architectures import JointModel

@hydra.main(version_base=None, config_path="../configs", config_name="joint_training")
def main(cfg):
    device = torch.device('cpu') #Quantization for CPU
    print("Initializing full model...")
    model = JointModel(cfg).to(device)
    checkpoint_path = cfg.training.resume or os.path.join(cfg.training.checkpoint_dir, 'checkpoint_epoch_48.pth')
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    # MEMORY CLEANUP: Crucial for OOM
    del ckpt
    del state_dict
    import gc
    gc.collect()
    
    model.eval()
    print("\nQuantizing model (Dynamic Quantization for Linear and Convolution Layers)...")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear,torch.nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    # 3. Compare Size
    def print_size_of_model(model, label=""):
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p")
        print(f"Model: {label:<15} Size: {size/1e6:.2f} MB")
        os.remove('temp.p')

    print_size_of_model(model, "Original")
    print_size_of_model(quantized_model, "Quantized")
    
    # 4. Save
    output_path = checkpoint_path.replace(".pth", "_quantized.pth")
    print(f"\nSaving quantized model to: {output_path}")
    torch.save(quantized_model.state_dict(), output_path)

if __name__ == '__main__':
    main()
