import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import warnings
from tqdm import tqdm  # For progress—pip if missing, but it's tiny

warnings.filterwarnings("ignore")

class ImageDataset(Dataset):
    """Simple loader for batch GPU processing."""
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform
        self.images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.input_dir, self.images[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.images[idx]
        except Exception as e:  # Skip corrupt files
            print(f"Skipping {img_path}: {e}")
            return None, None  # Will filter these out later

def process_batch(batch, device):
    """GPU batch "process"—now just stacks 'em (resize done upstream)."""
    # Filter out None (from errors)
    valid_batch = [img for img in batch if img is not None]
    if not valid_batch:
        return torch.empty(0)  # Empty batch
    
    batch_tensor = torch.stack(valid_batch).to(device)
    return batch_tensor  # Uniform shape, so stack flies

def blaze_images(args):
    """Main blazer—batch process dir to output."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # NEW: Resize/crop in transform for uniform tensors (fixes collate crash)
    if args.crop:
        # Preserve aspect: Scale smaller side to resize, then center-crop square
        transform = transforms.Compose([
            transforms.Resize(args.resize),  # int = smaller edge
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),
        ])
        print(f"Blazing on {device} (RTX 3050 vibes). Batch size: {args.batch_size}, Mode: Crop (aspect-safe)")
    else:
        # Stretch to square (original vibe)
        transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.ToTensor(),
        ])
        print(f"Blazing on {device} (RTX 3050 vibes). Batch size: {args.batch_size}, Mode: Stretch")
    
    dataset = ImageDataset(args.input_dir, transform)
    if len(dataset) == 0:
        print("No images found—check input_dir.")
        return
    
    # Filter valid items post-load (cheap since it's lazy; skips corrupts)
    valid_indices = [i for i in range(len(dataset)) if dataset[i][0] is not None]
    dataset.images = [dataset.images[i] for i in valid_indices]  # Update list
    print(f"Loaded {len(dataset)} valid images.")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    processed = 0
    pbar = tqdm(total=len(dataset), desc="Blazing images")
    
    for batch_tensors, filenames in dataloader:
        # Filter None again (paranoia)
        valid_tensors = [t for t in batch_tensors if t is not None]
        valid_fnames = [f for f in filenames if f is not None]
        
        if not valid_tensors:
            continue
            
        with torch.no_grad():
            output_batch = process_batch(valid_tensors, device)
        
        # Back to PIL, save as PNG (lossless, optimized)
        for i, (out_tensor, fname) in enumerate(zip(output_batch.cpu(), valid_fnames)):  # .cpu() here
            out_img = transforms.ToPILImage()(out_tensor)
            base_name = fname.rsplit('.', 1)[0]
            out_path = os.path.join(args.output_dir, f"blazed_{base_name}.png")
            out_img.save(out_path, 'PNG', optimize=True)
            processed += 1
            pbar.update(1)
    
    pbar.close()
    elapsed = time.time() - start_time
    print(f"Blazed {processed} images in {elapsed:.2f}s ({processed/elapsed:.1f} imgs/sec). "
          f"Output: {args.output_dir}. Freelance this bad boy—easy 100€/batch!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Batch Image Blazer—Freelance Dollar Machine")
    parser.add_argument('--input_dir', required=True, help="Folder of input images")
    parser.add_argument('--output_dir', default='./blazed', help="Output folder (default: ./blazed)")
    parser.add_argument('--resize', type=int, default=1024, help="Target square size (default: 1024)")
    parser.add_argument('--batch_size', type=int, default=32, help="GPU batch size (default: 32; tune for 3050)")
    parser.add_argument('--num_workers', type=int, default=2, help="DataLoader workers (default: 2; 0 for debug)")
    parser.add_argument('--crop', action='store_true', help="Center-crop after fit-resize (preserves aspect; default: stretch)")
    args = parser.parse_args()
    blaze_images(args)
