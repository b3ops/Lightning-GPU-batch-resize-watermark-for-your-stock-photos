import os
import argparse
from PIL import Image, ImageDraw, ImageFont
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

def add_watermark(img, text, font_size):
    """Add bottom-right watermark with PIL—scaled font, WSL-friendly."""
    draw = ImageDraw.Draw(img)
    font_path = None
    # WSL: Try Windows fonts first (super common)
    if os.path.exists('/mnt/c/Windows/Fonts/arial.ttf'):
        font_path = '/mnt/c/Windows/Fonts/arial.ttf'
    # Linux fallbacks
    elif os.path.exists('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    elif os.path.exists('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
        font_path = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
    
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"Loaded font: {os.path.basename(font_path)} ({font_size}px)")
        except Exception as e:
            print(f"Font load failed ({e}); using default.")
            font = ImageFont.load_default()
    else:
        print("No TTF font found—install with: sudo apt update && sudo apt install fonts-dejavu-core && sudo fc-cache -fv")
        print("Or for Windows fonts in WSL: Add symlink or config /etc/fonts/conf.d/ for /mnt/c/Windows/Fonts")
        font = ImageFont.load_default()
    
    # Bottom-right: Margin scaled to 1% of size
    w, h = img.size
    margin = int(min(w, h) * 0.01)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(0, w - text_w - margin)  # Clamp to avoid overflow
    y = max(0, h - text_h - margin)
    draw.text((x, y), text, fill='black', font=font)
    return img

def blaze_images(args):
    """Main blazer—batch process dir to output."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4K tuning: Override resize & batch
    if args.four_k:
        args.resize = 4096
        args.batch_size = 4  # VRAM-safe for 3050
        print(f"4K Mode engaged: {args.resize}px squares, batch={args.batch_size} (OOM-proof).")
    
    # Limit to max_images for testing
    if args.max_images:
        print(f"Testing mode: Limiting to {args.max_images} images.")
    
    # Resize/crop in transform for uniform tensors (fixes collate crash)
    if args.crop:
        # Preserve aspect: Scale smaller side to resize, then center-crop square
        transform = transforms.Compose([
            transforms.Resize(args.resize),  # int = smaller edge
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),
        ])
        mode = "Crop (aspect-safe)"
    else:
        # Stretch to square (original vibe)
        transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.ToTensor(),
        ])
        mode = "Stretch"
    
    print(f"Blazing on {device} (RTX 3050 vibes). Batch size: {args.batch_size}, Mode: {mode}")
    if args.watermark:
        font_size = max(48, int(args.resize / 21))  # ~2% scale: 48@1k, ~195@4k
        print(f"Watermark: '{args.watermark_text}' (bottom-right, font={font_size}px)")
    
    dataset = ImageDataset(args.input_dir, transform)
    if len(dataset) == 0:
        print("No images found—check input_dir.")
        return
    
    # Filter valid items post-load (cheap since it's lazy; skips corrupts)
    valid_indices = [i for i in range(len(dataset)) if dataset[i][0] is not None]
    dataset.images = [dataset.images[i] for i in valid_indices]  # Update list
    
    # Limit for testing
    if args.max_images and len(dataset) > args.max_images:
        dataset.images = dataset.images[:args.max_images]
        print(f"Limited to first {len(dataset)} images for testing.")
    
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
            if args.watermark:
                out_img = add_watermark(out_img, args.watermark_text, font_size if 'font_size' in locals() else 48)
            base_name = fname.rsplit('.', 1)[0]
            out_path = os.path.join(args.output_dir, f"blazed_{base_name}.png")
            out_img.save(out_path, 'PNG', optimize=True)
            processed += 1
            pbar.update(1)
    
    pbar.close()
    elapsed = time.time() - start_time
    imgs_sec = processed / elapsed if elapsed > 0 else 0
    print(f"Blazed {processed} images in {elapsed:.2f}s ({imgs_sec:.1f} imgs/sec). "
          f"Output: {args.output_dir}. Freelance this bad boy—easy 100€/batch!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Batch Image Blazer—Freelance Dollar Machine")
    parser.add_argument('--input_dir', required=True, help="Folder of input images")
    parser.add_argument('--output_dir', default='./blazed', help="Output folder (default: ./blazed)")
    parser.add_argument('--resize', type=int, default=1024, help="Target square size (default: 1024)")
    parser.add_argument('--batch_size', type=int, default=32, help="GPU batch size (default: 32; tune for 3050)")
    parser.add_argument('--num_workers', type=int, default=2, help="DataLoader workers (default: 2; 0 for debug)")
    parser.add_argument('--crop', action='store_true', help="Center-crop after fit-resize (preserves aspect; default: stretch)")
    parser.add_argument('--4k', dest='four_k', action='store_true', help="4K tune: resize=4096, batch=4 (VRAM-safe)")
    parser.add_argument('--watermark', action='store_true', help="Add bottom-right text watermark")
    parser.add_argument('--watermark_text', default="Blazed by Grok", help="Watermark text (default: 'Blazed by Grok')")
    parser.add_argument('--max_images', type=int, help="Max images to process (for testing)")
    args = parser.parse_args()
    blaze_images(args)
