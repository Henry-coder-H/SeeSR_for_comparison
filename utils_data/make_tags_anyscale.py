# tag_div2k_with_ram.py
import os, sys, glob, argparse
sys.path.append(os.getcwd())
import torch
from PIL import Image
from torchvision import transforms
from ram.models.ram import ram
from ram import inference_ram as inference

ram_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, required=True,
                    help="你的 SeeSR 训练对根目录，比如/data4/huangsiyu/SeeSR_baseline/preset/datasets/train_datasets/training_for_seesr")
parser.add_argument("--start_gpu", type=int, default=0)
parser.add_argument("--all_gpu", type=int, default=1)
args = parser.parse_args()

gt_path  = os.path.join(args.root_path, 'gt')
tag_path = os.path.join(args.root_path, 'tag')
os.makedirs(tag_path, exist_ok=True)

gt_lists = sorted(glob.glob(os.path.join(gt_path, '*.png')))
print(f'There are {len(gt_lists)} GT imgs')

model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
            image_size=384, vit='swin_l').eval().to('cuda')

start = args.start_gpu * len(gt_lists)//args.all_gpu
end   = (args.start_gpu+1) * len(gt_lists)//args.all_gpu
print(f'===== process [{start} , {end}) =====')

with torch.no_grad():
    for i, gt_file in enumerate(gt_lists[start:end], start=start):
        basename = os.path.splitext(os.path.basename(gt_file))[0]  # 0000001
        img = ram_transforms(Image.open(gt_file).convert("RGB")).unsqueeze(0).cuda()
        captions = inference(img, model)
        prompt = f"{captions[0]},"
        save_txt = os.path.join(tag_path, f"{basename}.txt")  # 修正了这里
        with open(save_txt, "w") as f:
            f.write(prompt)
        if (i - start) % 100 == 0:
            print(f"[{i}/{end}] wrote {save_txt}")
