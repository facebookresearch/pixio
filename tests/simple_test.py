# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from PIL import Image


def test_inference():
    import sys
    sys.path.append('./pixio')

    from PIL import Image
    from torchvision import transforms

    from pixio import pixio_base, pixio_large, pixio_huge, pixio_1b, pixio_5b

    for model_cls, ckpt_path, num_layers in [
        (pixio_base, '/checkpoint/data/huxu/code_pub/pixio_base_latest_release.pth', 12),
        (pixio_large, '/checkpoint/data/huxu/code_pub/pixio_large_latest_release.pth', 24),
        (pixio_huge, '/checkpoint/data/huxu/code_pub/pixio_huge_latest_release.pth', 32),
        (pixio_1b, '/checkpoint/data/huxu/code_pub/pixio_1b_latest_release.pth', 48),
        (pixio_5b, '/checkpoint/data/huxu/code_pub/pixio_5b_latest_release.pth', 48),
    ]:
        model = model_cls(pretrained=ckpt_path)
        
        # you can try larger resolution, but ensure both sides are divisible by 16
        transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3), # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        img = Image.open('assets/pixio.png').convert('RGB')
        img = transform(img)
        
        features = model(img.unsqueeze(0))
        assert len(features) == num_layers
        numel = {n: p.numel() for n, p in model.named_parameters()}
        print(sum(numel.values()))


def test_hf():
    from transformers import AutoModel
    model = AutoModel.from_pretrained('facebook/pixio-vit1b16')


if __name__ == "__main__":
    # test_inference()
    test_hf()
