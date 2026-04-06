import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def get_latest_snapshot(model_path):
    """获取 huggingface cache 中模型的最新 snapshot 路径"""
    snapshots_dir = Path(model_path) / "snapshots"
    if not snapshots_dir.exists():
        raise ValueError(f"Snapshots directory not found: {snapshots_dir}")
    
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        raise ValueError(f"No snapshots found in {snapshots_dir}")
    
    latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
    return str(latest_snapshot)


class Qwen3VLAttentionExtractor:
    """通过 output_attentions=True 提取 attention weights"""
    
    def __init__(self, model, processor, discard_ratio=0.9):
        self.model = model
        self.processor = processor
        self.discard_ratio = discard_ratio
        
        # 获取特殊 token IDs
        self.image_token_id = getattr(model.config, 'image_token_id', 151655)
        self.vision_start_token_id = getattr(model.config, 'vision_start_id', 151652)
        self.vision_end_token_id = getattr(model.config, 'vision_end_id', 151653)
        
        # 获取 merge_size
        if hasattr(model.config, 'vision_config'):
            self.merge_size = getattr(model.config.vision_config, 'spatial_merge_size', 2)
        else:
            self.merge_size = 2
        
        print("="*60)
        print("Model Configuration:")
        print(f"  Image token ID: {self.image_token_id}")
        print(f"  Vision start token ID: {self.vision_start_token_id}")
        print(f"  Vision end token ID: {self.vision_end_token_id}")
        print(f"  Merge size: {self.merge_size}")
        print("="*60)
    
    def _find_image_token_indices(self, input_ids):
        """找到图像 token 的精确位置"""
        input_ids = input_ids.squeeze().cpu().numpy()
        
        vision_start_positions = np.where(input_ids == self.vision_start_token_id)[0]
        vision_end_positions = np.where(input_ids == self.vision_end_token_id)[0]
        
        print(f"\nFinding image tokens:")
        print(f"  Sequence length: {len(input_ids)}")
        print(f"  Vision start positions: {vision_start_positions}")
        print(f"  Vision end positions: {vision_end_positions}")
        
        if len(vision_start_positions) == 0 or len(vision_end_positions) == 0:
            image_positions = np.where(input_ids == self.image_token_id)[0]
            print(f"  Fallback - found {len(image_positions)} image tokens")
            return image_positions if len(image_positions) > 0 else None
        
        start_pos = vision_start_positions[0]
        end_pos = vision_end_positions[0]
        image_token_indices = np.arange(start_pos + 1, end_pos)
        
        print(f"  Image token range: [{start_pos + 1}, {end_pos - 1}]")
        print(f"  Number of image tokens: {len(image_token_indices)}")
        
        return image_token_indices
    
    def _get_grid_size(self, inputs, num_tokens):
        """根据token数量推断grid尺寸"""
        if 'image_grid_thw' in inputs:
            grid = inputs['image_grid_thw'][0].cpu()
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            h_merged = h // self.merge_size
            w_merged = w // self.merge_size
            expected = t * h_merged * w_merged
            
            print(f"\n  Grid from image_grid_thw: T={t}, H={h}, W={w}")
            print(f"  After merge: H={h_merged}, W={w_merged}, expected={expected}")
            
            if expected == num_tokens:
                return h_merged, w_merged
        
        # Fallback: 找最接近的因数分解
        sqrt_n = int(np.sqrt(num_tokens))
        for h in range(sqrt_n, 0, -1):
            if num_tokens % h == 0:
                w = num_tokens // h
                print(f"  Inferred grid: {h}x{w}")
                return h, w
        return sqrt_n, sqrt_n
    
    def __call__(self, image, prompt="Describe this image."):
        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            print(f"\nInput shapes:")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}")
            
            image_token_indices = self._find_image_token_indices(inputs['input_ids'])
            
            # 生成回复
            print("\n" + "="*60)
            print("Generating response...")
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            input_len = inputs['input_ids'].shape[1]
            response = self.processor.batch_decode(
                generated_ids[:, input_len:], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            print(f"Prompt: {prompt}")
            print(f"Response: {response[:200]}...")
            print("="*60)
            
            # 获取注意力
            outputs = self.model(**inputs, output_attentions=True)
            
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                print(f"\nAttention layers: {len(outputs.attentions)}")
                print(f"Attention shape: {outputs.attentions[0].shape}")
                attentions = [attn.cpu().float() for attn in outputs.attentions]
                
                return self._compute_all_methods(attentions, inputs, image_token_indices), response
            else:
                print("No attention weights returned!")
                return {}, response
    
    def _compute_all_methods(self, attentions, inputs, image_token_indices):
        """尝试多种不同的attention计算方法"""
        results = {}
        
        if image_token_indices is None or len(image_token_indices) == 0:
            print("No image tokens found!")
            return results
        
        num_tokens = len(image_token_indices)
        h, w = self._get_grid_size(inputs, num_tokens)
        
        seq_len = attentions[0].size(-1)
        last_image_idx = image_token_indices[-1]
        first_image_idx = image_token_indices[0]
        
        # 找到文本token的范围
        text_start_idx = last_image_idx + 1
        text_end_idx = seq_len
        
        print(f"\n" + "="*60)
        print("Computing attention maps with MULTIPLE methods...")
        print(f"  Image tokens: [{first_image_idx}, {last_image_idx}] ({num_tokens} tokens)")
        print(f"  Text tokens: [{text_start_idx}, {text_end_idx - 1}] ({text_end_idx - text_start_idx} tokens)")
        print(f"  Grid size: {h}x{w}")
        print("="*60)
        
        # 尝试不同层的组合
        layer_configs = {
            'last_1': slice(-1, None),
            'last_4': slice(-4, None),
            'last_8': slice(-8, None),
            'first_4': slice(0, 4),
            'middle_4': slice(len(attentions)//2 - 2, len(attentions)//2 + 2),
            'all': slice(None),
        }
        
        for layer_name, layer_slice in layer_configs.items():
            layers = attentions[layer_slice]
            
            # 堆叠并平均
            stacked = torch.stack(layers)  # [num_layers, batch, heads, seq, seq]
            avg_attn = stacked.mean(dim=0).mean(dim=1)[0].numpy()  # [seq, seq]
            
            print(f"\n--- Layers: {layer_name} ({len(layers)} layers) ---")
            
            # ============ 方法1: 文本 -> 图像 ============
            if text_start_idx < seq_len:
                text_to_image = avg_attn[text_start_idx:, image_token_indices]
                # 对所有文本token求和/平均
                attn_map = text_to_image.sum(axis=0)
                results[f'{layer_name}_text_to_img_sum'] = self._reshape_and_normalize(attn_map, h, w)
                
                attn_map = text_to_image.mean(axis=0)
                results[f'{layer_name}_text_to_img_mean'] = self._reshape_and_normalize(attn_map, h, w)
                
                # 只用最后一个文本token
                attn_map = avg_attn[-1, image_token_indices]
                results[f'{layer_name}_last_token_to_img'] = self._reshape_and_normalize(attn_map, h, w)
                
                print(f"  text->img sum: min={text_to_image.sum(axis=0).min():.4f}, max={text_to_image.sum(axis=0).max():.4f}")
            
            # ============ 方法2: 图像 -> 文本 (反向) ============
            if text_start_idx < seq_len:
                image_to_text = avg_attn[image_token_indices, :][:, text_start_idx:]
                attn_map = image_to_text.sum(axis=1)
                results[f'{layer_name}_img_to_text_sum'] = self._reshape_and_normalize(attn_map, h, w)
                
                attn_map = image_to_text.mean(axis=1)
                results[f'{layer_name}_img_to_text_mean'] = self._reshape_and_normalize(attn_map, h, w)
                
                print(f"  img->text sum: min={image_to_text.sum(axis=1).min():.4f}, max={image_to_text.sum(axis=1).max():.4f}")
            
            # ============ 方法3: 图像自注意力 ============
            image_self_attn = avg_attn[image_token_indices, :][:, image_token_indices]
            
            # 每个图像token收到的总注意力 (column sum)
            attn_map = image_self_attn.sum(axis=0)
            results[f'{layer_name}_img_self_col_sum'] = self._reshape_and_normalize(attn_map, h, w)
            
            # 每个图像token发出的总注意力 (row sum)
            attn_map = image_self_attn.sum(axis=1)
            results[f'{layer_name}_img_self_row_sum'] = self._reshape_and_normalize(attn_map, h, w)
            
            # 对角线 (自身注意力)
            attn_map = np.diag(image_self_attn)
            results[f'{layer_name}_img_self_diag'] = self._reshape_and_normalize(attn_map, h, w)
            
            print(f"  img self col_sum: min={image_self_attn.sum(axis=0).min():.4f}, max={image_self_attn.sum(axis=0).max():.4f}")
            print(f"  img self row_sum: min={image_self_attn.sum(axis=1).min():.4f}, max={image_self_attn.sum(axis=1).max():.4f}")
            
            # ============ 方法4: CLS-like (第一个图像token对其他图像token) ============
            attn_map = image_self_attn[0, :]  # 第一个图像token看其他
            results[f'{layer_name}_first_img_to_others'] = self._reshape_and_normalize(attn_map, h, w)
            
            attn_map = image_self_attn[:, 0]  # 其他图像token看第一个
            results[f'{layer_name}_others_to_first_img'] = self._reshape_and_normalize(attn_map, h, w)
            
            # ============ 方法5: 全局attention (所有token对图像) ============
            all_to_image = avg_attn[:, image_token_indices]
            attn_map = all_to_image.sum(axis=0)
            results[f'{layer_name}_all_to_img'] = self._reshape_and_normalize(attn_map, h, w)
            
            # ============ 方法6: 图像对所有 ============
            image_to_all = avg_attn[image_token_indices, :]
            attn_map = image_to_all.sum(axis=1)
            results[f'{layer_name}_img_to_all'] = self._reshape_and_normalize(attn_map, h, w)
            
            # ============ 方法7: Attention Rollout ============
            rollout = self._attention_rollout(layers, image_token_indices, h, w)
            if rollout is not None:
                results[f'{layer_name}_rollout'] = rollout
        
        # ============ 方法8: 使用max而非mean融合heads ============
        stacked = torch.stack(attentions[-4:])
        max_attn = stacked.mean(dim=0).max(dim=1)[0][0].numpy()
        
        if text_start_idx < seq_len:
            attn_map = max_attn[text_start_idx:, image_token_indices].mean(axis=0)
            results['last4_maxhead_text_to_img'] = self._reshape_and_normalize(attn_map, h, w)
        
        image_self = max_attn[image_token_indices, :][:, image_token_indices]
        results['last4_maxhead_img_self_col'] = self._reshape_and_normalize(image_self.sum(axis=0), h, w)
        
        # ============ 方法9: 逐层分析 ============
        for layer_idx in [0, len(attentions)//4, len(attentions)//2, 3*len(attentions)//4, -1]:
            attn = attentions[layer_idx].mean(dim=1)[0].numpy()
            image_self = attn[image_token_indices, :][:, image_token_indices]
            results[f'layer{layer_idx}_img_self_col'] = self._reshape_and_normalize(image_self.sum(axis=0), h, w)
            
            if text_start_idx < seq_len:
                results[f'layer{layer_idx}_text_to_img'] = self._reshape_and_normalize(
                    attn[text_start_idx:, image_token_indices].mean(axis=0), h, w
                )
        
        print(f"\n  Total methods computed: {len(results)}")
        
        return results
    
    def _attention_rollout(self, attentions, image_token_indices, h, w):
        """Attention rollout"""
        try:
            result = None
            for attn in attentions:
                attn_heads_fused = attn.mean(dim=1)[0].numpy()
                
                # 添加残差连接
                attn_heads_fused = 0.5 * attn_heads_fused + 0.5 * np.eye(attn_heads_fused.shape[0])
                
                # 归一化
                attn_heads_fused = attn_heads_fused / attn_heads_fused.sum(axis=-1, keepdims=True)
                
                if result is None:
                    result = attn_heads_fused
                else:
                    result = result @ attn_heads_fused
            
            # 提取图像token的注意力
            image_attn = result[:, image_token_indices].sum(axis=0)
            return self._reshape_and_normalize(image_attn, h, w)
        except Exception as e:
            print(f"  Rollout failed: {e}")
            return None
    
    def _reshape_and_normalize(self, attn_map, h, w):
        """Reshape并归一化"""
        try:
            if len(attn_map) != h * w:
                # 尝试调整
                actual_len = len(attn_map)
                sqrt_n = int(np.sqrt(actual_len))
                if sqrt_n * sqrt_n == actual_len:
                    h, w = sqrt_n, sqrt_n
                else:
                    for new_h in range(sqrt_n, 0, -1):
                        if actual_len % new_h == 0:
                            h, w = new_h, actual_len // new_h
                            break
            
            reshaped = attn_map.reshape(h, w)
            
            if reshaped.max() > reshaped.min():
                normalized = (reshaped - reshaped.min()) / (reshaped.max() - reshaped.min())
            else:
                normalized = np.ones_like(reshaped) * 0.5
            
            return normalized
        except Exception as e:
            print(f"  Reshape failed: {e}")
            return np.ones((h, w)) * 0.5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--image_path', type=str, default='./examples/both.png')
    parser.add_argument('--prompt', type=str, default='What breed is this cat? What is it doing?')
    parser.add_argument('--discard_ratio', type=float, default=0.9)
    parser.add_argument('--model_cache_path', type=str, 
                        default=os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Thinking'))
    parser.add_argument('--output_dir', type=str, default='outputs/vit_image_explain')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    print(f"Using {'GPU' if args.use_cuda else 'CPU'}")
    return args


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def create_grid_comparison(np_img, masks_dict, cols=6):
    """创建所有方法的对比网格图"""
    h, w = np_img.shape[:2]
    
    # 过滤掉None
    valid_masks = {k: v for k, v in masks_dict.items() if v is not None}
    names = list(valid_masks.keys())
    n = len(names) + 1  # +1 for original
    
    rows = (n + cols - 1) // cols
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    # 原图
    grid[0:h, 0:w] = np_img
    cv2.putText(grid, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    for idx, name in enumerate(names):
        row = (idx + 1) // cols
        col = (idx + 1) % cols
        
        mask = valid_masks[name]
        mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
        overlay = show_mask_on_image(np_img, mask_resized)
        
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = overlay
        
        # 缩短名称以适应
        short_name = name.replace('last_', 'L').replace('first_', 'F').replace('middle_', 'M')
        short_name = short_name.replace('_to_', '>').replace('_sum', 'Σ').replace('_mean', 'μ')
        short_name = short_name.replace('text', 'T').replace('img', 'I').replace('self', 'S')
        short_name = short_name.replace('layer', 'Ly').replace('rollout', 'roll')
        
        cv2.putText(grid, short_name[:20], (col*w + 5, row*h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(grid, short_name[:20], (col*w + 5, row*h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    return grid


if __name__ == '__main__':
    args = get_args()
    
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = get_latest_snapshot(args.model_cache_path)
    print(f"Loading model from: {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if args.use_cuda else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    img = Image.open(args.image_path).convert('RGB')
    print(f"Original image size: {img.size}")
    img_display = img.resize((448, 448))

    attention_extractor = Qwen3VLAttentionExtractor(model, processor, discard_ratio=args.discard_ratio)
    masks_dict, response = attention_extractor(img, prompt=args.prompt)
    
    np_img = np.array(img_display)[:, :, ::-1]
    
    # 保存原图
    cv2.imwrite(str(output_dir / "input.png"), np_img)
    
    # 保存所有方法的单独图片
    methods_dir = output_dir / "methods"
    methods_dir.mkdir(exist_ok=True)
    
    for name, mask in masks_dict.items():
        if mask is not None:
            mask_resized = cv2.resize(mask.astype(np.float32), (np_img.shape[1], np_img.shape[0]))
            overlay = show_mask_on_image(np_img, mask_resized)
            cv2.imwrite(str(methods_dir / f"{name}.png"), overlay)
    
    # 保存大网格对比图
    grid = create_grid_comparison(np_img, masks_dict, cols=6)
    cv2.imwrite(str(output_dir / "all_methods_comparison.png"), grid)
    
    # 保存response
    with open(str(output_dir / "response.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {args.prompt}\n\nResponse:\n{response}\n")
        f.write(f"\n\nMethods computed: {len(masks_dict)}\n")
        for name in masks_dict.keys():
            f.write(f"  - {name}\n")
    
    print("\n" + "="*60)
    print(f"Saved {len(masks_dict)} attention maps to: {output_dir}")
    print("="*60)
    print("\nCHECK 'all_methods_comparison.png' to find which method works!")
    print("Look for: high attention (red/yellow) on the airplane in the center")
    print("\nMethod naming convention:")
    print("  L4 = last 4 layers, F4 = first 4 layers")
    print("  T>I = text to image, I>T = image to text")
    print("  I_S = image self-attention")
    print("  col = column sum (how much attention each token receives)")
    print("  row = row sum (how much attention each token gives)")
    print("="*60)