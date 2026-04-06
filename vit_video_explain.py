import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
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


def extract_frames_from_video(video_path, max_frames=16, target_fps=None):
    """从视频中提取帧"""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {video_fps:.2f} fps, {duration:.2f}s")
    
    if target_fps is not None:
        frame_interval = max(1, int(video_fps / target_fps))
    else:
        frame_interval = max(1, total_frames // max_frames)
    
    frames = []
    frame_indices = []
    frame_times = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0 and len(frames) < max_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_indices.append(frame_idx)
            frame_times.append(frame_idx / video_fps if video_fps > 0 else frame_idx)
        
        frame_idx += 1
    
    cap.release()
    
    print(f"Extracted {len(frames)} frames at indices: {frame_indices}")
    print(f"Frame times (s): {[f'{t:.2f}' for t in frame_times]}")
    
    return frames, frame_indices, frame_times


class Qwen3VLVideoAttentionExtractor:
    """视频帧级注意力提取器"""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
        # 获取特殊 token IDs
        self.video_token_id = getattr(model.config, 'video_token_id', 151656)
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
        print(f"  Video token ID: {self.video_token_id}")
        print(f"  Image token ID: {self.image_token_id}")
        print(f"  Vision start token ID: {self.vision_start_token_id}")
        print(f"  Vision end token ID: {self.vision_end_token_id}")
        print(f"  Merge size: {self.merge_size}")
        print("="*60)
    
    def _find_video_token_indices(self, input_ids):
        """找到视频 token 的精确位置"""
        input_ids = input_ids.squeeze().cpu().numpy()
        
        vision_start_positions = np.where(input_ids == self.vision_start_token_id)[0]
        vision_end_positions = np.where(input_ids == self.vision_end_token_id)[0]
        
        print(f"\nFinding video tokens:")
        print(f"  Sequence length: {len(input_ids)}")
        print(f"  Vision start positions: {vision_start_positions}")
        print(f"  Vision end positions: {vision_end_positions}")
        
        if len(vision_start_positions) == 0 or len(vision_end_positions) == 0:
            video_positions = np.where(input_ids == self.video_token_id)[0]
            if len(video_positions) == 0:
                video_positions = np.where(input_ids == self.image_token_id)[0]
            print(f"  Fallback - found {len(video_positions)} video/image tokens")
            return video_positions if len(video_positions) > 0 else None
        
        start_pos = vision_start_positions[0]
        end_pos = vision_end_positions[0]
        video_token_indices = np.arange(start_pos + 1, end_pos)
        
        print(f"  Video token range: [{start_pos + 1}, {end_pos - 1}]")
        print(f"  Number of video tokens: {len(video_token_indices)}")
        
        return video_token_indices
    
    def _get_video_grid_info(self, inputs, num_tokens):
        """获取视频的 T, H, W 信息"""
        if 'image_grid_thw' in inputs:
            grid = inputs['image_grid_thw'][0].cpu()
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            h_merged = h // self.merge_size
            w_merged = w // self.merge_size
            tokens_per_frame = h_merged * w_merged
            expected = t * tokens_per_frame
            
            print(f"\n  Grid from image_grid_thw: T={t}, H={h}, W={w}")
            print(f"  After merge: H={h_merged}, W={w_merged}")
            print(f"  Tokens per frame: {tokens_per_frame}, Total expected: {expected}")
            
            if expected == num_tokens:
                return t, h_merged, w_merged, tokens_per_frame
            else:
                print(f"  WARNING: Expected {expected} but got {num_tokens} tokens")
                # 尝试重新计算
                if num_tokens % tokens_per_frame == 0:
                    t = num_tokens // tokens_per_frame
                    print(f"  Adjusted T={t}")
                    return t, h_merged, w_merged, tokens_per_frame
        
        # Fallback
        possible_tokens_per_frame = [256, 196, 144, 100, 64, 49, 36]
        
        for tpf in possible_tokens_per_frame:
            if num_tokens % tpf == 0:
                t = num_tokens // tpf
                sqrt_tpf = int(np.sqrt(tpf))
                if sqrt_tpf * sqrt_tpf == tpf:
                    print(f"  Fallback inferred: T={t}, H={sqrt_tpf}, W={sqrt_tpf}, tokens_per_frame={tpf}")
                    return t, sqrt_tpf, sqrt_tpf, tpf
        
        print(f"  Could not determine grid, treating as single frame")
        return 1, int(np.sqrt(num_tokens)), int(np.sqrt(num_tokens)), num_tokens
    
    def __call__(self, video_path_or_frames, prompt="Describe this video.", frame_times=None):
        with torch.no_grad():
            # 处理输入
            if isinstance(video_path_or_frames, str):
                frames, frame_indices, frame_times = extract_frames_from_video(video_path_or_frames)
            else:
                frames = video_path_or_frames
                frame_indices = list(range(len(frames)))
                if frame_times is None:
                    frame_times = list(range(len(frames)))
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                videos=[frames],
                padding=True,
                return_tensors="pt"
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            print(f"\nInput shapes:")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}")
            
            video_token_indices = self._find_video_token_indices(inputs['input_ids'])
            
            if video_token_indices is None:
                print("ERROR: No video tokens found!")
                return {}, "", frames, frame_indices, frame_times
            
            num_tokens = len(video_token_indices)
            t, h, w, tokens_per_frame = self._get_video_grid_info(inputs, num_tokens)
            
            # 校验帧数
            actual_num_frames = len(frames)
            if t != actual_num_frames:
                print(f"  NOTE: Grid T={t} differs from input frames={actual_num_frames}")
                # 使用grid的T作为真实帧数
                actual_num_frames = t
            
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
            print(f"Response: {response[:500]}...")
            print("="*60)
            
            # 获取注意力
            outputs = self.model(**inputs, output_attentions=True)
            
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                print(f"\nAttention layers: {len(outputs.attentions)}")
                print(f"Attention shape: {outputs.attentions[0].shape}")
                attentions = [attn.cpu().float() for attn in outputs.attentions]
                
                frame_attention_results = self._compute_frame_attention_methods(
                    attentions, inputs, video_token_indices, actual_num_frames, tokens_per_frame
                )
                
                return frame_attention_results, response, frames[:actual_num_frames], frame_indices[:actual_num_frames], frame_times[:actual_num_frames]
            else:
                print("No attention weights returned!")
                return {}, response, frames, frame_indices, frame_times
    
    def _compute_frame_attention_methods(self, attentions, inputs, video_token_indices, 
                                          num_frames, tokens_per_frame):
        """计算帧级注意力 - 尝试多种方法"""
        results = {}
        
        seq_len = attentions[0].size(-1)
        last_video_idx = video_token_indices[-1]
        first_video_idx = video_token_indices[0]
        
        text_start_idx = last_video_idx + 1
        text_end_idx = seq_len
        
        print(f"\n" + "="*60)
        print("Computing FRAME-LEVEL attention with multiple methods...")
        print(f"  Video tokens: [{first_video_idx}, {last_video_idx}] ({len(video_token_indices)} tokens)")
        print(f"  Text tokens: [{text_start_idx}, {text_end_idx - 1}] ({text_end_idx - text_start_idx} tokens)")
        print(f"  Number of frames: {num_frames}")
        print(f"  Tokens per frame: {tokens_per_frame}")
        print("="*60)
        
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
            
            stacked = torch.stack(layers)
            avg_attn = stacked.mean(dim=0).mean(dim=1)[0].numpy()
            
            print(f"\n--- Layers: {layer_name} ({len(layers)} layers) ---")
            
            # ============ 方法1: 文本 -> 视频帧 ============
            if text_start_idx < seq_len:
                text_to_video = avg_attn[text_start_idx:, video_token_indices]
                
                frame_attn = self._aggregate_to_frames(
                    text_to_video.sum(axis=0), num_frames, tokens_per_frame
                )
                results[f'{layer_name}_text_to_frame_sum'] = self._normalize(frame_attn)
                
                frame_attn = self._aggregate_to_frames(
                    text_to_video.mean(axis=0), num_frames, tokens_per_frame
                )
                results[f'{layer_name}_text_to_frame_mean'] = self._normalize(frame_attn)
                
                frame_attn = self._aggregate_to_frames(
                    avg_attn[-1, video_token_indices], num_frames, tokens_per_frame
                )
                results[f'{layer_name}_last_token_to_frame'] = self._normalize(frame_attn)
            
            # ============ 方法2: 视频帧 -> 文本 ============
            if text_start_idx < seq_len:
                video_to_text = avg_attn[video_token_indices, :][:, text_start_idx:]
                
                frame_attn = self._aggregate_to_frames(
                    video_to_text.sum(axis=1), num_frames, tokens_per_frame
                )
                results[f'{layer_name}_frame_to_text_sum'] = self._normalize(frame_attn)
                
                frame_attn = self._aggregate_to_frames(
                    video_to_text.mean(axis=1), num_frames, tokens_per_frame
                )
                results[f'{layer_name}_frame_to_text_mean'] = self._normalize(frame_attn)
            
            # ============ 方法3: 帧间自注意力 ============
            video_self_attn = avg_attn[video_token_indices, :][:, video_token_indices]
            
            col_sum = video_self_attn.sum(axis=0)
            frame_attn = self._aggregate_to_frames(col_sum, num_frames, tokens_per_frame)
            results[f'{layer_name}_frame_self_col_sum'] = self._normalize(frame_attn)
            
            row_sum = video_self_attn.sum(axis=1)
            frame_attn = self._aggregate_to_frames(row_sum, num_frames, tokens_per_frame)
            results[f'{layer_name}_frame_self_row_sum'] = self._normalize(frame_attn)
            
            # ============ 方法4: 帧间注意力矩阵 (T x T) ============
            frame_to_frame_attn = self._compute_frame_to_frame_attention(
                video_self_attn, num_frames, tokens_per_frame
            )
            if frame_to_frame_attn is not None:
                frame_attn = frame_to_frame_attn.sum(axis=0)
                results[f'{layer_name}_frame_receives'] = self._normalize(frame_attn)
                
                frame_attn = frame_to_frame_attn.sum(axis=1)
                results[f'{layer_name}_frame_sends'] = self._normalize(frame_attn)
                
                frame_attn = np.diag(frame_to_frame_attn)
                results[f'{layer_name}_frame_self_diag'] = self._normalize(frame_attn)
                
                # 保存帧间注意力矩阵用于可视化
                results[f'{layer_name}_frame_matrix'] = frame_to_frame_attn
            
            # ============ 方法5: 第一帧/最后一帧作为anchor ============
            if num_frames > 1:
                first_frame_tokens = video_token_indices[:tokens_per_frame]
                last_frame_tokens = video_token_indices[-tokens_per_frame:]
                
                first_to_all = avg_attn[first_frame_tokens, :][:, video_token_indices].sum(axis=0)
                frame_attn = self._aggregate_to_frames(first_to_all, num_frames, tokens_per_frame)
                results[f'{layer_name}_first_frame_to_all'] = self._normalize(frame_attn)
                
                last_to_all = avg_attn[last_frame_tokens, :][:, video_token_indices].sum(axis=0)
                frame_attn = self._aggregate_to_frames(last_to_all, num_frames, tokens_per_frame)
                results[f'{layer_name}_last_frame_to_all'] = self._normalize(frame_attn)
                
                # 其他帧看第一帧
                all_to_first = avg_attn[video_token_indices, :][:, first_frame_tokens].sum(axis=1)
                frame_attn = self._aggregate_to_frames(all_to_first, num_frames, tokens_per_frame)
                results[f'{layer_name}_all_to_first_frame'] = self._normalize(frame_attn)
            
            # ============ 方法6: 全局 -> 帧 ============
            all_to_video = avg_attn[:, video_token_indices].sum(axis=0)
            frame_attn = self._aggregate_to_frames(all_to_video, num_frames, tokens_per_frame)
            results[f'{layer_name}_global_to_frame'] = self._normalize(frame_attn)
            
            # ============ 方法7: 帧 -> 全局 ============
            video_to_all = avg_attn[video_token_indices, :].sum(axis=1)
            frame_attn = self._aggregate_to_frames(video_to_all, num_frames, tokens_per_frame)
            results[f'{layer_name}_frame_to_global'] = self._normalize(frame_attn)
        
        # ============ 方法8: Max head fusion ============
        stacked = torch.stack(attentions[-4:])
        max_attn = stacked.mean(dim=0).max(dim=1)[0][0].numpy()
        
        if text_start_idx < seq_len:
            text_to_video = max_attn[text_start_idx:, video_token_indices].mean(axis=0)
            frame_attn = self._aggregate_to_frames(text_to_video, num_frames, tokens_per_frame)
            results['last4_maxhead_text_to_frame'] = self._normalize(frame_attn)
        
        video_self = max_attn[video_token_indices, :][:, video_token_indices]
        frame_attn = self._aggregate_to_frames(video_self.sum(axis=0), num_frames, tokens_per_frame)
        results['last4_maxhead_frame_col'] = self._normalize(frame_attn)
        
        # ============ 方法9: 逐层分析 ============
        layer_indices = [0, len(attentions)//4, len(attentions)//2, 3*len(attentions)//4, -1]
        for layer_idx in layer_indices:
            attn = attentions[layer_idx].mean(dim=1)[0].numpy()
            
            video_self = attn[video_token_indices, :][:, video_token_indices]
            frame_attn = self._aggregate_to_frames(video_self.sum(axis=0), num_frames, tokens_per_frame)
            results[f'layer{layer_idx}_frame_col'] = self._normalize(frame_attn)
            
            if text_start_idx < seq_len:
                text_to_video = attn[text_start_idx:, video_token_indices].mean(axis=0)
                frame_attn = self._aggregate_to_frames(text_to_video, num_frames, tokens_per_frame)
                results[f'layer{layer_idx}_text_to_frame'] = self._normalize(frame_attn)
        
        # ============ 方法10: Attention Rollout (帧级) ============
        rollout_frame_attn = self._attention_rollout_frames(
            attentions[-4:], video_token_indices, num_frames, tokens_per_frame, text_start_idx
        )
        if rollout_frame_attn is not None:
            results['last4_rollout_frame'] = rollout_frame_attn
        
        # ============ 方法11: 基于方差的重要性 ============
        stacked = torch.stack(attentions[-4:])
        avg_attn = stacked.mean(dim=0).mean(dim=1)[0].numpy()
        video_self = avg_attn[video_token_indices, :][:, video_token_indices]
        
        # 每个token的注意力分布方差
        token_variance = video_self.var(axis=1)
        frame_attn = self._aggregate_to_frames(token_variance, num_frames, tokens_per_frame)
        results['last4_variance_per_frame'] = self._normalize(frame_attn)
        
        # ============ 方法12: 熵计算 ============
        # 对每个token，计算它发出的注意力的熵
        epsilon = 1e-10
        video_self_normalized = video_self / (video_self.sum(axis=1, keepdims=True) + epsilon)
        token_entropy = -np.sum(video_self_normalized * np.log(video_self_normalized + epsilon), axis=1)
        frame_attn = self._aggregate_to_frames(token_entropy, num_frames, tokens_per_frame)
        results['last4_entropy_per_frame'] = self._normalize(frame_attn)
        
        print(f"\n  Total methods computed: {len(results)}")
        
        return results
    
    def _aggregate_to_frames(self, token_attention, num_frames, tokens_per_frame):
        """将token级注意力聚合到帧级"""
        actual_len = len(token_attention)
        expected_len = num_frames * tokens_per_frame
        
        if actual_len != expected_len:
            if actual_len % num_frames == 0:
                tokens_per_frame = actual_len // num_frames
            elif actual_len % tokens_per_frame == 0:
                num_frames = actual_len // tokens_per_frame
            else:
                print(f"  Warning: token count mismatch. Expected {expected_len}, got {actual_len}")
        
        frame_attention = np.zeros(num_frames)
        for i in range(num_frames):
            start_idx = i * tokens_per_frame
            end_idx = min(start_idx + tokens_per_frame, len(token_attention))
            if start_idx < len(token_attention):
                frame_attention[i] = token_attention[start_idx:end_idx].sum()
        
        return frame_attention
    
    def _compute_frame_to_frame_attention(self, video_self_attn, num_frames, tokens_per_frame):
        """计算帧到帧的注意力矩阵 (T x T)"""
        try:
            frame_to_frame = np.zeros((num_frames, num_frames))
            
            for i in range(num_frames):
                for j in range(num_frames):
                    i_start = i * tokens_per_frame
                    i_end = min(i_start + tokens_per_frame, video_self_attn.shape[0])
                    j_start = j * tokens_per_frame
                    j_end = min(j_start + tokens_per_frame, video_self_attn.shape[1])
                    
                    if i_start < i_end and j_start < j_end:
                        frame_to_frame[i, j] = video_self_attn[i_start:i_end, j_start:j_end].sum()
            
            return frame_to_frame
        except Exception as e:
            print(f"  Frame-to-frame attention failed: {e}")
            return None
    
    def _attention_rollout_frames(self, attentions, video_token_indices, num_frames, 
                                   tokens_per_frame, text_start_idx):
        """Attention rollout for frames"""
        try:
            result = None
            for attn in attentions:
                attn_heads_fused = attn.mean(dim=1)[0].numpy()
                attn_heads_fused = 0.5 * attn_heads_fused + 0.5 * np.eye(attn_heads_fused.shape[0])
                attn_heads_fused = attn_heads_fused / (attn_heads_fused.sum(axis=-1, keepdims=True) + 1e-10)
                
                if result is None:
                    result = attn_heads_fused
                else:
                    result = result @ attn_heads_fused
            
            # 文本token对视频token的累积注意力
            if text_start_idx < result.shape[0]:
                video_attn = result[text_start_idx:, video_token_indices].sum(axis=0)
            else:
                video_attn = result[:, video_token_indices].sum(axis=0)
            
            frame_attn = self._aggregate_to_frames(video_attn, num_frames, tokens_per_frame)
            return self._normalize(frame_attn)
        except Exception as e:
            print(f"  Rollout failed: {e}")
            return None
    
    def _normalize(self, arr):
        """归一化到 [0, 1]"""
        arr = np.array(arr)
        if arr.max() > arr.min():
            return (arr - arr.min()) / (arr.max() - arr.min())
        return np.ones_like(arr) * 0.5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--video_path', type=str, required=True, help='Path to video file')
    parser.add_argument('--prompt', type=str, default='Describe what happens in this video.')
    parser.add_argument('--max_frames', type=int, default=16, help='Maximum number of frames to extract')
    parser.add_argument('--model_cache_path', type=str, 
                        default=os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Thinking'))
    parser.add_argument('--output_dir', type=str, default='outputs/video_attention')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    print(f"Using {'GPU' if args.use_cuda else 'CPU'}")
    return args


def visualize_frame_attention(frames, frame_indices, frame_times, attention_dict, output_dir, cols=4):
    """可视化帧级注意力"""
    
    num_frames = len(frames)
    
    # 1. 条形图目录
    bar_dir = output_dir / "bar_charts"
    bar_dir.mkdir(exist_ok=True)
    
    # 过滤掉矩阵类型的结果
    scalar_results = {k: v for k, v in attention_dict.items() 
                      if v is not None and isinstance(v, np.ndarray) and v.ndim == 1}
    
    for method_name, attention_values in scalar_results.items():
        plt.figure(figsize=(max(12, num_frames * 0.8), 5))
        x = np.arange(len(attention_values))
        colors = plt.cm.RdYlGn(attention_values)  # 红-黄-绿配色
        
        bars = plt.bar(x, attention_values, color=colors, edgecolor='black', linewidth=0.5)
        
        # 标记最高的帧
        max_idx = np.argmax(attention_values)
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(3)
        
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Attention (normalized)', fontsize=12)
        plt.title(f'Frame Attention: {method_name}', fontsize=14)
        plt.xticks(x, [f'{i}\n({frame_times[i]:.1f}s)' if i < len(frame_times) else str(i) 
                       for i in range(len(attention_values))], fontsize=8)
        plt.ylim(0, 1.15)
        
        # 在最高的bar上标注
        plt.annotate(f'Max: Frame {max_idx}', 
                     xy=(max_idx, attention_values[max_idx]),
                     xytext=(max_idx, attention_values[max_idx] + 0.08),
                     ha='center', fontsize=10, color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        plt.savefig(bar_dir / f'{method_name}.png', dpi=150)
        plt.close()
    
    # 2. 帧网格图 (关键方法)
    key_methods = [
        'last_4_text_to_frame_sum',
        'last_4_frame_to_text_sum', 
        'last_4_frame_self_col_sum',
        'last_4_frame_receives',
        'last_4_global_to_frame',
        'all_text_to_frame_sum',
        'last4_rollout_frame',
    ]
    
    available_methods = [m for m in key_methods if m in scalar_results]
    if not available_methods and scalar_results:
        available_methods = list(scalar_results.keys())[:6]
    
    for method_name in available_methods:
        attention_values = scalar_results.get(method_name)
        if attention_values is None or len(attention_values) != num_frames:
            continue
        
        rows = (num_frames + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.array(axes).flatten() if rows > 1 or cols > 1 else [axes]
        
        for idx in range(num_frames):
            ax = axes[idx]
            frame = frames[idx]
            attn_val = attention_values[idx]
            
            ax.imshow(frame)
            
            # 根据注意力值设置边框颜色和粗细
            color = plt.cm.RdYlGn(attn_val)
            linewidth = 3 + attn_val * 12  # 3-15的线宽
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(linewidth)
            
            time_str = f'{frame_times[idx]:.2f}s' if idx < len(frame_times) else ''
            ax.set_title(f'Frame {idx} ({time_str})\nAttn: {attn_val:.3f}', 
                        fontsize=11, fontweight='bold' if attn_val > 0.7 else 'normal')
            ax.axis('off')
        
        for idx in range(num_frames, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Method: {method_name}\n(Border thickness ∝ attention)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'frames_{method_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. 帧间注意力矩阵热力图
    matrix_results = {k: v for k, v in attention_dict.items() 
                      if v is not None and isinstance(v, np.ndarray) and v.ndim == 2}
    
    if matrix_results:
        matrix_dir = output_dir / "frame_matrices"
        matrix_dir.mkdir(exist_ok=True)
        
        for method_name, matrix in matrix_results.items():
            plt.figure(figsize=(10, 8))
            plt.imshow(matrix, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Attention')
            plt.xlabel('Target Frame', fontsize=12)
            plt.ylabel('Source Frame', fontsize=12)
            plt.title(f'Frame-to-Frame Attention: {method_name}', fontsize=14)
            plt.xticks(range(matrix.shape[1]))
            plt.yticks(range(matrix.shape[0]))
            plt.tight_layout()
            plt.savefig(matrix_dir / f'{method_name}.png', dpi=150)
            plt.close()
    
    # 4. 所有方法对比总览图
    create_method_comparison(scalar_results, frame_times, output_dir)
    
    # 5. 创建带缩略图的时间线
    create_timeline_visualization(frames, frame_times, scalar_results, output_dir)


def create_method_comparison(attention_dict, frame_times, output_dir):
    """创建所有方法的对比图"""
    if not attention_dict:
        return
    
    num_methods = len(attention_dict)
    cols = 4
    rows = (num_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    axes = np.array(axes).flatten()
    
    for idx, (method_name, attention_values) in enumerate(attention_dict.items()):
        ax = axes[idx]
        x = np.arange(len(attention_values))
        colors = plt.cm.RdYlGn(attention_values)
        
        bars = ax.bar(x, attention_values, color=colors, edgecolor='gray', linewidth=0.3)
        
        max_idx = np.argmax(attention_values)
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(2)
        
        # 简化方法名
        short_name = method_name.replace('last_', 'L').replace('first_', 'F').replace('middle_', 'M')
        short_name = short_name.replace('_to_', '→').replace('_sum', 'Σ').replace('_mean', 'μ')
        short_name = short_name.replace('text', 'T').replace('frame', 'Fr').replace('global', 'G')
        
        ax.set_title(f'{short_name}\nMax: Fr{max_idx}', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.tick_params(labelsize=7)
    
    for idx in range(num_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Frame Attention Comparison - All Methods\n(Red border = max attention frame)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'all_methods_comparison.png', dpi=200)
    plt.close()


def create_timeline_visualization(frames, frame_times, attention_dict, output_dir):
    """创建带缩略图的时间线可视化"""
    
    # 选择几个代表性方法
    key_methods = ['last_4_text_to_frame_sum', 'last_4_frame_receives', 'all_text_to_frame_sum']
    available = [m for m in key_methods if m in attention_dict]
    
    if not available:
        available = list(attention_dict.keys())[:3]
    
    if not available:
        return
    
    num_frames = len(frames)
    
    for method_name in available:
        attention_values = attention_dict[method_name]
        if len(attention_values) != num_frames:
            continue
        
        fig, axes = plt.subplots(2, 1, figsize=(max(14, num_frames * 1.5), 8), 
                                  gridspec_kw={'height_ratios': [3, 1]})
        
        # 上方: 帧缩略图
        ax_frames = axes[0]
        thumb_width = 1.0 / num_frames
        
        for i, (frame, attn) in enumerate(zip(frames, attention_values)):
            # 计算位置
            left = i * thumb_width
            
            # 创建小的子axes显示帧
            ax_inset = ax_frames.inset_axes([left + 0.01, 0.1, thumb_width - 0.02, 0.8])
            ax_inset.imshow(frame)
            ax_inset.axis('off')
            
            # 边框颜色表示注意力
            color = plt.cm.RdYlGn(attn)
            for spine in ax_inset.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3 + attn * 7)
                spine.set_visible(True)
        
        ax_frames.set_xlim(0, 1)
        ax_frames.set_ylim(0, 1)
        ax_frames.axis('off')
        ax_frames.set_title(f'Video Frames (border color = attention)', fontsize=12)
        
        # 下方: 注意力曲线
        ax_attn = axes[1]
        x = np.arange(num_frames)
        colors = plt.cm.RdYlGn(attention_values)
        
        ax_attn.bar(x, attention_values, color=colors, edgecolor='black', linewidth=0.5, width=0.8)
        ax_attn.plot(x, attention_values, 'k-', linewidth=2, marker='o', markersize=6)
        
        max_idx = np.argmax(attention_values)
        ax_attn.axvline(x=max_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_attn.annotate(f'Peak: Frame {max_idx}', 
                        xy=(max_idx, attention_values[max_idx]),
                        xytext=(max_idx + 0.5, attention_values[max_idx] + 0.1),
                        fontsize=10, color='red')
        
        ax_attn.set_xlabel('Frame Index', fontsize=11)
        ax_attn.set_ylabel('Attention', fontsize=11)
        ax_attn.set_xticks(x)
        ax_attn.set_xticklabels([f'{i}\n{frame_times[i]:.1f}s' if i < len(frame_times) else str(i) 
                                  for i in x], fontsize=8)
        ax_attn.set_ylim(0, 1.2)
        ax_attn.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Frame Attention Timeline: {method_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'timeline_{method_name}.png', dpi=150, bbox_inches='tight')
        plt.close()


def generate_summary_report(frames, frame_times, attention_dict, response, output_dir):
    """生成结果摘要报告"""
    
    # 过滤掉矩阵结果
    scalar_results = {k: v for k, v in attention_dict.items() 
                      if v is not None and isinstance(v, np.ndarray) and v.ndim == 1}
    
    lines = []
    lines.append("="*70)
    lines.append("VIDEO FRAME ATTENTION ANALYSIS REPORT")
    lines.append("="*70)
    lines.append(f"\nNumber of frames analyzed: {len(frames)}")
    lines.append(f"Frame times: {[f'{t:.2f}s' for t in frame_times]}")
    lines.append(f"\nMethods computed: {len(scalar_results)}")
    
    lines.append("\n" + "="*70)
    lines.append("MODEL RESPONSE")
    lines.append("="*70)
    lines.append(response)
    
    lines.append("\n" + "="*70)
    lines.append("TOP ATTENTION FRAMES PER METHOD")
    lines.append("="*70)
    
    # 统计每帧被选为top的次数
    frame_top_counts = {i: 0 for i in range(len(frames))}
    
    for method_name, attention_values in sorted(scalar_results.items()):
        if len(attention_values) == 0:
            continue
        
        top_3_indices = np.argsort(attention_values)[-3:][::-1]
        top_3_values = [attention_values[i] for i in top_3_indices]
        
        lines.append(f"\n{method_name}:")
        lines.append(f"  Top 3 frames: {top_3_indices.tolist()}")
        lines.append(f"  Attention values: {[f'{v:.4f}' for v in top_3_values]}")
        
        frame_top_counts[top_3_indices[0]] += 1
    
    lines.append("\n" + "="*70)
    lines.append("CONSENSUS ANALYSIS")
    lines.append("="*70)
    lines.append("\nHow many methods selected each frame as TOP-1:")
    
    sorted_counts = sorted(frame_top_counts.items(), key=lambda x: x[1], reverse=True)
    for frame_idx, count in sorted_counts:
        if count > 0:
            time_str = f'{frame_times[frame_idx]:.2f}s' if frame_idx < len(frame_times) else 'N/A'
            lines.append(f"  Frame {frame_idx} ({time_str}): selected by {count} methods")
    
    # 找出共识帧
    if sorted_counts[0][1] > len(scalar_results) * 0.3:
        consensus_frame = sorted_counts[0][0]
        lines.append(f"\n>>> CONSENSUS: Frame {consensus_frame} is likely the most important frame")
    else:
        lines.append("\n>>> No strong consensus - attention is distributed across frames")
    
    return '\n'.join(lines)


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

    # 提取帧
    frames, frame_indices, frame_times = extract_frames_from_video(
        args.video_path, max_frames=args.max_frames
    )
    
    # 保存提取的帧
    frames_dir = output_dir / "extracted_frames"
    frames_dir.mkdir(exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{idx:03d}.jpg")

    # 提取注意力
    attention_extractor = Qwen3VLVideoAttentionExtractor(model, processor)
    attention_results, response, frames, frame_indices, frame_times = attention_extractor(
        frames, prompt=args.prompt, frame_times=frame_times
    )
    
    # 可视化
    print("\nGenerating visualizations...")
    visualize_frame_attention(frames, frame_indices, frame_times, attention_results, output_dir)
    
    # 生成报告
    report = generate_summary_report(frames, frame_times, attention_results, response, output_dir)
    
    with open(str(output_dir / "report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Video: {args.video_path}\n")
        f.write(f"Prompt: {args.prompt}\n\n")
        f.write(report)
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60)
    print("\nKey outputs:")
    print("  - all_methods_comparison.png: Overview of all methods")
    print("  - timeline_*.png: Timeline visualizations with frame thumbnails")
    print("  - frames_*.png: Frame grids with attention highlighting")  
    print("  - bar_charts/: Individual bar charts for each method")
    print("  - frame_matrices/: Frame-to-frame attention heatmaps")
    print("  - report.txt: Detailed analysis report with consensus")
    print("="*60)