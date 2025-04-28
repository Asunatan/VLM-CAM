import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, Qwen2VLForConditionalGeneration

from my_vlm.datasets import LazySupervisedDataset


class ModelEvaluator:
    def __init__(self, model_path, original_model_path, json_file_path, image_folder, mode='qwen2vl'):
        self.model_path = model_path
        self.original_model_path = original_model_path
        self.json_file_path = json_file_path
        self.image_folder = image_folder
        self.mode = mode
        self.processor = None
        self.model = None
        self.test_dataset_loader = None
        self.feature_maps = {}
        self.gradients = {}
        self.current_layer_name = None

    def replace_image_tokens(self, input_string):
        input_string = input_string.replace("<image>\n", "")
        input_string = input_string.replace("<image>", "")
        return input_string

    def find_vision_token_indexes(self, input_list):
        start_token = 151652
        end_token = 151653
        start_indices = []
        end_indices = []

        try:
            for i in range(len(input_list)):
                if input_list[i] == start_token:
                    start_indices.append(i + 1)  # 记录开始 token 的索引
                elif input_list[i] == end_token:
                    end_indices.append(i)  # 记录结束 token 的索引
            if len(start_indices) != len(end_indices):
                raise ValueError("开始和结束 token 的数量不一致")
            return start_indices, end_indices
        except Exception as e:
            print(f"An error occurred: {e}")
            return [-1], [-1]

    def setup_seeds(self, config):
        seed = config.run_cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    def collate_fn(self, batch):
        ids, images, videos, conversations, system_prompt = zip(*batch)
        ids = [item['id'] for item in batch]
        images = [item['images'] for item in batch]
        videos = [item['videos'] for item in batch]
        conversations = [item['conversations'] for item in batch]
        system_prompts = [item['system_prompt'] for item in batch]
        return {
            'ids': ids,
            'images': images,
            'videos': videos,
            'conversations': conversations,
            'system_prompts': system_prompts
        }



    # 修改特征图保存函数
    def save_feature_maps(self, layer_name, module, input, output):
        # 更新当前处理的层名称
        self.current_layer_name = layer_name
        # 如果层名称不存在于字典中，初始化一个空列表
        if layer_name not in self.feature_maps:
            self.feature_maps[layer_name] = []
        # 将特征图数据添加到对应层的列表中
        feature_map_data = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
        self.feature_maps[layer_name].append(feature_map_data)
        # 如果需要保留梯度，可以在这里实现
        output[0].retain_grad() if isinstance(output, tuple) else output.retain_grad()

    # 修改梯度保存函数
    def save_gradients(self, layer_name, module, grad_input, grad_output):
        # 如果层名称不存在于字典中，初始化一个空列表
        if layer_name not in self.gradients:
            self.gradients[layer_name] = []
        # 将梯度数据添加到对应层的列表中
        gradient_data = grad_output[0].detach().cpu() if isinstance(grad_output, tuple) else grad_output.detach().cpu()
        self.gradients[layer_name].append(gradient_data)

    def register_hooks(self, model):
        layers = []
        if self.mode == 'llava':
            for i in range(len(model.language_model.model.layers)):
                layers.append(model.language_model.model.layers[i].post_attention_layernorm)
        elif self.mode == 'qwen2vl':
            for i in range(len(model.model.layers)):
                layers.append(model.model.layers[i].post_attention_layernorm)
                # layers.append(model.visual.blocks[i].attn)
        # 为每一层注册钩子，并传递层名称
        for layer_idx, layer in enumerate(layers):
            layer_name = f"layer_{layer_idx}"
            # 为前向传播注册钩子
            layer.register_forward_hook(
                lambda module, input, output, name=layer_name: self.save_feature_maps(name, module, input, output))
            # 为反向传播注册钩子
            layer.register_full_backward_hook(
                lambda module, grad_input, grad_output, name=layer_name: self.save_gradients(name, module, grad_input,
                                                                                             grad_output))
    def initialize_model(self):
        if self.mode == 'llava':
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(self.model_path,
                                                                                device_map="cuda:3",
                                                                                attn_implementation="eager",
                                                                                torch_dtype=torch.bfloat16).eval()
        elif self.mode == 'qwen2vl':
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path,
                                                                        device_map="cuda:3",
                                                                        attn_implementation="eager",
                                                                        torch_dtype=torch.bfloat16).eval()
        self.processor = AutoProcessor.from_pretrained(self.original_model_path)
        self.test_dataset_loader = DataLoader(LazySupervisedDataset(data_path=self.json_file_path, image_folder=self.image_folder),
                                              num_workers=4, batch_size=1, collate_fn=self.collate_fn)

    def process_batch(self, batch_data):
        image_list = batch_data['images'][0]
        content = [{"type": "text", "text": '中间的彩色是什么？'}]
        for _ in image_list:
            content.append({"type": "image"})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=image_list, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        vision_token_start, vision_token_end = self.find_vision_token_indexes(inputs['input_ids'].squeeze(0).tolist())
        return inputs, vision_token_start, vision_token_end, image_list

    def generate_and_save_cams(self, inputs,all_token, vision_token_start, vision_token_end, image_list):
        for layer_idx in self.feature_maps.keys():
            feature_map = self.feature_maps[layer_idx]
            gradient = self.gradients[layer_idx]
            feature_map = torch.cat(feature_map, dim=1)
            gradient = torch.cat(gradient, dim=1)
            for img_idx in range(len(image_list)):
                w = h = 12 if self.mode == 'llava' else 14  # 根据模型调整
                w = h = 12 if self.mode == 'qwen2vl' else 14
                img_feature_map = rearrange(feature_map[:, vision_token_start[img_idx]:vision_token_end[img_idx], :], 'b (h w) c ->b c h w', h=h, w=w)
                img_gradient = rearrange(gradient[:, vision_token_start[img_idx]:vision_token_end[img_idx], :], 'b (h w) c -> b h w c', h=h, w=w)

                img_gradient = nn.ReLU()(img_gradient)
                pooled_gradients = torch.mean(img_gradient, dim=[0, 1, 2])
                activation = img_feature_map.squeeze(0)
                for i in range(activation.size(0)):
                    activation[i, :, :] *= pooled_gradients[i]
                heatmap = torch.mean(activation, dim=0).cpu().float().numpy()
                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)

                patch_size = 14
                heatmap = cv2.resize(heatmap, (336, 336))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                original_image = np.array(image_list[img_idx])
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                superimposed_img = heatmap * 0.4 + original_image
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                save_path = '/data/scy/SCY/my_vlm/heatmap'
                path_cam_img = os.path.join(save_path, f"layer_{layer_idx}_img_num{img_idx}.png")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(path_cam_img, superimposed_img)
            a=0
        b=0

    def train_and_generate(self):
        self.register_hooks(self.model)
        for idx, batch_data in tqdm(enumerate(self.test_dataset_loader), total=len(self.test_dataset_loader)):
            inputs, vision_token_start, vision_token_end, image_list = self.process_batch(batch_data)
            generated_tokens = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]
            if self.mode == 'llava':
                image_sizes = inputs["image_sizes"]
                position_ids = torch.arange(0, inputs.input_ids.shape[1]).to(self.model.device)
            elif self.mode == 'qwen2vl':
                image_grid_thw = inputs["image_grid_thw"]
            past_key_values = None
            cache_position = torch.arange(0, inputs.input_ids.shape[1]).to(self.model.device)
            i = 0
            sequence_logits = []
            while True:
                self.model.zero_grad()
                with torch.set_grad_enabled(True):
                    if self.mode == 'llava':
                        outputs = self.model(
                            input_ids=generated_tokens[:, -1:].to(self.model.device) if past_key_values else generated_tokens,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            cache_position=cache_position,
                            logits_to_keep=1
                        )
                    elif self.mode == 'qwen2vl':
                        outputs = self.model(
                            input_ids=generated_tokens[:, -1:] if past_key_values else generated_tokens,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            past_key_values=past_key_values,
                            use_cache=True,
                            cache_position=cache_position
                        )
                    logits = outputs.logits
                    sequence_logits.append(logits[:, -1, :])
                    past_key_values = outputs.past_key_values
                    pixel_values = None
                    cache_position = cache_position[-1].unsqueeze(-1) + 1

                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long).to(attention_mask.device)], dim=-1)
                i += 1

                if next_token.item() == self.processor.tokenizer.eos_token_id:
                    break
                if generated_tokens.shape[1] >= 2048:
                    break

            input_token_len = inputs.input_ids.shape[1]
            sequence_logits = torch.cat(sequence_logits, dim=0)
            target_logits = torch.sum(sequence_logits[
                                      torch.arange(sequence_logits.shape[0]), torch.argmax(sequence_logits, dim=-1)])
            target_logits.backward(retain_graph=True)
            self.generate_and_save_cams(inputs,generated_tokens, vision_token_start, vision_token_end, image_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, default='/data/scy/SCY/Qwen/Qwen2-VL-2B-Instruct',
                        choices=['/data/scy/SCY/LLava/LLaVA-Onevision-7B', '/data/scy/SCY/LLava/Llama-3.2-11B-Vision-Instruct'])
    parser.add_argument("--original_model_path", type=str, default='/data/scy/SCY/Qwen/Qwen2-VL-2B-Instruct',
                        choices=['/data/scy/SCY/LLava/LLaVA-Onevision-7B',
                                 '/data/scy/SCY/LLava/Llama-3.2-11B-Vision-Instruct'])
    parser.add_argument("--json_file_path", type=str, default=['/data/scy/SCY/my_vlm/dataset/test300.json'],)
    parser.add_argument("--image_folder", type=str, default='/data/scy/SCY/2024testnew')
    args = parser.parse_known_args()[0]

    evaluator = ModelEvaluator(
        model_path=args.model_path,
        original_model_path=args.original_model_path,
        json_file_path=args.json_file_path,
        image_folder=args.image_folder
    )
    evaluator.initialize_model()
    evaluator.train_and_generate()
