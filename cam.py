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
        self.feature_maps = []
        self.gradients = None

    def replace_image_tokens(self, input_string):
        input_string = input_string.replace("<image>\n", "")
        input_string = input_string.replace("<image>", "")
        return input_string
    def find_vision_token_indexes(self,input_list):
        '''
        A message from train_data/data.json may look like below:
            {
                "messages": [
                    {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]},
                    {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
                ]
            }
        After apply_chat_template, the text will look like below:
            ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

        This function tries to find the indexes of the assistant content in the input_ids list to build labels.
        '''
        # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
        # [151644, 77091, 198]
        # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
        # [151645, 198]

        try:
            # 查找第一个特定值的索引位置
            start_index = input_list.index(151655)
            # 查找最后一个特定值的索引位置
            end_index = len(input_list) - 1 - input_list[::-1].index(151655)
            return start_index, end_index
        except ValueError:
            # 如果列表中没有找到特定值，返回默认值或进行其他处理
            return -1, -1

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

    def save_feature_maps(self, module, input, output):
        # self.feature_maps = output[0] if isinstance(output, tuple) else output
        self.feature_maps.append(output[0] if isinstance(output, tuple) else output)
        output[0].retain_grad() if isinstance(output, tuple) else output.retain_grad()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach().cpu() if isinstance(grad_output, tuple) else grad_output.detach().cpu())


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

    def generate_and_save_cam(self):
        for idx, batch_data in tqdm(enumerate(self.test_dataset_loader), total=len(self.test_dataset_loader)):
            image_list = batch_data['images'][0]
            content = [{"type": "text", "text": '图中哪里是血管？'}]
            for _ in image_list:
                content.append({"type": "image"})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=image_list, padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            vision_token_start,vision_token_end=self.find_vision_token_indexes(inputs['input_ids'].squeeze(0).tolist())
            if self.mode == 'llava':
                self.model.language_model.model.layers[0].post_attention_layernorm.register_forward_hook(self.save_feature_maps)
                self.model.language_model.model.layers[0].post_attention_layernorm.register_backward_hook(self.save_gradients)
            elif self.mode == 'qwen2vl':
                self.model.model.layers[0].register_forward_hook(self.save_feature_maps)
                self.model.model.layers[0].register_backward_hook(self.save_gradients)

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
            # outputs = self.model.generate(**inputs, use_cache=True, max_new_tokens=4096,
            #                                     # pad_token_id=processor.tokenizer.eos_token_id,
            #                                     return_dict_in_generate=True,
            #                                     output_attentions=True,
            #                                     output_scores=True
            #                                     )
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
                    sequence_logits.append(logits)
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
            sequence_logits = torch.cat(sequence_logits, dim=1)
            target_logits = torch.sum(sequence_logits[0][input_token_len-1:generated_tokens.shape[1], :][
                                      torch.arange(generated_tokens.shape[1]-input_token_len), generated_tokens[:,input_token_len:].squeeze(0)])
            target_logits.backward(retain_graph=True)
            num_token = self.feature_maps.shape[0]
            h = int(np.sqrt(num_token//len(image_list)))
            self.feature_maps = torch.cat(self.feature_maps,dim=1)
            self.feature_maps = rearrange(self.feature_maps[num_token//len(image_list):, :].detach(), '(h w) c -> c h w ', w=h,
                                          h=h)
            self.gradients = rearrange(self.gradients[num_token//len(image_list):, :].detach(), '(h w) c -> h w c', w=h, h=h)
            self.gradients = nn.ReLU()(self.gradients)
            pooled_gradients = torch.mean(self.gradients, dim=[0, 1])
            activation = self.feature_maps.squeeze(0)
            for i in range(activation.size(0)):
                activation[i, :, :] *= pooled_gradients[i]
            # 创建热力图
            # activation = activation.permute(0,2,1)
            heatmap = torch.mean(activation, dim=0).cpu().float().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)

            # 特征筛选
            # threshold = 0.5  # 可以调整这个值
            # heatmap[heatmap < threshold] = 0  # 将低于阈值的值设为0
            patch_size = 14
            heatmap = cv2.resize(heatmap, (h*patch_size, h*patch_size))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 将热力图叠加到原始图像上
            original_image = np.array(image_list[1])
            superimposed_img = heatmap * 0.4 + original_image
            superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
            save_path = '/data/scy/SCY/my_vlm/heatmap'
            path_cam_img = os.path.join(save_path, f"layer_{31}.jpg")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(path_cam_img, superimposed_img)
            return heatmap, superimposed_img

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
    evaluator.generate_and_save_cam()