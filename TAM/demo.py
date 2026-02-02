import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText
from qwen_utils import process_vision_info
from PIL import Image
from tam import TAM


def tam_demo_for_qwen2_vl(image_path, prompt_text, save_dir='vis_results'):
    # Load Qwen2-VL model and processor
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Prepare input message with image/video and prompt
    if isinstance(image_path, list):
        messages = [{"role": "user", "content": [{"type": "video", "video": image_path}, {"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]

    # Process input text and visual info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Generate model output with hidden states for visualization
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        output_hidden_states=True, # ---> TAM needs hidden states
        return_dict_in_generate=True
    )

    generated_ids = outputs.sequences

    # === TAM code part ====

    # Compute logits from last hidden states with vocab classifier for TAM
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

    # Define special token IDs to separate image/prompt/answer tokens
    # See TAM in tam.py about its usage. See ids from the specific model.
    special_ids = {'img_id': [151652, 151653],
                   'prompt_id': [151653, [151645, 198, 151644, 77091]], 
                   'answer_id': [[198, 151644, 77091, 198], -1]}

    # get shape of vision output
    if isinstance(image_path, list):
        vision_shape = (inputs['video_grid_thw'][0, 0], inputs['video_grid_thw'][0, 1] // 2, inputs['video_grid_thw'][0, 2] // 2)
    else:
        vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)

    # get img or video inputs for next vis
    vis_inputs = [[video_inputs[0][i] for i in range(0, len(video_inputs[0]))]] if isinstance(image_path, list) else image_inputs

    # === TAM Visualization ===
    # Call TAM() to generate token activation map for each generation round
    # Arguments:
    # - token ids (inputs and generations)
    # - shape of vision token
    # - logits for each round
    # - special token identifiers for localization
    # - image / video inputs for visualization
    # - processor for decoding
    # - output image path to save the visualization
    # - round index (0 here)
    # - raw_vis_records: list to collect intermediate visualization data
    # - eval only, False to vis
    # return TAM vision map for eval, saving multimodal TAM in the function
    raw_map_records = []
    for i in range(len(logits)):
        img_map = TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape,
            logits,
            special_ids,
            vis_inputs,
            processor,
            os.path.join(save_dir, str(i) + '.jpg'),
            i,
            raw_map_records,
            False)


# InternVL2_5 uses its independent code, here I vis InternVL3 from the official transformers.
def tam_demo_for_internvl3(image_path, prompt_text, save_dir='vis_results'):
    torch_device = "cuda"
    model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)
    
    # Fixed size, if you want to use dymamic reso, please map crops to raw size.
    image = Image.open(img)
    image = image.resize((448, 448))
    conversation = [{"role":"user", "content":[{"type":"image",}, {"type":"text", "text":"%s" % (prompt)}]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device).to(model.dtype)

    outputs = model.generate(**inputs, max_new_tokens=256, output_hidden_states=True, return_dict_in_generate=True)
    generated_ids = outputs.sequences

    # === TAM code part ====

    # Compute logits from last hidden states with vocab classifier for TAM
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

    # InternVL has different special ids, please vis inputs['input_ids'] for special ids
    special_ids = {'img_id': [151665, 151666],
                   'prompt_id': [[151666, 198], [151645, 198, 151644, 77091]], 
                   'answer_id': [[198, 151644, 77091, 198], -1]}

    vision_shape = (16, 16)
    vis_inputs = image
    
    raw_map_records = []
    for i in range(len(logits)):
        img_map = TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape,
            logits,
            special_ids,
            vis_inputs,
            processor,
            os.path.join(save_dir, str(i) + '.jpg'),
            i,
            raw_map_records,
            False)


if __name__ == "__main__":
    # single img demo (qwen)
    img = "imgs/demo.jpg"
    prompt = "Describe this image."
    tam_demo_for_qwen2_vl(img, prompt, save_dir='imgs/vis_img')

    # single img demo (internvl)
    tam_demo_for_internvl3(img, prompt, save_dir='imgs/vis_img_internvl')

    # video demo (qwen)
    imgs = []
    for i in range(10):
        # QWen merges next frames, repeating to vis each frame
        imgs.extend(["imgs/frames/%s.jpg" % (str(i).zfill(4))] * 2)
    prompt = "Describe this video."
    tam_demo_for_qwen2_vl(imgs, prompt, save_dir='imgs/vis_video')
