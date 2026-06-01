from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import math
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
import re
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer, util

# ===== Region-crop utilities for multi-round generation =====
BBOX_PATTERN = re.compile(r"<\|box_start\|>\[([^\]]+)\]<\|box_end\|>")


def _smart_resize_like_qwen(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    """
    Qwen-like smart resize (aligned with training mm_plugin.py):
    - Round to multiples of factor
    - If too big: scale down; if too small: scale up
    Returns (new_h, new_w)
    """
    if height < factor or width < factor:
        h_bar = max(factor, math.ceil(height / factor) * factor)
        w_bar = max(factor, math.ceil(width / factor) * factor)
        return h_bar, w_bar

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    num = h_bar * w_bar

    if num > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif num < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    h_bar = max(factor, int(h_bar))
    w_bar = max(factor, int(w_bar))
    return h_bar, w_bar


def _resize_pil_to_processor(img: Image.Image, processor) -> Image.Image:
    """
    Resize PIL img to the processor-aligned size using smart_resize.
    This matches how training data coordinates were computed.
    """
    ip = getattr(processor, "image_processor", None)
    patch = getattr(ip, "patch_size", 14) if ip is not None else 14
    merge = getattr(ip, "merge_size", 2) if ip is not None else 2
    factor = patch * merge

    min_pixels = getattr(ip, "min_pixels", None) if ip is not None else None
    if min_pixels is None:
        min_pixels = 3136
    max_pixels = getattr(ip, "max_pixels", None) if ip is not None else None
    if max_pixels is None:
        max_pixels = 12845056

    orig_w, orig_h = img.size
    proc_h, proc_w = _smart_resize_like_qwen(
        orig_h, orig_w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )

    if (proc_w, proc_h) != (orig_w, orig_h):
        img = img.resize((proc_w, proc_h), Image.BILINEAR)
    return img


def crop_2d_image(image, box):
    """Crop 2D image with bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = [int(c) for c in box[:4]]
    w, h = image.size
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return image.crop((x1, y1, x2, y2))


def parse_last_bbox(text):
    """Parse the last bbox from generated text. Returns coords list or None."""
    matches = list(BBOX_PATTERN.finditer(text))
    if not matches:
        return None
    coord_str = matches[-1].group(1)
    try:
        coords = [float(x.strip()) for x in coord_str.split(",")]
        if len(coords) == 4:
            return coords
    except:
        pass
    return None

client = OpenAI(
    api_key=os.getenv(""),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)



from sentence_transformers import SentenceTransformer, util
import difflib
import numpy as np
import re

_model = SentenceTransformer('/volume/med-train/users/shuyan/lhg/model/all-MiniLM-L6-v2')

def evaluate_answer_similarity(student_answer, ground_truth):
    

    sa = student_answer.strip().lower()
    gt = ground_truth.strip().lower()

    embeddings = _model.encode([sa, gt], convert_to_tensor=True)
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    cosine_sim = max(0.0, min(1.0, cosine_sim))

    
    
    return cosine_sim 


    
class Qwen2VLModule(VLMBaseModule):
    
    def __init__(self):
        super().__init__()
        
    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case "general":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer>  answer here </answer>"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def format_reward_gen(completions, **kwargs):
        """Check if the Qwen model output has <think> and <answer> tags in correct format."""
        import re
        import os
        from datetime import datetime

        # 更宽松的正则，只检查 <think>...</think> 和 <answer>...</answer>
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def llm_reward(completions, solution, **kwargs):
        """
        支持 batched 输入的 reward 函数。
        相似度用 sentence-transformer 和编辑距离的最小值。
        """
        def extract_answer(text):
            match = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
            return match[-1].strip() if match else text.strip()

        rewards = []
        contents = [completion[0]["content"] for completion in completions]

        for content, sol in zip(contents, solution):
            student_answer = extract_answer(content)
            ground_truth = extract_answer(sol)
            reward = evaluate_answer_similarity(student_answer, ground_truth)
            rewards.append(reward)

        #print(np.mean(rewards))
        return rewards
    
    

    # @staticmethod
    # def llm_reward(completions, solution, **kwargs):
    #     # 只在第一次调用时加载模型
        
    #     sentence_encoder = SentenceTransformer(
    #             '/volume/med-train/users/shuyan/lhg/model/all-MiniLM-L6-v2'
    #         )
    #     def extract_answer(text):
    #         match = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    #         return match[-1].strip() if match else text.strip()
    
    #     contents = [extract_answer(c[0]["content"]) for c in completions]
    #     refs = [extract_answer(s) for s in solution]
    #     content=[contents[0],refs[0]]
    #     emb_contents = sentence_encoder.encode(content)
        

    #     sims = util.cos_sim(emb_contents[0], emb_contents[1]).diagonal()
    #     rewards = ((sims + 1) / 2).tolist()

    #     return rewards

    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)
            
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case "general":
                    return Qwen2VLModule.llm_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case "general":
                    return Qwen2VLModule.format_reward_gen
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")

    def multi_round_generate(
        self,
        model,
        processing_class,
        prompt_inputs,
        images_per_sample,
        prompts,
        generation_config,
        max_rounds=10,
    ):
        """
        Multi-round generation with region-crop support.
        Processes one sample at a time (batch_size=1 during generation).

        After multi-round generation completes, rebuilds the FULL input from scratch
        with all cropped images so that log prob computation uses the correct multimodal context.

        Args:
            model: the unwrapped model to call .generate() on
            processing_class: the processor (tokenizer + image_processor)
            prompt_inputs: dict with input_ids, attention_mask, pixel_values, image_grid_thw
            images_per_sample: list of list of PIL images, one list per sample
            prompts: list of message lists (the original conversational prompts)
            generation_config: GenerationConfig object
            max_rounds: maximum number of generation rounds per sample

        Returns:
            all_prompt_inputs: dict with input_ids, attention_mask, pixel_values, image_grid_thw
                              (rebuilt with all images including crops)
            all_completion_ids: list of 1-D tensors, one per sample
        """
        device = prompt_inputs["input_ids"].device
        batch_size = prompt_inputs["input_ids"].size(0)

        box_end_id = processing_class.tokenizer.convert_tokens_to_ids("<|box_end|>")
        eos_id = processing_class.tokenizer.eos_token_id

        # Vision token IDs for inserting crop image tokens into completions
        vision_start_id = processing_class.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        image_pad_id = processing_class.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_end_id = processing_class.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        # Merge size for computing number of image pad tokens
        _ip = getattr(processing_class, "image_processor", None)
        merge_length = getattr(_ip, "merge_size", 2) ** 2 if _ip is not None else 4

        # Modify generation config to stop on box_end_id as well
        multi_round_gen_config = deepcopy(generation_config)
        multi_round_gen_config.eos_token_id = [eos_id, box_end_id]

        all_completion_ids = []
        crop_images_per_sample = []  # Track crop images per sample for pixel_values

        for b in range(batch_size):
            # Get original images for this sample
            sample_images = list(images_per_sample[b])  # copy to avoid mutation
            # Precompute processor-aligned images for bbox cropping
            proc_aligned_images = [_resize_pil_to_processor(img, processing_class) for img in sample_images]

            # Build initial messages for this sample
            sample_prompt = prompts[b]  # original message list
            messages = deepcopy(sample_prompt)

            # System message: get from messages or use empty
            # The messages should already have the system message from the dataset

            sample_completion_ids = []
            sample_crops = []  # Track crops for this sample

            for round_idx in range(max_rounds):
                # Collect all images from messages
                image_inputs = []
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        for item in msg["content"]:
                            if item.get("type") == "image":
                                image_inputs.append(item["image"])
                    elif isinstance(msg.get("content"), str):
                        pass  # text-only message

                # Also include images from user content that uses the {"type": "image"} format
                # without an "image" key (just referencing by position)
                # For the initial prompt, images come from image_inputs; for crops, they're in messages

                # Apply chat template
                text = processing_class.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Remove intermediate assistant delimiters to make it one continuous assistant turn
                delimiter = "<|im_end|>\n<|im_start|>assistant\n"
                first_idx = text.find(delimiter)
                if first_idx != -1:
                    before_first = text[:first_idx + len(delimiter)]
                    after_first = text[first_idx + len(delimiter):]
                    after_first = after_first.replace(delimiter, "")
                    text = before_first + after_first

                # Process inputs
                round_inputs = processing_class(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    return_tensors="pt",
                    padding=True,
                )
                round_inputs = {k: v.to(device) for k, v in round_inputs.items()}

                # Generate
                generated_ids = model.generate(
                    **round_inputs,
                    generation_config=multi_round_gen_config,
                )

                # Trim input to get only generated tokens
                input_len = round_inputs["input_ids"].size(1)
                new_ids = generated_ids[0, input_len:]
                sample_completion_ids.append(new_ids)

                # Check last token
                last_token = new_ids[-1].item()

                if last_token == box_end_id:
                    # Decode to parse bbox
                    response_text = processing_class.tokenizer.decode(new_ids, skip_special_tokens=False)
                    coords = parse_last_bbox(response_text)

                    if coords is not None and len(proc_aligned_images) > 0:
                        # Crop from the first (original) processor-aligned image
                        cropped = crop_2d_image(proc_aligned_images[0], coords)

                        # Compute vision tokens for the cropped image
                        crop_processed = processing_class.image_processor(
                            images=[cropped], return_tensors="pt"
                        )
                        crop_grid_thw = crop_processed["image_grid_thw"]  # (1, 3)
                        num_pads = int(crop_grid_thw.prod(dim=1).item()) // merge_length

                        vision_token_ids = (
                            [vision_start_id]
                            + [image_pad_id] * num_pads
                            + [vision_end_id]
                        )
                        vision_tokens = torch.tensor(
                            vision_token_ids, dtype=torch.long, device=device
                        )
                        sample_completion_ids.append(vision_tokens)
                        sample_crops.append(cropped)

                        # Add assistant's partial response with cropped image to messages
                        messages.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": response_text},
                                {"type": "image", "image": cropped},
                            ]
                        })

                        # Track the cropped image for later
                        sample_images.append(cropped)
                        continue
                    else:
                        # No valid bbox found, stop
                        break
                else:
                    # EOS or max_new_tokens reached, stop
                    break

            # Concatenate all completion IDs from all rounds
            if sample_completion_ids:
                all_completion_ids.append(torch.cat(sample_completion_ids, dim=0))
            else:
                # Empty completion
                all_completion_ids.append(torch.tensor([], dtype=torch.long, device=device))

            crop_images_per_sample.append(sample_crops)

        # ===== Rebuild prompt inputs (original prompt only, no assistant text) =====
        # The completion_ids now include vision tokens for crops, so the prompt
        # should only contain the original prompt (system + user messages).
        rebuilt_prompts_text = []
        for b in range(batch_size):
            text = processing_class.apply_chat_template(
                prompts[b], tokenize=False, add_generation_prompt=True,
            )
            rebuilt_prompts_text.append(text)

        # Process prompt with original images only
        all_original_images = [img for imgs in images_per_sample for img in imgs]
        rebuilt_inputs = processing_class(
            text=rebuilt_prompts_text,
            images=all_original_images if all_original_images else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        rebuilt_inputs = {k: v.to(device) for k, v in rebuilt_inputs.items()}

        # If there are crops, rebuild pixel_values and image_grid_thw with ALL images
        # ordered correctly: for each sample, original images first, then crops
        has_crops = any(len(crops) > 0 for crops in crop_images_per_sample)
        if has_crops:
            all_images_ordered = []
            for b in range(batch_size):
                for img in images_per_sample[b]:
                    all_images_ordered.append(img)
                for crop in crop_images_per_sample[b]:
                    all_images_ordered.append(crop)
            if all_images_ordered:
                full_img_processed = processing_class.image_processor(
                    images=all_images_ordered, return_tensors="pt"
                )
                rebuilt_inputs["pixel_values"] = full_img_processed["pixel_values"].to(device)
                rebuilt_inputs["image_grid_thw"] = full_img_processed["image_grid_thw"].to(device)

        # Pad completion_ids to same length
        max_comp_len = max(c.size(0) for c in all_completion_ids) if all_completion_ids else 0
        if max_comp_len == 0:
            padded_completion_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        else:
            padded_completion_ids = torch.full(
                (batch_size, max_comp_len),
                processing_class.tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            for b, comp_ids in enumerate(all_completion_ids):
                padded_completion_ids[b, :comp_ids.size(0)] = comp_ids

        return rebuilt_inputs, padded_completion_ids
