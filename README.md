# DyCo-RL: Dynamic Cross-Modal Coordination for Visual Reasoning


<font size=4><div align='center'>[[📄 Tech Report](https://arxiv.org/abs/2504.07615)] </div></font>

<div align="center">
<img src="./fig/pipline.png" width="900"/>
<div>
  <font size=4>
    <p>🎉  <b>We propose DyCo-RL, a novel approach thatembeds dynamic, token-level cross-modal coordination into the RLVR framework.</b></p>
  </font>
</div>
</div>

Reinforcement Learning with Verifiable Rewards (RLVR) has recently become a strong way to improve visual reasoning in Multimodal Large Language Models (MLLMs), but most existing methods still treat all tokens equally during optimization. This ignores an important fact: different tokens actually serve different roles—some need to focus more on images, while others depend more on text context. Through token-level attention analysis, we find that many reasoning errors come from cross-modal misalignment, where visual-related tokens fail to fully attend to images and text-related tokens underuse surrounding context. We further verify this through intervention experiments, showing that this mismatch directly leads to performance drops. Based on these insights, we introduce DyCo-RL, a simple and plug-and-play method that dynamically adjusts cross-modal coordination at the token level during RLVR training. Specifically, we estimate each token’s functional role using Fisher–Rao geodesic distance between modality-specific attention patterns, and use a modality attention ratio as a signal to reweight token-level advantages in policy optimization. DyCo-RL is compatible with multiple RLVR algorithms, including GRPO, GSPO, DAPO, and SAPO, and brings consistent gains when applied to Qwen2.5-VL-3B/7B, improving performance by an average of +1.5% across both visual perception and mathematical reasoning benchmarks.

![image](./assets/performance.png)



## 🛠️ Setup

```bash
conda create -n vlm-r1 python=3.10
conda activate vlm-r1
bash setup.sh
```

## 💪🏻 Training

### Referring Expression Comprehension (REC)

#### 📚 GRPO

1. Download the [COCO Train2014 image](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip) and unzip it, and we refer to the image dir as `<your_image_root>`.
2. Download the [RefCOCO/+/g and LISA-Grounding Annotation files](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip) and unzip it (LISA-Grounding is used for out-of-domain evaluation).
3. Change the `data_paths` and `image_folders` in the [run_scripts/run_grpo_rec.sh](run_scripts/run_grpo_rec.sh) file.

```bash
# These jsonl files are included in the annotation files at step 2.
# Note: please use jsonl files instead of json files.
data_paths="path/to/refcoco_train.jsonl:path/to/refcocop_train.jsonl:path/to/refcocog_train.jsonl"
image_folders="path/to/coco:path/to/coco:path/to/coco"
```


### For your own data

<div style="text-align: justify;">

We support data loading the jsonl data of this format in [`src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`](src/open-r1-multimodal/src/open_r1/grpo_jsonl.py). Please note that you may need to use different reward functions for your specialized tasks. Welcome to PR to add your own reward functions or share any other interesting findings!

</div>

The jsonl has the format as follows:

```json
{
  "id": 1,
  "image": "Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16885.png",
  "conversations": [
    {"from": "human", "value": "<image>What number of purple metallic balls are there?"},
    {"from": "gpt", "value": "0"}
  ]
}
```

If you want to use multi-image input, you can use the following format:

```json
{
  "id": 1,
  "image": ["Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16885.png", "Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16886.png"],
  "conversations": [
    {"from": "human", "value": "<image><image>What number of purple metallic balls in total within the two images?"},
    {"from": "gpt", "value": "3"}
  ]
}
```

> [!NOTE]
> The image path in the jsonl file should be relative to the image folder specified in `--image_folders`. The absolute path of the input image is constructed as `os.path.join(image_folder, data['image'])`. For example:

- If your jsonl has `"image": "folder1/image1.jpg"`
- And you specify `--image_folders "/path/to/images/"`
- The full image path will be `/path/to/images/folder1/image1.jpg`

Multiple data files and image folders can be specified using ":" as a separator:

```bash
--data_file_paths /path/to/data1.jsonl:/path/to/data2.jsonl \
--image_folders /path/to/images1/:/path/to/images2/
```

The script can be run like this:

```bash
# You could refer to the run_grpo_rec.sh for the example
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
  src/open_r1/grpo_jsonl.py \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \
    --data_file_paths /path/to/your/data.jsonl \ # can be multiple, separated by ":"
    --image_folders /path/to/your/image/folder \ # can be multiple, separated by ":"
    ...
```

<div style="text-align: justify;">

## 📊 Evaluation

![image](./assets/data2.png)

1. Download the provided [LISA-Grounding images](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/lisa-test.zip).

```bash
cd ./src/eval

# Remember to change the model path, image root, and annotation path in the script
torchrun --nproc_per_node=X test_rec_r1.py # for GRPO. 'X' is the number of GPUs you have.
torchrun --nproc_per_node=X test_rec_baseline.py # for SFT.
```

## 🤝 Acknowledgements

We would like to express our sincere gratitude to [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V), [RefCOCO](https://github.com/lichengunc/refer), [RefGTA](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [OVDEval](https://github.com/om-ai-lab/OVDEval), [GUI-Testing-Arena](https://huggingface.co/datasets/songjah/GTArena-UI-Defects), and [LISA](https://github.com/dvlab-research/LISA) for providing open-source resources that contributed to the development of this project.

## ⭐️ Citation

If you find this project useful, welcome to cite us.

```bib
@article{shen2025vlm,
  title={Vlm-r1: A stable and generalizable r1-style large vision-language model},
  author={Shen, Haozhan and Liu, Peng and Li, Jingcheng and Fang, Chunxin and Ma, Yibo and Liao, Jiajia and Shen, Qiaoli and Zhang, Zilun and Zhao, Kangjia and Zhang, Qianqian and Xu, Ruochen and Zhao, Tiancheng },
  journal={arXiv preprint arXiv:2504.07615},
  year={2025}
}
```
