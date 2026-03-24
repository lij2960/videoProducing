# text2video

用文字描述生成视频。输入一段中文文字，自动拆分场景、AI 生成图片帧、插帧过渡，最终合成 MP4 视频。

## 原理

```
文字输入 → 按句子拆分场景 → 中文翻译+质量增强 → Stable Diffusion 生成关键帧
       → img2img 插帧（让画面动起来）→ moviepy 合成视频
```

## 项目结构

```
text2video/
├── main.py          # 入口
├── generator.py     # text2img 生成关键帧
├── interpolator.py  # img2img 插帧
├── translator.py    # 中文翻译 + 提示词增强
├── composer.py      # 合成视频
└── requirements.txt
```

## 安装

Python 3.9+，建议使用虚拟环境。

```bash
# 安装 PyTorch
python -m pip install torch torchvision

# 安装其余依赖
python -m pip install diffusers==0.25.1 transformers==4.38.0 huggingface_hub==0.21.4
python -m pip install accelerate moviepy==1.0.3 Pillow tqdm "numpy<2"
```

## 快速开始

```bash
cd text2video
python main.py --text "一只猫在草地上玩耍。夕阳西下，天空变成橙红色。星空下，城市灯火通明。"
```

首次运行会自动从 Hugging Face 下载模型（约 4GB），需要一点时间。

生成的视频默认保存为 `output.mp4`，关键帧图片保存在 `frames/` 目录。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--text` | - | 直接输入文字，句号分隔场景（与 `--file` 二选一） |
| `--file` | - | 从文件读取提示词，每行一个场景 |
| `--output` | `output.mp4` | 输出视频路径 |
| `--frames-dir` | `frames` | 关键帧保存目录 |
| `--model` | `Lykon/dreamshaper-8` | Hugging Face 模型 ID |
| `--steps` | `40` | text2img 推理步数，越高质量越好越慢 |
| `--interp-frames` | `8` | 每两个关键帧之间插入几帧，越多越流畅 |
| `--interp-steps` | `20` | img2img 插帧推理步数 |
| `--width` | `512` | 图片宽度 |
| `--height` | `768` | 图片高度 |
| `--fps` | `24` | 视频帧率 |
| `--no-translate` | - | 禁用自动翻译（提示词已是英文时使用） |
| `--no-interpolate` | - | 禁用插帧，退回静态切换模式 |

## 使用示例

从文件读取场景：
```bash
# prompts.txt 每行一个场景
python main.py --file prompts.txt --output my_video.mp4
```

更流畅的插帧（更慢）：
```bash
python main.py --text "..." --interp-frames 12
```

快速预览（质量低但快）：
```bash
python main.py --text "..." --steps 20 --interp-frames 4 --interp-steps 10
```

使用写实风格模型：
```bash
python main.py --text "..." --model "SG161222/Realistic_Vision_V5.1_noVAE"
```

英文提示词直接输入（跳过翻译）：
```bash
python main.py --text "a cat playing in the meadow. golden hour sunset." --no-translate
```

## 推荐模型

| 模型 | 风格 | HF ID |
|------|------|-------|
| Dreamshaper 8（默认） | 通用，画质好 | `Lykon/dreamshaper-8` |
| Realistic Vision | 写实摄影风 | `SG161222/Realistic_Vision_V5.1_noVAE` |
| Stable Diffusion 1.5 | 基础模型 | `runwayml/stable-diffusion-v1-5` |

## 性能参考

| 设备 | 每帧耗时（约） |
|------|--------------|
| CPU | 3~10 分钟 |
| Mac M 系列（MPS） | 30~60 秒 |
| NVIDIA GPU（CUDA） | 5~15 秒 |

CPU 跑比较慢，建议先用少量场景测试效果：
```bash
python main.py --text "夕阳西下。" --steps 20 --interp-frames 4
```

## 注意事项

- 中文提示词会自动翻译成英文（优先用 Helsinki-NLP 本地模型，失败则用内置词典兜底）
- SD 模型对英文理解更好，英文提示词效果通常优于中文
- 图片尺寸建议保持 512 的倍数（如 512x512、512x768）
- 生成的帧会保留在 `frames/` 目录，重新运行会覆盖
