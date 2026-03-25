"""
generator.py - 使用 Stable Diffusion 根据文字提示生成图片帧
"""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from translator import translate_and_enhance, NEGATIVE_PROMPT


def load_pipeline(model_id: str = "Lykon/dreamshaper-8") -> StableDiffusionPipeline:
    """
    加载 Stable Diffusion 模型
    默认使用 dreamshaper-8，效果比 v1.5 好很多
    其他推荐模型:
      - "SG161222/Realistic_Vision_V5.1_noVAE"  写实风格
      - "runwayml/stable-diffusion-v1-5"         基础模型
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )

    # 换用 DPM++ 调度器，同等步数下质量更好
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_frames(
    prompts: list[str],
    output_dir: str = "frames",
    frames_per_prompt: int = 3,
    width: int = 512,
    height: int = 768,       # 默认竖版，更适合场景展示
    num_inference_steps: int = 40,   # 提高到 40 步
    guidance_scale: float = 8.0,     # 稍微提高引导强度
    translate: bool = True,          # 自动翻译中文
    pipe: StableDiffusionPipeline = None,
) -> list[str]:
    """
    根据提示词列表生成图片帧

    Args:
        prompts: 文字提示词列表（支持中文，自动翻译）
        output_dir: 图片输出目录
        frames_per_prompt: 每个提示词生成几帧
        width/height: 图片尺寸
        num_inference_steps: 推理步数
        guidance_scale: 提示词引导强度（7-12 之间效果好）
        translate: 是否自动翻译并增强提示词
        pipe: 已加载的 pipeline
    """
    if pipe is None:
        pipe = load_pipeline()

    # 翻译 + 增强提示词
    if translate:
        print("\n正在翻译和增强提示词...")
        enhanced_prompts = translate_and_enhance(prompts)
    else:
        enhanced_prompts = prompts

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    frame_idx = 0

    for i, prompt in enumerate(tqdm(enhanced_prompts, desc="生成场景")):
        print(f"生成场景 {i+1}/{len(enhanced_prompts)}: {prompt[:80]}")

        for j in range(frames_per_prompt):
            generator = torch.Generator().manual_seed(i * 100 + j)
            image: Image.Image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,   # 加入负面提示词
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            frame_file = out_path / f"frame_{frame_idx:04d}.png"
            image.save(frame_file)
            frame_paths.append(str(frame_file))
            frame_idx += 1

    print(f"\n共生成 {len(frame_paths)} 帧，保存至 {output_dir}/")
    return frame_paths
