"""
interpolator.py - 使用 img2img 在关键帧之间生成过渡帧，让画面动起来
"""
import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def load_img2img_pipeline(model_id: str, base_pipe=None):
    """
    加载 img2img pipeline
    如果已有 text2img pipe，直接复用权重，不重复加载模型
    """
    if base_pipe is not None:
        pipe = StableDiffusionImg2ImgPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(base_pipe.device)
        return pipe

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None
    ).to(device)
    return pipe


def blend_images(img_a: Image.Image, img_b: Image.Image, alpha: float) -> Image.Image:
    """线性混合两张图片，alpha=0 返回 img_a，alpha=1 返回 img_b"""
    arr_a = np.array(img_a).astype(float)
    arr_b = np.array(img_b).astype(float)
    blended = (arr_a * (1 - alpha) + arr_b * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def generate_transition_frames(
    keyframe_a: Image.Image,
    keyframe_b: Image.Image,
    prompt_a: str,
    prompt_b: str,
    pipe,
    num_transition_frames: int = 8,
    num_inference_steps: int = 20,
    output_dir: str = "frames",
    start_idx: int = 0,
) -> list[str]:
    """
    在两张关键帧之间生成过渡帧

    原理：
    - 将 img_a 和 img_b 按比例混合作为 init_image
    - strength 从小到大，让生成结果逐渐偏向 img_b 的风格
    - prompt 也按比例从 prompt_a 过渡到 prompt_b（通过 negative prompt 控制）
    """
    from translator import NEGATIVE_PROMPT

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_paths = []

    for i in range(num_transition_frames):
        alpha = (i + 1) / (num_transition_frames + 1)  # 0 到 1 之间均匀分布

        # 混合两帧作为初始图
        init_image = blend_images(keyframe_a, keyframe_b, alpha)

        # strength 控制改变程度：越小越接近原图，越大越自由发挥
        # 过渡帧用较小的 strength 保持连贯性
        strength = 0.25 + alpha * 0.2  # 0.25 ~ 0.45

        # 提示词按 alpha 混合：前半段用 prompt_a，后半段用 prompt_b
        prompt = prompt_a if alpha < 0.5 else prompt_b

        generator = torch.Generator().manual_seed(start_idx + i)
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        frame_file = out_path / f"frame_{start_idx + i:04d}.png"
        result.save(frame_file)
        frame_paths.append(str(frame_file))

    return frame_paths


def interpolate_keyframes(
    keyframe_paths: list[str],
    prompts: list[str],
    output_dir: str = "frames",
    num_transition_frames: int = 8,
    num_inference_steps: int = 20,
    model_id: str = "Lykon/dreamshaper-8",
    base_pipe=None,
) -> list[str]:
    """
    对所有关键帧两两之间插入过渡帧，返回完整帧序列路径

    Args:
        keyframe_paths: 关键帧图片路径列表
        prompts: 对应的提示词列表（已翻译增强）
        output_dir: 输出目录
        num_transition_frames: 每两帧之间插入几帧（越多越流畅，越慢）
        num_inference_steps: img2img 推理步数
        model_id: 模型 ID
        base_pipe: 已有的 text2img pipeline，复用权重

    Returns:
        包含关键帧 + 过渡帧的完整路径列表（按顺序）
    """
    print("\n加载 img2img pipeline...")
    img2img_pipe = load_img2img_pipeline(model_id, base_pipe)

    all_frames = []
    frame_counter = len(keyframe_paths) * 100  # 过渡帧用不同编号避免覆盖

    print(f"\n开始生成过渡帧（每段 {num_transition_frames} 帧）...")

    for i in tqdm(range(len(keyframe_paths) - 1), desc="插帧进度"):
        img_a = Image.open(keyframe_paths[i]).convert("RGB")
        img_b = Image.open(keyframe_paths[i + 1]).convert("RGB")

        # 确保尺寸一致
        img_b = img_b.resize(img_a.size, Image.LANCZOS)

        # 加入当前关键帧
        all_frames.append(keyframe_paths[i])

        # 生成过渡帧
        transition = generate_transition_frames(
            keyframe_a=img_a,
            keyframe_b=img_b,
            prompt_a=prompts[i] if i < len(prompts) else prompts[-1],
            prompt_b=prompts[i + 1] if i + 1 < len(prompts) else prompts[-1],
            pipe=img2img_pipe,
            num_transition_frames=num_transition_frames,
            num_inference_steps=num_inference_steps,
            output_dir=output_dir,
            start_idx=frame_counter,
        )
        all_frames.extend(transition)
        frame_counter += num_transition_frames

    # 加入最后一帧
    all_frames.append(keyframe_paths[-1])

    print(f"插帧完成，总帧数: {len(all_frames)}")
    return all_frames
