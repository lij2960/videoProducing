"""
controlnet_generator.py - 使用 ControlNet (Canny) 保持跨场景主体一致性

原理：
  1. 用普通 text2img 生成第一帧（参考帧）
  2. 对参考帧提取 Canny 边缘图
  3. 后续场景用 ControlNet 以边缘图为条件生成，主体轮廓保持一致
"""
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from translator import NEGATIVE_PROMPT


def extract_canny(image: Image.Image, low_threshold: int = 50, high_threshold: int = 150) -> Image.Image:
    """从图片提取 Canny 边缘图"""
    img_array = np.array(image.convert("RGB"))
    edges = cv2.Canny(img_array, low_threshold, high_threshold)
    # ControlNet 需要 3 通道
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def load_controlnet_pipeline(
    base_model_id: str = "Lykon/dreamshaper-8",
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny",
) -> StableDiffusionControlNetPipeline:
    """加载 ControlNet pipeline"""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    print("加载 ControlNet 模型...")
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=dtype)

    print("加载 Stable Diffusion 模型...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_frames_with_controlnet(
    prompts: list[str],
    output_dir: str = "frames",
    width: int = 512,
    height: int = 768,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 0.8,
    canny_low: int = 50,
    canny_high: int = 150,
    base_model_id: str = "Lykon/dreamshaper-8",
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny",
    base_pipe=None,
) -> list[str]:
    """
    生成关键帧，第一帧自由生成，后续帧用第一帧的 Canny 边缘约束主体形状

    Args:
        prompts: 已翻译增强的英文提示词列表
        controlnet_conditioning_scale: ControlNet 影响强度（0~1），越高越贴近轮廓
        canny_low/canny_high: Canny 边缘检测阈值
        base_pipe: 已有的普通 pipeline，用于生成第一帧
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_paths = []

    # --- 第一帧：用普通 text2img 自由生成 ---
    print(f"\n生成参考帧（场景 1）: {prompts[0][:80]}")
    if base_pipe is None:
        from generator import load_pipeline
        base_pipe = load_pipeline(base_model_id)

    generator = torch.Generator().manual_seed(42)
    ref_image: Image.Image = base_pipe(
        prompt=prompts[0],
        negative_prompt=NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    ref_path = out_path / "frame_0000.png"
    ref_image.save(ref_path)
    frame_paths.append(str(ref_path))
    print(f"  参考帧已保存: {ref_path}")

    if len(prompts) == 1:
        return frame_paths

    # --- 提取参考帧的 Canny 边缘 ---
    canny_image = extract_canny(ref_image, canny_low, canny_high)
    canny_path = out_path / "canny_reference.png"
    canny_image.save(canny_path)
    print(f"  Canny 边缘图已保存: {canny_path}")

    # --- 加载 ControlNet pipeline ---
    print("\n加载 ControlNet pipeline...")
    cn_pipe = load_controlnet_pipeline(base_model_id, controlnet_model_id)

    # --- 后续帧：用 ControlNet 约束主体轮廓 ---
    for i, prompt in enumerate(tqdm(prompts[1:], desc="生成场景"), start=1):
        print(f"\n生成场景 {i+1}/{len(prompts)}: {prompt[:80]}")
        generator = torch.Generator().manual_seed(i * 100)
        image: Image.Image = cn_pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=canny_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images[0]

        frame_file = out_path / f"frame_{i:04d}.png"
        image.save(frame_file)
        frame_paths.append(str(frame_file))

    print(f"\n共生成 {len(frame_paths)} 帧，保存至 {output_dir}/")
    return frame_paths
