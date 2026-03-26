"""
ipadapter_generator.py - 使用 IP-Adapter 保持跨场景主体外观一致

原理：
  IP-Adapter 将参考图的图像特征（外观、颜色、纹理）编码后注入 UNet cross-attention，
  后续场景生成时会参考第一帧的视觉特征，猫的花色/品种保持一致。

需要提前下载 IP-Adapter 权重：
  https://huggingface.co/h94/IP-Adapter
  下载 models/ip-adapter_sd15.bin 放到 ./ip_adapter_weights/ 目录
"""
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from translator import NEGATIVE_PROMPT

IP_ADAPTER_WEIGHTS_DIR = "./ip_adapter_weights"
IP_ADAPTER_SUBFOLDER = "models"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.bin"


def load_pipeline_with_ipadapter(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
) -> StableDiffusionPipeline:
    """
    加载带 IP-Adapter 的 pipeline
    注意：IP-Adapter 官方权重基于 SD1.5，base_model 需用 SD1.5 系列
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    weights_path = Path(IP_ADAPTER_WEIGHTS_DIR)
    weight_file = weights_path / IP_ADAPTER_SUBFOLDER / IP_ADAPTER_WEIGHT_NAME
    encoder_config = weights_path / IP_ADAPTER_SUBFOLDER / "image_encoder" / "config.json"
    encoder_weights = weights_path / IP_ADAPTER_SUBFOLDER / "image_encoder" / "pytorch_model.bin"

    if not (weight_file.exists() and encoder_config.exists() and encoder_weights.exists()):
        print(f"\n[提示] 未找到 IP-Adapter 权重，正在自动下载...")
        _download_ip_adapter_weights(weights_path)

    pipe.load_ip_adapter(
        str(weights_path),
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT_NAME,
    )
    print("IP-Adapter 加载成功")
    return pipe


def _download_ip_adapter_weights(save_dir: Path):
    """从 HuggingFace 下载 IP-Adapter 权重和 image_encoder"""
    from huggingface_hub import hf_hub_download
    save_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        "models/ip-adapter_sd15.bin",
        "models/image_encoder/config.json",
        "models/image_encoder/pytorch_model.bin",
    ]

    for filename in files_to_download:
        dest = save_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"  已存在，跳过: {filename}")
            continue
        print(f"  下载: {filename}")
        hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename=filename,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False,
        )

    print("下载完成")


def generate_frames_with_ipadapter(
    prompts: list[str],
    output_dir: str = "frames",
    width: int = 512,
    height: int = 768,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    ip_adapter_scale: float = 0.6,
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    base_pipe=None,
) -> list[str]:
    """
    生成关键帧：第一帧自由生成作为参考，后续帧用 IP-Adapter 保持外观一致

    Args:
        prompts: 已翻译增强的英文提示词列表
        ip_adapter_scale: IP-Adapter 影响强度（0~1）
            - 越高越像参考图外观，但场景变化越小
            - 推荐 0.5~0.7，平衡一致性和场景多样性
        base_model_id: 必须是 SD1.5 系列（IP-Adapter 官方权重基于 SD1.5）
        base_pipe: 已有的普通 pipeline，会被替换为带 IP-Adapter 的版本
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 加载带 IP-Adapter 的 pipeline
    pipe = load_pipeline_with_ipadapter(base_model_id)
    frame_paths = []

    # --- 第一帧：不使用 IP-Adapter，自由生成参考帧 ---
    print(f"\n生成参考帧（场景 1）: {prompts[0][:80]}")
    pipe.set_ip_adapter_scale(0.0)  # 关闭 IP-Adapter，自由生成
    generator = torch.Generator().manual_seed(42)
    ref_image: Image.Image = pipe(
        prompt=prompts[0],
        negative_prompt=NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ip_adapter_image=Image.new("RGB", (width, height)),  # 占位图
        generator=generator,
    ).images[0]

    ref_path = out_path / "frame_0000.png"
    ref_image.save(ref_path)
    frame_paths.append(str(ref_path))
    print(f"  参考帧已保存: {ref_path}")

    if len(prompts) == 1:
        return frame_paths

    # --- 后续帧：开启 IP-Adapter，以参考帧外观为条件 ---
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    print(f"\n启用 IP-Adapter（scale={ip_adapter_scale}），以参考帧约束后续场景外观...")

    for i, prompt in enumerate(tqdm(prompts[1:], desc="生成场景"), start=1):
        print(f"\n生成场景 {i+1}/{len(prompts)}: {prompt[:80]}")
        generator = torch.Generator().manual_seed(i * 100)
        image: Image.Image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_image=ref_image,   # 参考第一帧的外观
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        frame_file = out_path / f"frame_{i:04d}.png"
        image.save(frame_file)
        frame_paths.append(str(frame_file))

    print(f"\n共生成 {len(frame_paths)} 帧，保存至 {output_dir}/")
    return frame_paths
