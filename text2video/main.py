"""
main.py - 文字生成视频入口

用法:
    python main.py --text "一只猫在草地上玩耍。夕阳西下，天空变成橙红色。"
    python main.py --file prompts.txt --output my_video.mp4 --interpolate
"""
import argparse
from generator import load_pipeline, generate_frames
from composer import frames_to_video
from translator import translate_and_enhance


def parse_prompts(text: str) -> list[str]:
    """将输入文字按句子分割为场景提示词列表"""
    import re
    sentences = re.split(r'[。！？.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="文字生成视频")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="直接输入文字描述（句子间用句号分隔）")
    group.add_argument("--file", type=str, help="从文件读取提示词，每行一个场景")

    parser.add_argument("--output", type=str, default="output.mp4", help="输出视频路径")
    parser.add_argument("--frames-dir", type=str, default="frames", help="图片帧保存目录")
    parser.add_argument("--frames-per-prompt", type=int, default=1, help="每个场景生成几张关键帧")
    parser.add_argument("--duration", type=float, default=0.1, help="插帧模式下每帧显示时长（秒）")
    parser.add_argument("--fps", type=int, default=24, help="视频帧率")
    parser.add_argument("--steps", type=int, default=40, help="text2img 推理步数")
    parser.add_argument("--interp-steps", type=int, default=20, help="img2img 插帧推理步数")
    parser.add_argument("--interp-frames", type=int, default=8, help="每两个关键帧之间插入几帧")
    parser.add_argument("--width", type=int, default=512, help="图片宽度")
    parser.add_argument("--height", type=int, default=768, help="图片高度")
    parser.add_argument("--no-translate", action="store_true", help="禁用自动翻译")
    parser.add_argument("--no-interpolate", action="store_true", help="禁用插帧，退回静态切换模式")
    parser.add_argument(
        "--model",
        type=str,
        default="Lykon/dreamshaper-8",
        help="Hugging Face 模型 ID",
    )
    args = parser.parse_args()

    # 1. 解析提示词
    if args.text:
        prompts = parse_prompts(args.text)
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        print("错误：未找到有效的提示词")
        return

    print(f"共 {len(prompts)} 个场景:")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {p}")

    # 2. 翻译增强提示词（提前做，插帧时也需要用）
    if not args.no_translate:
        print("\n正在翻译和增强提示词...")
        enhanced_prompts = translate_and_enhance(prompts)
    else:
        enhanced_prompts = prompts

    # 3. 加载模型，生成关键帧
    pipe = load_pipeline(args.model)
    keyframe_paths = generate_frames(
        prompts=prompts,
        output_dir=args.frames_dir,
        frames_per_prompt=args.frames_per_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        translate=False,          # 已经翻译过了，不重复翻译
        pipe=pipe,
    )

    # 4. img2img 插帧
    if not args.no_interpolate and len(keyframe_paths) > 1:
        from interpolator import interpolate_keyframes
        all_frame_paths = interpolate_keyframes(
            keyframe_paths=keyframe_paths,
            prompts=enhanced_prompts,
            output_dir=args.frames_dir,
            num_transition_frames=args.interp_frames,
            num_inference_steps=args.interp_steps,
            model_id=args.model,
            base_pipe=pipe,
        )
        duration = args.duration  # 插帧后每帧时长短，靠帧数堆流畅度
        crossfade = 0.0           # 插帧模式不需要额外淡入淡出
    else:
        all_frame_paths = keyframe_paths
        duration = 1.5
        crossfade = 0.4

    # 5. 合成视频
    frames_to_video(
        frame_paths=all_frame_paths,
        output_path=args.output,
        fps=args.fps,
        duration_per_frame=duration,
        crossfade_duration=crossfade,
    )

    print(f"\n完成！视频已保存至 {args.output}")


if __name__ == "__main__":
    main()
