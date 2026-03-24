"""
composer.py - 将图片帧合成为视频，支持淡入淡出过渡
"""
from moviepy.editor import ImageClip, concatenate_videoclips, VideoFileClip
from pathlib import Path


def frames_to_video(
    frame_paths: list[str],
    output_path: str = "output.mp4",
    fps: int = 24,
    duration_per_frame: float = 1.0,
    crossfade_duration: float = 0.5,
) -> str:
    """
    将图片帧列表合成为视频

    Args:
        frame_paths: 图片路径列表
        output_path: 输出视频路径
        fps: 帧率
        duration_per_frame: 每张图片显示时长（秒）
        crossfade_duration: 淡入淡出过渡时长（秒），0 表示无过渡

    Returns:
        输出视频路径
    """
    if not frame_paths:
        raise ValueError("frame_paths 不能为空")

    clips = []
    for path in frame_paths:
        clip = ImageClip(path).set_duration(duration_per_frame)
        clips.append(clip)

    if crossfade_duration > 0 and len(clips) > 1:
        # 添加淡入淡出过渡
        faded = [clips[0].crossfadeout(crossfade_duration)]
        for clip in clips[1:-1]:
            faded.append(clip.crossfadein(crossfade_duration).crossfadeout(crossfade_duration))
        faded.append(clips[-1].crossfadein(crossfade_duration))
        final = concatenate_videoclips(faded, padding=-crossfade_duration, method="compose")
    else:
        final = concatenate_videoclips(clips, method="compose")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(output_path, fps=fps, codec="libx264", audio=False)
    print(f"视频已保存: {output_path}")
    return output_path
