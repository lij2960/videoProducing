"""
translator.py - 中文提示词翻译 + 质量词增强
优先用本地词典快速翻译，无网络依赖
"""

# 质量增强词（追加到每个提示词后面）
QUALITY_SUFFIX = (
    "masterpiece, best quality, highly detailed, sharp focus, "
    "cinematic lighting, 8k uhd, photorealistic"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, deformed, ugly, "
    "bad anatomy, watermark, text, signature, cropped"
)


def enhance_prompt(prompt: str) -> str:
    """给提示词追加质量词"""
    return f"{prompt}, {QUALITY_SUFFIX}"


def translate_and_enhance(prompts: list[str], use_api: bool = True) -> list[str]:
    """
    将中文提示词翻译为英文并增强质量

    Args:
        prompts: 中文提示词列表
        use_api: 是否尝试调用翻译 API（需要网络）

    Returns:
        增强后的英文提示词列表
    """
    results = []
    for prompt in prompts:
        translated = _translate(prompt, use_api)
        enhanced = enhance_prompt(translated)
        print(f"  原文: {prompt}")
        print(f"  译文: {enhanced}\n")
        results.append(enhanced)
    return results


def _translate(text: str, use_api: bool) -> str:
    """翻译单条文本，优先 API，失败则用本地映射"""
    if _is_english(text):
        return text

    if use_api:
        try:
            return _translate_via_helsinki(text)
        except Exception as e:
            print(f"  [翻译 API 失败，使用本地映射] {e}")

    return _local_translate(text)


def _is_english(text: str) -> bool:
    """判断是否已经是英文"""
    return all(ord(c) < 128 for c in text.replace(" ", ""))


def _translate_via_helsinki(text: str) -> str:
    """使用 Helsinki-NLP 本地翻译模型（无需 API key，但需下载模型）"""
    from transformers import pipeline as hf_pipeline
    translator = hf_pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-zh-en",
        max_length=512,
    )
    result = translator(text)[0]["translation_text"]
    return result


def _local_translate(text: str) -> str:
    """
    关键词替换兜底方案，按长词优先匹配避免误替换
    """
    mapping = {
        # 长词优先（避免被短词提前替换）
        "一只猫": "a cat",
        "一只狗": "a dog",
        "夕阳西下": "sunset, golden hour",
        "灯火通明": "glowing with lights",
        "星空下": "under the starry night sky",
        "草地上": "in the green meadow",
        "天空变成": "the sky turns",
        "城市灯火": "city lights",
        # 常用词
        "猫": "cat",
        "狗": "dog",
        "草地": "green meadow",
        "玩耍": "playing",
        "夕阳": "sunset",
        "天空": "sky",
        "橙红色": "orange and red",
        "橙色": "orange",
        "红色": "red",
        "蓝色": "blue",
        "星空": "starry night sky",
        "星星": "stars",
        "城市": "city",
        "森林": "forest",
        "大海": "ocean",
        "海边": "seaside",
        "山": "mountain",
        "雪": "snow",
        "雨": "rain",
        "花": "flowers",
        "树": "trees",
        "人": "person",
        "女孩": "girl",
        "男孩": "boy",
        "建筑": "architecture",
        "古风": "ancient Chinese style",
        "科幻": "sci-fi",
        "梦幻": "dreamy fantasy",
        "黄昏": "dusk, golden hour",
        "清晨": "early morning, dawn",
        "夜晚": "at night",
        "夜": "night",
        "白天": "daytime",
        "阳光": "sunlight",
        "云": "clouds",
        "风": "wind",
        "河流": "river",
        "湖": "lake",
        "沙漠": "desert",
        "街道": "street",
        "公园": "park",
        "下": "under",
        "上": "on",
        "中": "in",
        "里": "inside",
        "，": ", ",
        "。": ". ",
    }

    # 按键长度降序排列，长词优先匹配
    sorted_mapping = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)

    result = text
    for zh, en in sorted_mapping:
        result = result.replace(zh, en)

    # 清理残留中文字符，替换为空格
    import re
    result = re.sub(r'[\u4e00-\u9fff]+', ' ', result)
    result = re.sub(r'\s+', ' ', result).strip()

    return result
