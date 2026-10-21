import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from pydantic import BaseModel

from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64

# Optional video processing dependency. qwen-vl-utils selects and imports its
# decoder lazily when a video is actually processed.
fetch_video, _has_qwen_vl = optional_import("qwen_vl_utils", "fetch_video")


def _clamp_nframes(video_arg: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp a fixed ``nframes`` request to the clip length.

    qwen_vl_utils' ``smart_nframes`` raises when ``nframes`` exceeds the clip's
    frame count (it only clamps in the ``fps`` branch), and ``fetch_video`` then
    falls back to torchvision's removed ``io.read_video``. Round down to an even
    count because ``smart_nframes`` rounds to a multiple of 2.
    """
    if "nframes" not in video_arg or not isinstance(video_arg["video"], str):
        return video_arg
    import cv2

    path = video_arg["video"]
    cap = cv2.VideoCapture(path[7:] if path.startswith("file://") else path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if total_frames >= 2 and video_arg["nframes"] > total_frames:
        video_arg["nframes"] = total_frames // 2 * 2
    return video_arg


class ChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: Any


class ChatVideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: Any
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ChatAudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    url: Any


ChatContent = Union[ChatTextContent, ChatImageContent, ChatVideoContent, ChatAudioContent]


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: List[ChatContent]


class ChatMessages(BaseModel):
    messages: List[ChatMessage]

    def extract_media(self):
        images = []
        videos = []
        audios = []

        for message in self.messages:
            for content in message.content:
                if content.type == "image":
                    images.append(content.url)
                elif content.type == "video":
                    videos.append(content.url)
                elif content.type == "audio":
                    audios.append(content.url)

        return images, videos, audios

    def to_hf_messages(
        self,
        video_kwargs: Optional[Dict[str, str]] = None,
        image_kwargs: Optional[Dict[str, str]] = None,
    ):
        if video_kwargs is None:
            video_kwargs = {}
        if image_kwargs is None:
            image_kwargs = {}
        _num_frames = video_kwargs.get("nframes", 32)  # noqa: F841
        hf_messages = []
        for message in self.messages:
            hf_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    hf_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    hf_message["content"].append({"type": "image", "image": content.url, **image_kwargs})
                elif content.type == "video":
                    video_item = {"type": "video", "video": content.url, **video_kwargs}
                    if content.start_time is not None:
                        video_item["video_start"] = content.start_time
                    if content.end_time is not None:
                        video_item["video_end"] = content.end_time
                    hf_message["content"].append(video_item)
                elif content.type == "audio":
                    hf_message["content"].append({"type": "audio", "audio": content.url})
            hf_messages.append(hf_message)
        return hf_messages

    def to_openai_messages(self, video_kwargs: Optional[Dict[str, str]] = None, pass_video_url: bool = False):
        image_format = os.getenv("LMMS_IMAGE_ENCODE_FORMAT", "PNG").upper()
        if image_format == "JPG":  # PIL only knows "JPEG"
            image_format = "JPEG"
        jpeg_quality = int(os.getenv("LMMS_IMAGE_JPEG_QUALITY", "85"))
        quality = jpeg_quality if image_format in {"JPEG", "WEBP"} else None
        openai_messages = self._build_openai_messages(video_kwargs, pass_video_url, image_format, quality)

        # Opt-in cap for gateways that reject large bodies (e.g. HTTP 400 "unexpected EOF"
        # at ~10 MB). Only requests over the cap are re-encoded as JPEG, so every request
        # that already fits keeps its original lossless encoding.
        max_request_mb = float(os.getenv("LMMS_MAX_REQUEST_MB", "0"))
        if max_request_mb > 0 and image_format != "JPEG":
            size_mb = len(json.dumps(openai_messages)) / (1024 * 1024)
            if size_mb > max_request_mb:
                openai_messages = self._build_openai_messages(video_kwargs, pass_video_url, "JPEG", jpeg_quality)
                new_size_mb = len(json.dumps(openai_messages)) / (1024 * 1024)
                eval_logger.warning(f"Request body {size_mb:.1f} MB exceeds LMMS_MAX_REQUEST_MB={max_request_mb:g}; re-encoded images as JPEG -> {new_size_mb:.1f} MB")
        return openai_messages

    def _build_openai_messages(self, video_kwargs: Optional[Dict[str, str]], pass_video_url: bool, image_format: str, quality: Optional[int]):
        if video_kwargs is None:
            video_kwargs = {}
        openai_messages = []
        encode_cache: Dict[Tuple[object, ...], str] = {}
        mime_type = f"image/{image_format.lower()}"
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{self.encode_image(content.url, encode_cache, image_format, quality)}"},
                        }
                    )
                elif content.type == "video":
                    if pass_video_url:
                        # Forward the video as a URL so the server can decode it (e.g., vLLM's
                        # media_io_kwargs). Local paths are normalized to file:// so absolute-time
                        # frame indexing is preserved instead of being lost in client-side decoding.
                        url = content.url
                        if not url.startswith(("http://", "https://", "file://", "data:")):
                            url = f"file://{os.path.abspath(url)}"
                        openai_message["content"].append({"type": "video_url", "video_url": {"url": url}})
                        continue
                    if fetch_video is None:
                        raise ImportError("qwen_vl_utils is required for video processing. Please install it with: pip install qwen-vl-utils")
                    video_arg = {"type": "video", "video": content.url, **video_kwargs}
                    if content.start_time is not None:
                        video_arg["video_start"] = content.start_time
                    if content.end_time is not None:
                        video_arg["video_end"] = content.end_time
                    video_input = fetch_video(_clamp_nframes(video_arg))
                    for frame in video_input:
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{self.encode_image(image, encode_cache, image_format, quality)}"},
                            }
                        )
                # TODO, audio hasn't been implemented yet
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def to_qwen3_vl_openai_messages(self, video_kwargs: Optional[Dict[str, str]] = None):
        if video_kwargs is None:
            video_kwargs = {}
        openai_messages = []
        encode_cache: Dict[Tuple[object, ...], str] = {}
        image_format = os.getenv("LMMS_IMAGE_ENCODE_FORMAT", "PNG").upper()
        mime_type = f"image/{'jpeg' if image_format == 'JPG' else image_format.lower()}"
        quality = int(os.getenv("LMMS_IMAGE_JPEG_QUALITY", "85")) if image_format in {"JPEG", "JPG", "WEBP"} else None
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{self.encode_image(content.url, encode_cache, image_format, quality)}"},
                        }
                    )
                elif content.type == "video":
                    if fetch_video is None:
                        raise ImportError("qwen_vl_utils is required for video processing. Please install it with: pip install qwen-vl-utils")
                    video_arg = {"type": "video", "video": content.url, **video_kwargs}
                    if content.start_time is not None:
                        video_arg["video_start"] = content.start_time
                    if content.end_time is not None:
                        video_arg["video_end"] = content.end_time
                    video_input, fps = fetch_video(
                        _clamp_nframes(video_arg),
                        return_video_metadata=True,
                        return_video_sample_fps=True,
                    )
                    frames, video_metadata = video_input
                    timestamps = self._calculate_timestamps(video_metadata)
                    for frame, timestamp in zip(frames, timestamps):
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append({"type": "text", "text": f"<{timestamp:.1f} seconds>"})
                        openai_message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{self.encode_image(image, encode_cache, image_format, quality)}"},
                            }
                        )
                # TODO, audio hasn't been implemented yet
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def _calculate_timestamps(self, video_metadata: Dict[str, Any]):
        indices = video_metadata["frames_indices"]
        if not isinstance(indices, list):
            indices = indices.tolist()
        fps = video_metadata["fps"]
        # Note this is a hardcode value for Qwen3-VL, should only be used for Qwen3-VL
        merge_size = 2
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / fps for idx in indices]
        # timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)]
        return timestamps

    def encode_image(
        self,
        image: Union[Image.Image, str],
        cache: Optional[Dict[Tuple[object, ...], str]] = None,
        image_format: str = "PNG",
        quality: Optional[int] = None,
    ):
        normalized_image_format = image_format.upper()
        return encode_image_to_base64(
            image,
            image_format=normalized_image_format,
            convert_rgb=normalized_image_format in {"JPEG", "JPG", "WEBP"},
            quality=quality,
            copy_if_pil=False,
            cache=cache,
            use_path_cache=True,
        )
