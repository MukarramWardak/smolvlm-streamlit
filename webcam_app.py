import time
import cv2
import av
import torch
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from transformers import AutoProcessor, AutoModelForVision2Seq

# ---------------- Page config ----------------
st.set_page_config(
    page_title="SmolVLM Live Gender Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 SmolVLM Live Gender Classifier")
st.caption("Live Camera → SmolVLM → Male / Female")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"**Device:** {DEVICE}")

# ---------------- Controls ----------------
st.subheader("⚙️ Settings")

interval_seconds = st.slider(
    "Run inference every N seconds",
    min_value=1,
    max_value=10,
    value=5,
)

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        size={"longest_edge": 512},
    )

    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        _attn_implementation="eager",
    ).to(DEVICE)

    model.eval()
    return processor, model


processor, model = load_model()

# ---------------- Video Processor ----------------
class SmolVLMProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_infer_time = 0.0
        self.last_result = "Detecting…"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        now = time.time()

        # ---- Time-based inference ----
        if now - self.last_infer_time >= interval_seconds:
            self.last_infer_time = now

            image = Image.fromarray(img)

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a strict binary image classifier. "
                                "Output exactly one word: Male or Female. "
                                "No extra words."
                            )
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Classify the person in this image."},
                    ],
                },
            ]

            prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=2,
                    do_sample=False,
                )

            decoded = processor.batch_decode(
                output_ids,
                skip_special_tokens=True
            )[0]

            self.last_result = decoded.split("Assistant:")[-1].strip()

        # ---- Overlay ----
        overlay = img.copy()
        label = self.last_result

        color = (0, 255, 0) if label == "Male" else (255, 0, 255)

        cv2.putText(
            overlay,
            label,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            color,
            3,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(overlay, format="rgb24")


# ---------------- WebRTC Stream ----------------
webrtc_streamer(
    key="smolvlm-live",
    video_processor_factory=SmolVLMProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    },
)
