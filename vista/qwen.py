from html import parser
from typing import Dict
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from transformers import (
    AutoModel,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwen_vl_utils import process_vision_info
from vista.BindingOutput import DetectionList
try :
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = SamplingParams = None

try :
	from unsloth import FastVisionModel
except ImportError:
	       FastVisionModel = None

from .utils_fun import set_seed, image_to_base64, resize_image, log
from unsloth import FastVisionModel
import torch


def build_qwen_chat(current_frame, cfg, history):
    messages = []

    # ----------------------------
    # System (ONCE)
    # ----------------------------
    messages.append({
        "role": "system",
        "content": [
            {"type": "text", "text": cfg["qwen"]["system_prompt"]}
        ]
    })

    # ----------------------------
    # Past frames + past responses
    # ----------------------------
    for idx, (frame, response) in enumerate(history):
        if cfg.get("resize_image", True):
            frame = resize_image(frame, cfg["qwen"].get("image_target_size", 1024))

        b64 = image_to_base64(frame)

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{b64}"},
                {"type": "text", "text": f"Frame t-{len(history)-idx}"}
            ]
        })

        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": response}
            ]
        })

    # ----------------------------
    # Current frame
    # ----------------------------
    if cfg.get("resize_image", True):
        current_frame = resize_image(
            current_frame,
            cfg["qwen"].get("image_target_size", 1024)
        )

    b64 = image_to_base64(current_frame)

    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": f"data:image;base64,{b64}"},
            {"type": "text", "text": cfg["qwen"]["user_prompt"]}
        ]
    })

    return messages


class QwenVLforObjectDetection:
    def generate(self, **kwargs):
        raise NotImplementedError("This is a placeholder for QwenVLforObjectDetection.")


class QwenVLHF(QwenVLforObjectDetection):
    def __init__(self, model, processor, sampling_params, cfg, device):
        self.model = model
        self.processor = processor
        self.sampling_params = sampling_params
        self.cfg = cfg
        self.device = device

    def generate(self, frame, history):
        messages = build_qwen_chat(frame, self.cfg, history)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        log("Prepared input messages for the model")

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        if video_inputs:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            for v in video_inputs:
                log(f"Video input detected with shape: {v.shape}")
        else:
            video_metadatas = None, None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            video_metadata=video_metadatas,
            **video_kwargs,
        ).to(self.device)

        log("Inputs tokenized and moved to device")
        parser = JsonSchemaParser(DetectionList.schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(self.processor.tokenizer, parser)
        with torch.no_grad():
            # Qwen3-VL video-native
            out_ids = self.model.generate(
                **inputs,
                prefix_allowed_tokens_fn=prefix_function,
                **self.sampling_params,
            )
        log("Model generation completed")

        # Trim input_ids
        gen_ids = [out[len(inp) :] for inp, out in zip(inputs.input_ids, out_ids)]
        return self.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


class QwenVLUnsloth(QwenVLforObjectDetection):
    def __init__(self, model, processor, tokenizer, sampling_params, cfg, device):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.cfg = cfg
        self.device = device

    def prepare_inputs_for_vllm(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # qwen_vl_utils 0.0.14+ reqired
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        print(f"video_kwargs: {video_kwargs}")

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return self.tokenizer(
            image_inputs,
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

    def generate(self, frame, history):
        parser = JsonSchemaParser(DetectionList.schema())#Bruno
        prefix_function = build_transformers_prefix_allowed_tokens_fn(self.processor.tokenizer, parser)#Bruno
        messages = build_qwen_chat(frame, self.cfg, history)
        inputs = [self.prepare_inputs_for_vllm(message) for message in [messages]][0]

        outputs = self.model.generate(**inputs,prefix_allowed_tokens_fn=prefix_function, **self.sampling_params)#Bruno
        
        # Trim input_ids
        gen_ids = [out[len(inp) :] for inp, out in zip(inputs.input_ids, outputs)]
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]


# ============================================================
# Model loader
# ============================================================


class AutoModelQwenVL:
    @staticmethod
    def from_pretrained(model_id: str):
        if model_id.startswith("Qwen/Qwen2-VL"):
            loader = Qwen2VLForConditionalGeneration
        elif model_id.startswith("Qwen/Qwen2.5-VL"):
            loader = Qwen2_5_VLForConditionalGeneration
        elif model_id.startswith("Qwen/Qwen3-VL-235B"):
            loader = Qwen3VLMoeForConditionalGeneration
        elif model_id.startswith("Qwen/Qwen3-VL"):
            loader = Qwen3VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model ID: {model_id}")

        return loader.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
        )


def get_model(cfg):
    model_id = cfg["qwen"]["model_id"]
    device = cfg["device"]

    if model_id == "Qwen/Qwen3-VL-32B-Instruct-FP8":
        processor = AutoProcessor.from_pretrained(cfg["qwen"]["model_id"])
        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            gpu_memory_utilization=0.70,
            enforce_eager=False,
            tensor_parallel_size=1,
            seed=0,
        )

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            top_k=-1,
            stop_token_ids=[],
        )
        return QwenVLHF(
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
            device=device,
            cfg=cfg,
        )

    if (
        model_id.startswith("Qwen/Qwen3-VL")
        or model_id.startswith("Qwen/Qwen2.5-VL")
        or model_id.startswith("Qwen/Qwen2-VL")
        or model_id.startswith("Qwen/Qwen3-VL-235B")
    ):
        model = AutoModelQwenVL.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(cfg["qwen"]["model_id"], device_map="auto")
        model.eval()
        return QwenVLHF(
            model=model,
            processor=processor,
            sampling_params={
                "max_new_tokens": cfg["qwen"].get("max_new_tokens", 512)
            },
            device=device,
            cfg=cfg,
        )
    if model_id.startswith("unsloth/"):
        return get_unsloth(model_id, device, cfg)
    model = AutoModel.from_pretrained(model_id)


def get_unsloth(model_id, device, cfg):
    max_seq_length = 16384  # Must be this long for VLMs
    processor = AutoProcessor.from_pretrained(model_id)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=False,  # Enable vLLM fast inference
        gpu_memory_utilization=1.0,  # Reduce if out of memory
    )
    FastVisionModel.for_inference(model)  # Enable native 2x faster inference
    model.eval()

    return QwenVLUnsloth(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        sampling_params=dict(
            max_new_tokens=1024, use_cache=True, temperature=1.0, min_p=0.1
        ),
        device=device,
        cfg=cfg,
    )
