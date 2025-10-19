"""
ComfyUI Workflow Engine
Smart Vision - Advanced AI Video Generation System
"""

import json
import os
import time
import requests
from urllib import request
from urllib.error import HTTPError
import shutil
import threading
import asyncio
from typing import Dict, List, Optional, Callable, Any

# Global configuration
COMFYUI_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = "output"
INPUT_DIR = "input"
VIDEO_WIDTH: Optional[int] = None
VIDEO_HEIGHT: Optional[int] = None
SCENE_INPUTS: Dict[str, Any] = {}

def set_comfyui_url(url: str):
    """Set the ComfyUI URL"""
    global COMFYUI_URL
    COMFYUI_URL = url

def set_directories(output_dir: str, input_dir: str):
    """Set output and input directories"""
    global OUTPUT_DIR, INPUT_DIR
    OUTPUT_DIR = output_dir
    INPUT_DIR = input_dir

def set_video_dimensions(width: Optional[int], height: Optional[int]):
    """Set preferred video/image dimensions to be injected into workflows when applicable."""
    global VIDEO_WIDTH, VIDEO_HEIGHT
    VIDEO_WIDTH = int(width) if width is not None else None
    VIDEO_HEIGHT = int(height) if height is not None else None

def set_scene_inputs(inputs: Dict[str, Any]):
    """Provide scene-level inputs (image_prompt, video_prompt, sound_prompt, audio_prompt)."""
    global SCENE_INPUTS
    SCENE_INPUTS = inputs or {}

def load_workflow_config(config_path: str = "config.json") -> Dict:
    """Load workflow configuration from JSON file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def save_workflow_config(config: Dict, config_path: str = "config.json"):
    """Save workflow configuration to JSON file"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def parse_type_prompts(file_path: str) -> List[str]:
    """Read and parse type_prompts.txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines
    except Exception as e:
        print(f"Error reading type_prompts.txt: {e}")
        return []

def check_comfyui_status() -> bool:
    """Check ComfyUI service status"""
    try:
        response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        print(f"Unable to connect to ComfyUI: {e}")
        return False

def convert_workflow_to_api_format(workflow_data: Dict) -> Dict:
    """Convert ComfyUI workflow to API format"""
    api_prompt = {}
    
    # Node types to filter out (non-executable nodes)
    excluded_node_types = {
        "MarkdownNote", "Note", "Reroute", "PrimitiveNode", "Widget",
    }
    
    # Get subgraph IDs from definitions
    subgraph_ids = set()
    if "definitions" in workflow_data and "subgraphs" in workflow_data["definitions"]:
        for subgraph in workflow_data["definitions"]["subgraphs"]:
            subgraph_ids.add(subgraph.get("id"))
    
    # Iterate through all nodes
    for node in workflow_data.get("nodes", []):
        node_id = str(node["id"])
        node_type = node["type"]
        node_mode = node.get("mode", 0)  # mode 4 = muted
        
        # Skip non-executable nodes
        if node_type in excluded_node_types:
            continue
        
        # Skip muted nodes (mode 4)
        if node_mode == 4:
            continue
        
        # Skip subgraph nodes (UUID-based types)
        if node_type in subgraph_ids:
            continue
        
        # Skip nodes with UUID-like types (fallback check)
        if len(node_type) == 36 and node_type.count('-') == 4:
            continue
        
        widgets_values = node.get("widgets_values", [])
        
        # Build node inputs
        inputs = {}
        
        # Process input connections
        for input_conn in node.get("inputs", []):
            if "link" in input_conn:
                link_id = input_conn["link"]
                for link in workflow_data.get("links", []):
                    if link[0] == link_id:
                        source_node = str(link[1])
                        source_slot = link[2]
                        inputs[input_conn["name"]] = [source_node, source_slot]
                        break
        
        # Process widget values
        if widgets_values:
            inputs = _process_widget_values(node_type, widgets_values, inputs, node)
        
        # Build API format node
        api_prompt[node_id] = {
            "class_type": node_type,
            "inputs": inputs
        }
    
    # Apply scene-level prompt overrides (e.g., CLIPTextEncode text)
    api_prompt = _apply_scene_inputs_to_prompt(api_prompt)

    # Clean up broken node references from skipped/muted nodes
    skipped_nodes = set()
    for node in workflow_data.get("nodes", []):
        node_id = str(node.get("id"))
        if node_id not in api_prompt:
            skipped_nodes.add(node_id)

    for nid, ndata in api_prompt.items():
        inputs = ndata.get("inputs", {})
        broken_inputs = []
        for in_name, in_val in inputs.items():
            if isinstance(in_val, list) and len(in_val) >= 1:
                ref_node = str(in_val[0])
                if ref_node in skipped_nodes:
                    broken_inputs.append(in_name)
        for bin_name in broken_inputs:
            del inputs[bin_name]
    return api_prompt

def _apply_scene_inputs_to_prompt(prompt: Dict[str, Any]) -> Dict[str, Any]:
    """Inject scene prompts into appropriate nodes based on workflow type.
    If the workflow includes WanImageToVideo, prefer video_prompt; otherwise use image_prompt.
    """
    try:
        if not SCENE_INPUTS:
            return prompt
        
        # Detect whether this prompt contains WAN video nodes
        has_wan = any(n.get("class_type") in ["WanImageToVideo", "WanSoundImageToVideo", "WanSoundImageToVideoExtend"] for n in prompt.values())
        
        # Handle CLIPTextEncode nodes based on workflow type
        if has_wan:
            # For WAN workflows, override CLIPTextEncode with video_prompt
            video_text = SCENE_INPUTS.get("video_prompt")
            if video_text:
                video_text = str(video_text).strip()
                if video_text:
                    for node_id, node_data in prompt.items():
                        if node_data.get("class_type") == "CLIPTextEncode":
                            inputs = node_data.get("inputs", {})
                            cur = inputs.get("text", "")
                            # Override positive prompts (empty or short)
                            # Keep negative prompts (usually long with quality terms)
                            if (cur is None) or (isinstance(cur, str) and len(cur.strip()) < 50):
                                inputs["text"] = video_text
        else:
            # For non-WAN workflows, use image_prompt
            image_text = SCENE_INPUTS.get("image_prompt")
            if image_text:
                image_text = str(image_text).strip()
                if image_text:
                    for node_id, node_data in prompt.items():
                        if node_data.get("class_type") == "CLIPTextEncode":
                            inputs = node_data.get("inputs", {})
                            cur = inputs.get("text", "")
                            if (cur is None) or (isinstance(cur, str) and cur.strip() == ""):
                                inputs["text"] = image_text
        
        # Inject image_prompt specifically into TextEncodeQwenImageEditPlus (for character workflows)
        image_text = SCENE_INPUTS.get("image_prompt")
        if image_text:
            image_text = str(image_text).strip()
            if image_text:  # Only if non-empty
                for node_id, node_data in prompt.items():
                    if node_data.get("class_type") == "TextEncodeQwenImageEditPlus":
                        inputs = node_data.get("inputs", {})
                        # Always override to respect user input
                        inputs["prompt"] = image_text
        
        # Inject sound_prompt into AudioXPromptHelper for audio workflows
        sound_text = SCENE_INPUTS.get("sound_prompt")
        if sound_text:
            sound_text = str(sound_text).strip()
            if sound_text:  # Only if non-empty
                for node_id, node_data in prompt.items():
                    if node_data.get("class_type") == "AudioXPromptHelper":
                        inputs = node_data.get("inputs", {})
                        # Always override to respect user input
                        inputs["base_prompt"] = sound_text
        
        # Inject character image into LoadImage nodes for character workflows
        character_image = SCENE_INPUTS.get("character_image")
        if character_image:
            character_image = str(character_image)
            for node_id, node_data in prompt.items():
                if node_data.get("class_type") == "LoadImage":
                    inputs = node_data.get("inputs", {})
                    # Override image input
                    inputs["image"] = character_image
                    # Remove upload field if present
                    inputs.pop("upload", None)
        
        # Inject voice audio into LoadAudio nodes for speaking video workflows
        voice_audio = SCENE_INPUTS.get("voice_audio")
        if voice_audio:
            voice_audio = str(voice_audio).strip()
            if voice_audio:
                for node_id, node_data in prompt.items():
                    if node_data.get("class_type") == "LoadAudio":
                        inputs = node_data.get("inputs", {})
                        # Override audio input
                        inputs["audio"] = voice_audio
        
        # Inject audio_prompt into VibeVoiceSingleSpeakerNode for speaking video workflows
        audio_prompt_text = SCENE_INPUTS.get("audio_prompt")
        if audio_prompt_text:
            audio_prompt_text = str(audio_prompt_text).strip()
            if audio_prompt_text:
                for node_id, node_data in prompt.items():
                    if node_data.get("class_type") == "VibeVoiceSingleSpeakerNode":
                        inputs = node_data.get("inputs", {})
                        # Override text input with audio prompt
                        inputs["text"] = audio_prompt_text
        return prompt
    except Exception:
        return prompt

def _process_widget_values(node_type: str, widgets_values: list, inputs: dict, node: dict) -> dict:
    """Process widget values for different node types"""
    
    if node_type == "KSampler":
        if len(widgets_values) >= 7:
            inputs["seed"] = widgets_values[0]
            inputs["steps"] = widgets_values[2]
            inputs["cfg"] = widgets_values[3]
            inputs["sampler_name"] = widgets_values[4]
            inputs["scheduler"] = widgets_values[5]
            inputs["denoise"] = widgets_values[6]
    
    elif node_type == "KSamplerAdvanced":
        if len(widgets_values) >= 10:
            inputs["noise_mode"] = widgets_values[0]
            inputs["seed"] = widgets_values[1]
            inputs["control_after_generate"] = widgets_values[2]
            inputs["steps"] = widgets_values[3]
            inputs["cfg"] = widgets_values[4]
            inputs["sampler_name"] = widgets_values[5]
            inputs["scheduler"] = widgets_values[6]
            inputs["denoise"] = widgets_values[7]
            inputs["start_at_step"] = widgets_values[7]
            inputs["end_at_step"] = widgets_values[8]
            inputs["return_with_leftover_noise"] = widgets_values[9]
            inputs["add_noise"] = widgets_values[0]  # Use original add_noise setting
            inputs["noise_seed"] = widgets_values[1]
        else:
            inputs["start_at_step"] = 0
            inputs["end_at_step"] = -1
            inputs["add_noise"] = "enable"
            inputs["return_with_leftover_noise"] = "disable"
            inputs["noise_seed"] = 12345
    
    elif node_type == "LoadImage":
        if len(widgets_values) >= 1:
            inputs["image"] = widgets_values[0]
    
    elif node_type == "SaveImage":
        if len(widgets_values) >= 1:
            inputs["filename_prefix"] = widgets_values[0]
        else:
            inputs["filename_prefix"] = "ComfyUI"
    
    elif node_type == "CLIPTextEncode":
        if len(widgets_values) >= 1:
            inputs["text"] = widgets_values[0]
    
    elif node_type == "VAELoader":
        if len(widgets_values) >= 1:
            inputs["vae_name"] = widgets_values[0]
    
    elif node_type == "CLIPLoader":
        if len(widgets_values) >= 3:
            inputs["clip_name"] = widgets_values[0]
            inputs["type"] = widgets_values[1]
            inputs["stop_at_clip_layer"] = widgets_values[2]
        elif len(widgets_values) >= 1:
            inputs["clip_name"] = widgets_values[0]
            inputs["type"] = "qwen_image"
            inputs["stop_at_clip_layer"] = -1
    
    elif node_type == "CLIPVisionLoader":
        # widgets_values: [clip_name]
        if len(widgets_values) >= 1:
            inputs["clip_name"] = widgets_values[0]
        else:
            inputs["clip_name"] = "clip_vision_h.safetensors"

    elif node_type == "CLIPVisionEncode":
        # widgets_values: [crop]
        if len(widgets_values) >= 1:
            inputs["crop"] = widgets_values[0]
        else:
            inputs["crop"] = "none"

    elif node_type == "DualCLIPLoader":
        if len(widgets_values) >= 4:
            inputs["clip_name1"] = widgets_values[0]
            inputs["clip_name2"] = widgets_values[1]
            inputs["type"] = widgets_values[2]
            inputs["stop_at_clip_layer"] = widgets_values[3]
        elif len(widgets_values) >= 2:
            inputs["clip_name1"] = widgets_values[0]
            inputs["clip_name2"] = widgets_values[1]
            inputs["type"] = "flux"
            inputs["stop_at_clip_layer"] = -1
    
    elif node_type == "UnetLoaderGGUF":
        # Some environments restrict available choices; avoid forcing an invalid default.
        # If the workflow supplies a value, prefer to leave it unset here so the server can use its default/selection.
        if len(widgets_values) >= 1:
            # Only set if the value looks like a valid non-empty string; otherwise skip to let backend decide
            try:
                candidate = str(widgets_values[0]).strip()
                if candidate:
                    inputs["unet_name"] = candidate
            except Exception:
                pass
    
    elif node_type == "EmptyLatentImage":
        if len(widgets_values) >= 3:
            inputs["width"] = widgets_values[0]
            inputs["height"] = widgets_values[1]
            inputs["batch_size"] = widgets_values[2]
        elif len(widgets_values) >= 2:
            inputs["width"] = widgets_values[0]
            inputs["height"] = widgets_values[1]
            inputs["batch_size"] = 1
    
    elif node_type == "EmptySD3LatentImage":
        if len(widgets_values) >= 3:
            inputs["width"] = widgets_values[0]
            inputs["height"] = widgets_values[1]
            inputs["batch_size"] = widgets_values[2]
        elif len(widgets_values) >= 2:
            inputs["width"] = widgets_values[0]
            inputs["height"] = widgets_values[1]
            inputs["batch_size"] = 1
    
    elif node_type == "FluxGuidance":
        if len(widgets_values) >= 1:
            inputs["guidance"] = widgets_values[0]
        else:
            inputs["guidance"] = 2.5
    
    elif node_type == "VHS_VideoCombine":
        if isinstance(widgets_values, dict):
            inputs["frame_rate"] = widgets_values.get("frame_rate", 8)
            inputs["loop_count"] = widgets_values.get("loop_count", 0)
            inputs["filename_prefix"] = widgets_values.get("filename_prefix", "AnimateDiff")
            inputs["format"] = widgets_values.get("format", "video/h264-mp4")
            inputs["pingpong"] = widgets_values.get("pingpong", False)
            inputs["save_output"] = widgets_values.get("save_output", True)
        elif isinstance(widgets_values, list) and len(widgets_values) >= 1 and isinstance(widgets_values[0], dict):
            vhs_config = widgets_values[0]
            inputs["frame_rate"] = vhs_config.get("frame_rate", 8)
            inputs["loop_count"] = vhs_config.get("loop_count", 0)
            inputs["filename_prefix"] = vhs_config.get("filename_prefix", "AnimateDiff")
            inputs["format"] = vhs_config.get("format", "video/h264-mp4")
            inputs["pingpong"] = vhs_config.get("pingpong", False)
            inputs["save_output"] = vhs_config.get("save_output", True)
        else:
            inputs["frame_rate"] = 8
            inputs["loop_count"] = 0
            inputs["filename_prefix"] = "AnimateDiff"
            inputs["format"] = "video/h264-mp4"
            inputs["pingpong"] = False
            inputs["save_output"] = True
    
    elif node_type == "VHS_LoadVideo":
        if isinstance(widgets_values, dict):
            inputs["video"] = widgets_values.get("video", "")
            inputs["force_rate"] = widgets_values.get("force_rate", 0)
            inputs["custom_width"] = widgets_values.get("custom_width", 0)
            inputs["custom_height"] = widgets_values.get("custom_height", 0)
            inputs["frame_load_cap"] = widgets_values.get("frame_load_cap", 0)
            inputs["skip_first_frames"] = widgets_values.get("skip_first_frames", 0)
            inputs["select_every_nth"] = widgets_values.get("select_every_nth", 1)
            inputs["filename_prefix"] = widgets_values.get("filename_prefix", "video")
        else:
            inputs["video"] = ""
            inputs["filename_prefix"] = "video"
            inputs["force_rate"] = 0
            inputs["custom_width"] = 0
            inputs["custom_height"] = 0
            inputs["frame_load_cap"] = 0
            inputs["skip_first_frames"] = 0
            inputs["select_every_nth"] = 1
    
    elif node_type == "MMAudioModelLoader":
        if len(widgets_values) >= 2:
            inputs["mmaudio_model"] = widgets_values[0]
            inputs["base_precision"] = widgets_values[1]
        elif len(widgets_values) >= 1:
            inputs["mmaudio_model"] = widgets_values[0]
            inputs["base_precision"] = "fp16"
    
    elif node_type == "MMAudioFeatureUtilsLoader":
        if len(widgets_values) >= 5:
            inputs["vae_model"] = widgets_values[0]
            inputs["synchformer_model"] = widgets_values[1]
            inputs["clip_model"] = widgets_values[2]
            inputs["mode"] = widgets_values[3]
            inputs["precision"] = widgets_values[4]
        elif len(widgets_values) >= 3:
            inputs["vae_model"] = widgets_values[0]
            inputs["synchformer_model"] = widgets_values[1]
            inputs["clip_model"] = widgets_values[2]
            inputs["mode"] = "44k"
            inputs["precision"] = "fp16"
    
    elif node_type == "MMAudioSampler":
        if len(widgets_values) >= 9:
            inputs["steps"] = widgets_values[0]
            inputs["cfg"] = widgets_values[1]
            inputs["seed"] = widgets_values[2]
            inputs["seed_mode"] = widgets_values[3]
            inputs["prompt"] = widgets_values[4]
            inputs["negative_prompt"] = widgets_values[5]
            inputs["mask_away_clip"] = widgets_values[6]
            inputs["force_offload"] = widgets_values[7]
        elif len(widgets_values) >= 3:
            inputs["steps"] = widgets_values[0]
            inputs["cfg"] = widgets_values[1]
            inputs["seed"] = widgets_values[2]
            inputs["seed_mode"] = "fixed"
            inputs["prompt"] = ""
            inputs["negative_prompt"] = ""
            inputs["mask_away_clip"] = False
            inputs["force_offload"] = True
    
    elif node_type == "ModelPatchTorchSettings":
        if len(widgets_values) >= 1:
            inputs["use_torch_compile"] = widgets_values[0]
        inputs["enable_fp16_accumulation"] = False
    
    elif node_type == "ModelSamplingSD3":
        # widgets_values: [shift]
        if len(widgets_values) >= 1:
            inputs["shift"] = widgets_values[0]
        else:
            inputs["shift"] = 5.0
    
    elif node_type == "AudioXModelLoader":
        # model_name is required, use the first value or default to model.ckpt
        if len(widgets_values) > 0:
            inputs["model_name"] = widgets_values[0]
        else:
            inputs["model_name"] = "model.ckpt"
        inputs["device"] = widgets_values[1] if len(widgets_values) > 1 else "auto"
    
    elif node_type == "AudioXPromptHelper":
        inputs["base_prompt"] = widgets_values[0] if len(widgets_values) > 0 else ""
        inputs["template"] = widgets_values[1] if len(widgets_values) > 1 else "default"
        inputs["add_quality_terms"] = widgets_values[2] if len(widgets_values) > 2 else True
        inputs["enhance_automatically"] = widgets_values[3] if len(widgets_values) > 3 else True
    
    elif node_type == "CFGNorm":
        # widgets_values: [strength]
        if len(widgets_values) >= 1:
            inputs["strength"] = widgets_values[0]
        else:
            inputs["strength"] = 1.0
    
    elif node_type == "ImageScaleToTotalPixels":
        # widgets_values: [upscale_method, megapixels]
        if len(widgets_values) >= 2:
            inputs["upscale_method"] = widgets_values[0]
            inputs["megapixels"] = widgets_values[1]
        elif len(widgets_values) >= 1:
            inputs["upscale_method"] = widgets_values[0]
            inputs["megapixels"] = 1
        else:
            inputs["upscale_method"] = "lanczos"
            inputs["megapixels"] = 1
    
    elif node_type == "ModelSamplingAuraFlow":
        # widgets_values: [shift]
        if len(widgets_values) >= 1:
            inputs["shift"] = widgets_values[0]
        else:
            inputs["shift"] = 3.0
    
    elif node_type == "TextEncodeQwenImageEditPlus":
        # widgets_values: [prompt]
        inputs["prompt"] = widgets_values[0] if len(widgets_values) > 0 else ""
    
    elif node_type == "AudioXEnhancedVideoToAudio":
        # Ensure we only use numeric values for numeric parameters
        if len(widgets_values) > 0 and isinstance(widgets_values[0], (int, float)):
            inputs["video_weight"] = widgets_values[0]
        else:
            inputs["video_weight"] = 1.0
            
        if len(widgets_values) > 1 and isinstance(widgets_values[1], (int, float)):
            inputs["steps"] = widgets_values[1]
        else:
            inputs["steps"] = 20
            
        if len(widgets_values) > 2 and isinstance(widgets_values[2], (int, float)):
            inputs["seed"] = widgets_values[2]
        else:
            inputs["seed"] = 0
            
        if len(widgets_values) > 3 and isinstance(widgets_values[3], (int, float)):
            inputs["video_cfg_scale"] = widgets_values[3]
        else:
            inputs["video_cfg_scale"] = 7.5
            
        if len(widgets_values) > 4 and isinstance(widgets_values[4], (int, float)):
            inputs["duration_seconds"] = widgets_values[4]
        else:
            inputs["duration_seconds"] = 10.0
            
        if len(widgets_values) > 5 and isinstance(widgets_values[5], (int, float)):
            inputs["text_weight"] = widgets_values[5]
        else:
            inputs["text_weight"] = 1.0
            
        if len(widgets_values) > 6 and isinstance(widgets_values[6], (int, float)):
            # Ensure text_cfg_scale is within valid range (0.1 to 20.0)
            cfg_scale = float(widgets_values[6])
            if 0.1 <= cfg_scale <= 20.0:
                inputs["text_cfg_scale"] = cfg_scale
            else:
                inputs["text_cfg_scale"] = 7.5
        else:
            inputs["text_cfg_scale"] = 7.5
    
    elif node_type == "AudioXVolumeControl":
        inputs["volume_db"] = widgets_values[0] if len(widgets_values) > 0 else 0.0
    
    elif node_type == "SaveAudio":
        inputs["filename_prefix"] = widgets_values[0] if len(widgets_values) > 0 else "audio"

    elif node_type == "LoraLoaderModelOnly":
        # widgets_values: [lora_name, strength_model]
        if len(widgets_values) >= 1:
            inputs["lora_name"] = widgets_values[0]
        else:
            inputs["lora_name"] = ""
        inputs["strength_model"] = widgets_values[1] if len(widgets_values) >= 2 else 1.0

    elif node_type == "WanImageToVideo":
        if len(widgets_values) >= 4:
            # Prefer UI-provided dimensions when set
            inputs["width"] = VIDEO_WIDTH if VIDEO_WIDTH is not None else widgets_values[0]
            inputs["height"] = VIDEO_HEIGHT if VIDEO_HEIGHT is not None else widgets_values[1]
            inputs["num_frames"] = widgets_values[2]
            inputs["fps"] = widgets_values[3]
            inputs["batch_size"] = 1
            inputs["length"] = widgets_values[2]
    
    elif node_type == "Image Save":
        # WAS Node Suite "Image Save" schema (as in your workflows):
        # widgets_values order:
        # 0: filename_prefix (STRING)
        # 1: output_path (STRING)
        # 2: filename_delimiter (STRING)
        # 3: filename_number_padding (INT)
        # 4: filename_number_start (COMBO -> 'false'/'true')
        # 5: extension (COMBO)
        # 6: dpi (INT)
        # 7: quality (INT)
        # 8: optimize_image (COMBO -> 'false'/'true')
        # 9: lossless_webp (COMBO -> 'false'/'true')
        # 10: overwrite_mode (COMBO -> 'false'/'true')
        # 11: show_history (COMBO -> 'false'/'true')
        # 12: show_history_by_prefix (COMBO -> 'false'/'true')
        # 13: embed_workflow (COMBO -> 'false'/'true')
        # 14: show_previews (COMBO -> 'false'/'true')

        def gv(idx, default=None):
            try:
                return widgets_values[idx]
            except Exception:
                return default

        def to_bool_str(v, default=False):
            try:
                if isinstance(v, str):
                    v_lower = v.strip().lower()
                    if v_lower in ("true", "false"):
                        return v_lower
                    if v_lower in ("1", "yes", "y"): return "true"
                    if v_lower in ("0", "no", "n"): return "false"
                if isinstance(v, (int, float)):
                    return "true" if v else "false"
                if isinstance(v, bool):
                    return "true" if v else "false"
            except Exception:
                pass
            return "true" if default else "false"

        def to_int(v, default):
            try:
                return int(v)
            except Exception:
                return default

        inputs["filename_prefix"] = str(gv(0, inputs.get("filename_prefix", "ComfyUI")))
        inputs["output_path"] = str(gv(1, inputs.get("output_path", OUTPUT_DIR)))
        inputs["filename_delimiter"] = str(gv(2, inputs.get("filename_delimiter", "_")))
        inputs["filename_number_padding"] = to_int(gv(3, inputs.get("filename_number_padding", 4)), 4)
        inputs["filename_number_start"] = to_bool_str(gv(4, inputs.get("filename_number_start", False)), False)
        inputs["extension"] = str(gv(5, inputs.get("extension", "png")))
        inputs["dpi"] = to_int(gv(6, inputs.get("dpi", 72)), 72)
        inputs["quality"] = to_int(gv(7, inputs.get("quality", 90)), 90)
        inputs["optimize_image"] = to_bool_str(gv(8, inputs.get("optimize_image", False)), False)
        inputs["lossless_webp"] = to_bool_str(gv(9, inputs.get("lossless_webp", False)), False)
        inputs["overwrite_mode"] = to_bool_str(gv(10, inputs.get("overwrite_mode", False)), False)
        inputs["show_history"] = to_bool_str(gv(11, inputs.get("show_history", True)), True)
        inputs["show_history_by_prefix"] = to_bool_str(gv(12, inputs.get("show_history_by_prefix", True)), True)
        inputs["embed_workflow"] = to_bool_str(gv(13, inputs.get("embed_workflow", True)), True)
        inputs["show_previews"] = to_bool_str(gv(14, inputs.get("show_previews", False)), False)
    
    elif node_type == "ImageResizeKJv2":
        # widgets_values: [width, height, upscale_method, keep_proportion, pad_color, crop_position, divisible_by, device, per_batch]
        def gv_ir(idx, default=None):
            try:
                return widgets_values[idx]
            except Exception:
                return default
        inputs["width"] = VIDEO_WIDTH if VIDEO_WIDTH is not None else gv_ir(0, 640)
        inputs["height"] = VIDEO_HEIGHT if VIDEO_HEIGHT is not None else gv_ir(1, 360)
        inputs["upscale_method"] = gv_ir(2, "lanczos")
        inputs["keep_proportion"] = gv_ir(3, "pad")
        inputs["pad_color"] = gv_ir(4, "0, 0, 0")
        inputs["crop_position"] = gv_ir(5, "center")
        inputs["divisible_by"] = gv_ir(6, 2)
        inputs["device"] = gv_ir(7, "cpu")
        inputs["per_batch"] = gv_ir(8, 0)
    
    # Audio and Latent Processing Nodes
    elif node_type == "AudioEncoderLoader":
        # widgets_values: [audio_encoder_name]
        if len(widgets_values) >= 1:
            inputs["audio_encoder_name"] = widgets_values[0]
        else:
            inputs["audio_encoder_name"] = "wav2vec2_large_english_fp16.safetensors"
    
    elif node_type == "AudioEncoderEncode":
        # No widget values - only has connection inputs
        pass
    
    elif node_type == "LatentConcat":
        # widgets_values: [dim] - "t" for temporal
        if len(widgets_values) >= 1:
            inputs["dim"] = widgets_values[0]
        else:
            inputs["dim"] = "t"
    
    elif node_type == "LatentCut":
        # widgets_values: [dim, index, amount]
        if len(widgets_values) >= 3:
            inputs["dim"] = widgets_values[0]
            inputs["index"] = widgets_values[1]
            inputs["amount"] = widgets_values[2]
        else:
            inputs["dim"] = "t"
            inputs["index"] = 0
            inputs["amount"] = 1
    
    # Primitive Nodes
    elif node_type == "PrimitiveInt":
        # widgets_values: [value, control]
        if len(widgets_values) >= 1:
            inputs["value"] = widgets_values[0]
        if len(widgets_values) >= 2:
            inputs["control"] = widgets_values[1]
    
    elif node_type == "PrimitiveFloat":
        # widgets_values: [value]
        if len(widgets_values) >= 1:
            inputs["value"] = widgets_values[0]
    
    # WAN Video Nodes
    elif node_type == "WanSoundImageToVideo":
        # widgets_values: [width, height, length, batch_size]
        if len(widgets_values) >= 4:
            inputs["width"] = VIDEO_WIDTH if VIDEO_WIDTH is not None else widgets_values[0]
            inputs["height"] = VIDEO_HEIGHT if VIDEO_HEIGHT is not None else widgets_values[1]
            inputs["length"] = widgets_values[2]
            inputs["batch_size"] = widgets_values[3]
        elif len(widgets_values) >= 3:
            inputs["width"] = VIDEO_WIDTH if VIDEO_WIDTH is not None else widgets_values[0]
            inputs["height"] = VIDEO_HEIGHT if VIDEO_HEIGHT is not None else widgets_values[1]
            inputs["length"] = widgets_values[2]
            inputs["batch_size"] = 1
    
    elif node_type == "WanSoundImageToVideoExtend":
        # widgets_values: [length]
        if len(widgets_values) >= 1:
            inputs["length"] = widgets_values[0]
        else:
            inputs["length"] = 77
    
    # VibeVoice Nodes
    elif node_type == "VibeVoiceSingleSpeakerNode":
        # widgets_values: [text, model, attention_type, quantize_llm, free_memory_after_generate, 
        #                  diffusion_steps, seed, control_after_generate, cfg_scale, use_sampling, 
        #                  temperature, top_p, max_words_per_chunk, voice_speed_factor]
        if len(widgets_values) >= 1:
            inputs["text"] = widgets_values[0]
        if len(widgets_values) >= 2:
            inputs["model"] = widgets_values[1]  # FIX: was model_name
        if len(widgets_values) >= 3:
            inputs["attention_type"] = widgets_values[2]  # FIX: was device
        if len(widgets_values) >= 4:
            inputs["quantize_llm"] = widgets_values[3]  # FIX: was precision
        if len(widgets_values) >= 5:
            inputs["free_memory_after_generate"] = widgets_values[4]  # FIX: was use_lora
        if len(widgets_values) >= 6:
            inputs["diffusion_steps"] = widgets_values[5]  # FIX: was max_new_tokens
        if len(widgets_values) >= 7:
            inputs["seed"] = widgets_values[6]
        if len(widgets_values) >= 8:
            inputs["control_after_generate"] = widgets_values[7]
        if len(widgets_values) >= 9:
            inputs["cfg_scale"] = widgets_values[8]  # FIX: was temperature
        if len(widgets_values) >= 10:
            inputs["use_sampling"] = widgets_values[9]
        if len(widgets_values) >= 11:
            inputs["temperature"] = widgets_values[10]  # FIX: was top_k
        if len(widgets_values) >= 12:
            inputs["top_p"] = widgets_values[11]
        if len(widgets_values) >= 13:
            inputs["max_words_per_chunk"] = widgets_values[12]  # NEW: was missing
        if len(widgets_values) >= 14:
            inputs["voice_speed_factor"] = widgets_values[13]  # NEW: was missing
        
        # Note: voice_to_clone connection is already handled in input processing
        # No need to add defaults - all parameters are in widgets_values
    
    elif node_type == "VibeVoiceFreeMemoryNode":
        # No widget values - passthrough node
        pass
    
    # Audio and Image Loading Nodes
    elif node_type == "LoadAudio":
        # widgets_values: [audio_file, start_time, duration]
        if len(widgets_values) >= 1:
            inputs["audio"] = widgets_values[0]
        # start_time and duration are optional
    
    elif node_type == "ImageFromBatch":
        # widgets_values: [batch_index, length]
        if len(widgets_values) >= 2:
            inputs["batch_index"] = widgets_values[0]
            inputs["length"] = widgets_values[1]
        elif len(widgets_values) >= 1:
            inputs["batch_index"] = widgets_values[0]
            inputs["length"] = 1
    
    return inputs

def queue_prompt(prompt: Dict) -> Optional[str]:
    """Submit prompt to ComfyUI"""
    try:
        p = {"prompt": prompt}
        headers = {'Content-Type': 'application/json'}
        data = json.dumps(p).encode('utf-8')
        req = request.Request(f"{COMFYUI_URL}/prompt", data=data, headers=headers)
        response = request.urlopen(req)
        result = response.read().decode('utf-8')
        return result
    except HTTPError as e:
        error_body = ""
        if e.fp:
            error_body = e.fp.read().decode('utf-8')
        print(f"HTTP error {e.code}: {e.reason} - {error_body}")
        return None
    except Exception as e:
        print(f"Submission failed: {e}")
        return None

def wait_for_completion(prompt_id: str, timeout: int = 300000, log_callback: Optional[Callable] = None) -> bool:
    """Wait for task completion"""
    start_time = time.time()
    check_count = 0
    
    if log_callback:
        log_callback(f"Monitoring task {prompt_id}...")
    
    while time.time() - start_time < timeout:
        try:
            check_count += 1
            response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5)
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    if log_callback:
                        log_callback(f"Task {prompt_id} completed (checked {check_count} times)")
                    return True
                else:
                    if check_count % 10 == 0 and log_callback:
                        log_callback(f"Task still running... (checked {check_count} times)")
            else:
                if log_callback:
                    log_callback(f"Failed to check task status: HTTP {response.status_code}")
        except Exception as e:
            if log_callback:
                log_callback(f"Error checking task status: {e}")
        time.sleep(2)
    
    if log_callback:
        log_callback(f"Task {prompt_id} timed out")
    return False

def get_task_outputs(prompt_id: str) -> Optional[Dict]:
    """Get task output results"""
    try:
        response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        if response.status_code == 200:
            history = response.json()
            if prompt_id in history:
                task_data = history[prompt_id]
                if "outputs" in task_data:
                    return task_data["outputs"]
        return None
    except Exception as e:
        print(f"Failed to get task outputs: {e}")
        return None

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        
        # Get GPU memory if available
        gpu_memory = {}
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory[f"gpu_{i}_allocated"] = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    gpu_memory[f"gpu_{i}_cached"] = torch.cuda.memory_reserved(i) / 1024**3  # GB
        except Exception:
            pass
        
        return {
            "process_rss": memory_info.rss / 1024**3,  # GB
            "process_vms": memory_info.vms / 1024**3,  # GB
            "system_used": system_memory.used / 1024**3,  # GB
            "system_available": system_memory.available / 1024**3,  # GB
            "system_percent": system_memory.percent,
            **gpu_memory
        }
    except Exception as e:
        return {"error": str(e)}

def free_memory_and_wait(wait_time: int = 20, log_callback: Optional[Callable] = None, 
                        aggressive: bool = True, retry_count: int = 3) -> bool:
    """Enhanced memory freeing with multiple methods and better error handling"""
    
    def log_with_timestamp(message: str):
        if log_callback:
            timestamp = time.strftime("%H:%M:%S")
            log_callback(f"[{timestamp}] {message}")
    
    def execute_with_retry(func, method_name: str, max_retries: int = 3) -> bool:
        """Execute a function with retry logic"""
        for attempt in range(max_retries):
            try:
                result = func()
                if result:
                    log_with_timestamp(f"✅ {method_name} succeeded (attempt {attempt + 1})")
                    return True
                else:
                    log_with_timestamp(f"⚠️ {method_name} returned False (attempt {attempt + 1})")
            except Exception as e:
                log_with_timestamp(f"❌ {method_name} failed (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
        
        log_with_timestamp(f"❌ {method_name} failed after {max_retries} attempts")
        return False
    
    try:
        # Log initial memory state
        initial_memory = get_memory_usage()
        log_with_timestamp("🧹 Starting enhanced memory cleanup...")
        log_with_timestamp(f"📊 Initial memory: {initial_memory}")
        
        # Wait before starting cleanup
        if wait_time > 0:
            log_with_timestamp(f"⏳ Waiting {wait_time} seconds before cleanup...")
            time.sleep(wait_time)
        
        success_count = 0
        total_methods = 0
        
        # Method 1: Enhanced /free API with parameters
        def free_api_method():
            headers = {'Content-Type': 'application/json'}
            # Send both unload_models and free_memory flags
            payload = {
                "unload_models": True,
                "free_memory": True
            }
            response = requests.post(f"{COMFYUI_URL}/free", 
                                   json=payload, 
                                   headers=headers, 
                                   timeout=15)
            return response.status_code == 200
        
        total_methods += 1
        if execute_with_retry(free_api_method, "ComfyUI /free API"):
            success_count += 1
        
        # Method 2: Unload models API
        def unload_models_method():
            headers = {'Content-Type': 'application/json'}
            response = requests.post(f"{COMFYUI_URL}/unload_models", 
                                    json={}, 
                                    headers=headers, 
                                    timeout=15)
            return response.status_code == 200
        
        total_methods += 1
        if execute_with_retry(unload_models_method, "Unload Models API"):
            success_count += 1
        
        # Method 3: Force garbage collection (multiple passes)
        def gc_method():
            import gc
            # Multiple passes for thorough cleanup
            for i in range(3):
                collected = gc.collect()
                if i == 0:
                    log_with_timestamp(f"🗑️ Garbage collection pass {i+1}: collected {collected} objects")
            return True
        
        total_methods += 1
        if execute_with_retry(gc_method, "Garbage Collection"):
            success_count += 1
        
        # Method 4: CUDA memory cleanup
        def cuda_cleanup_method():
            try:
                import torch
                if torch.cuda.is_available():
                    # Clear cache on all devices
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Additional CUDA cleanup
                    torch.cuda.ipc_collect()
                    return True
                return False
            except Exception:
                return False
        
        total_methods += 1
        if execute_with_retry(cuda_cleanup_method, "CUDA Memory Cleanup"):
            success_count += 1
        
        # Method 5: System memory cleanup (if aggressive mode)
        if aggressive:
            def system_cleanup_method():
                try:
                    import os
                    if os.name != 'nt':  # Linux/Unix
                        # Drop page cache, dentries and inodes
                        os.system('sync && echo 3 > /proc/sys/vm/drop_caches')
                    return True
                except Exception:
                    return False
            
            total_methods += 1
            if execute_with_retry(system_cleanup_method, "System Memory Cleanup"):
                success_count += 1
        
        # Method 6: Additional Python memory cleanup
        def python_cleanup_method():
            try:
                import sys
                # Clear Python's internal caches
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
                return True
            except Exception:
                return False
        
        total_methods += 1
        if execute_with_retry(python_cleanup_method, "Python Internal Cleanup"):
            success_count += 1
        
        # Method 7: Clear ComfyUI queue if possible
        def clear_queue_method():
            try:
                headers = {'Content-Type': 'application/json'}
                # Try to clear the queue
                response = requests.post(f"{COMFYUI_URL}/queue", 
                                       json={"clear": True}, 
                                       headers=headers, 
                                       timeout=10)
                return response.status_code == 200
            except Exception:
                return False
        
        total_methods += 1
        if execute_with_retry(clear_queue_method, "Clear ComfyUI Queue"):
            success_count += 1
        
        # Log final memory state
        final_memory = get_memory_usage()
        log_with_timestamp(f"📊 Final memory: {final_memory}")
        
        # Calculate memory freed
        if "error" not in initial_memory and "error" not in final_memory:
            memory_freed = {
                "process_rss": initial_memory.get("process_rss", 0) - final_memory.get("process_rss", 0),
                "system_available": final_memory.get("system_available", 0) - initial_memory.get("system_available", 0)
            }
            log_with_timestamp(f"💾 Memory freed: {memory_freed}")
        
        # Summary
        success_rate = (success_count / total_methods) * 100 if total_methods > 0 else 0
        log_with_timestamp(f"📈 Cleanup summary: {success_count}/{total_methods} methods succeeded ({success_rate:.1f}%)")
        
        # Final wait
        if wait_time > 0:
            log_with_timestamp(f"⏳ Final wait: {wait_time} seconds...")
            time.sleep(wait_time)
        
        return success_count > 0
        
    except Exception as e:
        log_with_timestamp(f"💥 Critical error in memory cleanup: {e}")
        return False

def free_memory_async(wait_time: int = 20, log_callback: Optional[Callable] = None, 
                     aggressive: bool = True, retry_count: int = 3) -> threading.Thread:
    """Start memory cleanup in a separate thread to avoid blocking"""
    
    def cleanup_thread():
        """Thread function for memory cleanup"""
        try:
            result = free_memory_and_wait(wait_time, log_callback, aggressive, retry_count)
            if log_callback:
                log_callback(f"🧹 Async memory cleanup completed: {'Success' if result else 'Failed'}")
        except Exception as e:
            if log_callback:
                log_callback(f"💥 Async memory cleanup error: {e}")
    
    thread = threading.Thread(target=cleanup_thread, daemon=True)
    thread.start()
    return thread

async def free_memory_async_await(wait_time: int = 20, log_callback: Optional[Callable] = None, 
                                 aggressive: bool = True, retry_count: int = 3) -> bool:
    """Asynchronous memory cleanup using asyncio"""
    
    def log_with_timestamp(message: str):
        if log_callback:
            timestamp = time.strftime("%H:%M:%S")
            log_callback(f"[{timestamp}] {message}")
    
    try:
        log_with_timestamp("🚀 Starting async memory cleanup...")
        
        # Run the synchronous cleanup in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            free_memory_and_wait, 
            wait_time, 
            log_callback, 
            aggressive, 
            retry_count
        )
        
        log_with_timestamp(f"✅ Async memory cleanup completed: {'Success' if result else 'Failed'}")
        return result
        
    except Exception as e:
        log_with_timestamp(f"💥 Async memory cleanup error: {e}")
        return False

def copy_output_to_input(output_filename: str, subfolder: str = "", output_type: str = "output") -> Optional[str]:
    """Copy file from ComfyUI output (with optional subfolder) to input folder and return the basename.
    ComfyUI history entries usually include { filename, subfolder, type }.
    """
    try:
        # Build source path honoring subfolder when provided and type indicates output
        src_dir = OUTPUT_DIR
        if output_type == "output" and subfolder:
            src_dir = os.path.join(OUTPUT_DIR, subfolder)
        output_path = os.path.join(src_dir, output_filename)

        os.makedirs(INPUT_DIR, exist_ok=True)
        input_path = os.path.join(INPUT_DIR, output_filename)
        
        if not os.path.exists(output_path):
            print(f"Source file does not exist: {output_path}")
            return None
        
        shutil.copy2(output_path, input_path)
        
        return output_filename if os.path.exists(input_path) else None
    except Exception as e:
        print(f"Error copying file: {e}")
        return None

def copy_video_to_input(output_filename: str, subfolder: str = "", output_type: str = "output") -> Optional[str]:
    """Copy video from output folder to input folder"""
    return copy_output_to_input(output_filename, subfolder=subfolder, output_type=output_type)

def update_workflow_with_previous_output(workflow_prompt: Dict, previous_outputs: Dict, workflow_name: str, log_callback: Optional[Callable] = None) -> Dict:
    """Update current workflow with output from previous workflow"""
    if not previous_outputs:
        if log_callback:
            log_callback("No previous workflow output data")
        return workflow_prompt
    
    if log_callback:
        log_callback("Analyzing previous workflow output...")
    
    # Copy workflow to avoid modifying original data
    updated_prompt = {}
    for node_id, node_data in workflow_prompt.items():
        updated_prompt[node_id] = {
            "class_type": node_data["class_type"],
            "inputs": node_data["inputs"].copy()
        }
    
    # Find image output from previous workflow
    previous_image_output = None
    for node_id, node_output in previous_outputs.items():
        # Ensure node_output is a dictionary
        if not isinstance(node_output, dict):
            if log_callback:
                log_callback(f"Skipping node {node_id}: output is not a dictionary (type: {type(node_output)})")
            continue
            
        if "images" in node_output and len(node_output["images"]) > 0:
            previous_image_output = node_output["images"][0]
            if log_callback:
                log_callback(f"Found previous workflow image output: {previous_image_output['filename']}")
            break
        # Some nodes report under "files" instead
        if "files" in node_output and len(node_output["files"]) > 0:
            file_entry = node_output["files"][0]
            # Normalize to images-like entry shape
            previous_image_output = {
                "filename": file_entry.get("filename", file_entry if isinstance(file_entry, str) else ""),
                "subfolder": file_entry.get("subfolder", ""),
                "type": file_entry.get("type", "output")
            }
            if log_callback and previous_image_output.get("filename"):
                log_callback(f"Found previous workflow file output: {previous_image_output['filename']}")
            break
    
    if previous_image_output:
        output_filename = previous_image_output.get("filename")
        subfolder = previous_image_output.get("subfolder", "")
        output_type = previous_image_output.get("type", "output")
        copied_filename = copy_output_to_input(output_filename, subfolder=subfolder, output_type=output_type)
        
        if not copied_filename:
            if log_callback:
                log_callback(f"Unable to copy image file: {output_filename}")
            return updated_prompt
        
        # Update LoadImage nodes in current workflow
        updated_count = 0
        for node_id, node_data in updated_prompt.items():
            if node_data["class_type"] == "LoadImage":
                old_image = node_data["inputs"].get("image", "Not set")
                # Ensure filename is a clean string (no stray spaces)
                node_data["inputs"]["image"] = str(copied_filename).strip()
                # Remove upload field if present; not part of API inputs
                if "upload" in node_data["inputs"]:
                    node_data["inputs"].pop("upload", None)
                if log_callback:
                    log_callback(f"Updated LoadImage node {node_id}: {old_image} -> {copied_filename}")
                updated_count += 1
        
        if updated_count == 0 and log_callback:
            log_callback("No LoadImage nodes found to update")
    else:
        if log_callback:
            log_callback("No previous workflow image output found")
    
    return updated_prompt

def update_workflow_with_video_input(workflow_prompt: Dict, previous_outputs: Dict, workflow_name: str, log_callback: Optional[Callable] = None) -> Dict:
    """Update current workflow with video output from previous workflow"""
    if not previous_outputs:
        if log_callback:
            log_callback("No previous workflow video output data")
        return workflow_prompt
    
    if log_callback:
        log_callback("Analyzing previous workflow video output...")
        log_callback(f"Previous outputs keys: {list(previous_outputs.keys())}")
        for node_id, node_output in previous_outputs.items():
            if isinstance(node_output, dict):
                log_callback(f"Node {node_id} output keys: {list(node_output.keys())}")
                # Log the actual content for debugging
                for key, value in node_output.items():
                    if isinstance(value, list) and len(value) > 0:
                        log_callback(f"  {key}: {value}")
                    elif isinstance(value, str) and value.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        log_callback(f"  {key}: {value} (VIDEO FILE DETECTED)")
            else:
                log_callback(f"Node {node_id} output type: {type(node_output)}")
    
    # Copy workflow to avoid modifying original data
    updated_prompt = {}
    for node_id, node_data in workflow_prompt.items():
        updated_prompt[node_id] = {
            "class_type": node_data["class_type"],
            "inputs": node_data["inputs"].copy()
        }
    
    # Find video output from previous workflow
    previous_video_output = None
    for node_id, node_output in previous_outputs.items():
        # Ensure node_output is a dictionary
        if not isinstance(node_output, dict):
            if log_callback:
                log_callback(f"Skipping node {node_id}: output is not a dictionary (type: {type(node_output)})")
            continue
            
        # Check for videos in multiple possible keys
        video_found = False
        
        # Check "videos" key
        if "videos" in node_output and len(node_output["videos"]) > 0:
            previous_video_output = node_output["videos"][0]
            if log_callback:
                log_callback(f"Found previous workflow video output (videos): {previous_video_output['filename']}")
            video_found = True
            break
            
        # Check "files" key
        elif "files" in node_output and len(node_output["files"]) > 0:
            for file_info in node_output["files"]:
                if isinstance(file_info, dict) and file_info.get("filename", "").lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    previous_video_output = file_info
                    if log_callback:
                        log_callback(f"Found previous workflow video output (files): {previous_video_output['filename']}")
                    video_found = True
                    break
            if video_found:
                break
                
        # Check "filenames" key (for VHS_FILENAMES output)
        elif "filenames" in node_output and len(node_output["filenames"]) > 0:
            for filename_info in node_output["filenames"]:
                if isinstance(filename_info, dict) and filename_info.get("filename", "").lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    previous_video_output = filename_info
                    if log_callback:
                        log_callback(f"Found previous workflow video output (filenames): {previous_video_output['filename']}")
                    video_found = True
                    break
            if video_found:
                break

        # Some ComfyUI builds place video info under a "gifs" key (despite containing mp4)
        elif "gifs" in node_output and len(node_output["gifs"]) > 0:
            for gif_info in node_output["gifs"]:
                if isinstance(gif_info, dict) and gif_info.get("filename", "").lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    previous_video_output = gif_info
                    if log_callback:
                        log_callback(f"Found previous workflow video output (gifs): {previous_video_output['filename']}")
                    video_found = True
                    break
            if video_found:
                break
                
        # Check for direct filename in node output (some nodes might output filename directly)
        elif "filename" in node_output and node_output["filename"].lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            previous_video_output = {
                "filename": node_output["filename"],
                "subfolder": node_output.get("subfolder", ""),
                "type": node_output.get("type", "output")
            }
            if log_callback:
                log_callback(f"Found previous workflow video output (direct filename): {previous_video_output['filename']}")
            video_found = True
            break
    
    if previous_video_output:
        output_filename = previous_video_output.get("filename")
        subfolder = previous_video_output.get("subfolder", "")
        output_type = previous_video_output.get("type", "output")
        copied_filename = copy_video_to_input(output_filename, subfolder=subfolder, output_type=output_type)
        
        if not copied_filename:
            if log_callback:
                log_callback(f"Unable to copy video file: {output_filename}")
            return updated_prompt
        
        # Update VHS_LoadVideo nodes in current workflow
        updated_count = 0
        for node_id, node_data in updated_prompt.items():
            if node_data["class_type"] == "VHS_LoadVideo":
                old_video = node_data["inputs"].get("video", "Not set")
                node_data["inputs"]["video"] = copied_filename
                if log_callback:
                    log_callback(f"Updated VHS_LoadVideo node {node_id}: {old_video} -> {copied_filename}")
                updated_count += 1
        
        if log_callback:
            log_callback(f"Total VHS_LoadVideo nodes updated: {updated_count}")
        
        if updated_count == 0 and log_callback:
            log_callback("No VHS_LoadVideo nodes found to update")
    else:
        if log_callback:
            log_callback("No previous workflow video output found")
    
    return updated_prompt

def execute_single_workflow(
    workflow_path: str,
    workflow_name: str,
    previous_outputs: Optional[Dict],
    chain_idx: int,
    timeout: int = 300000,
    log_callback: Optional[Callable] = None
) -> Optional[Dict]:
    """Execute a single workflow and return its outputs"""
    
    try:
        # Load workflow file
        if log_callback:
            log_callback(f"Loading workflow: {workflow_path}")
        
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        
        # Convert to API format
        if log_callback:
            log_callback("Converting workflow to API format...")
        
        workflow_prompt = convert_workflow_to_api_format(workflow_data)
        
        # Handle data transfer from previous workflow
        if previous_outputs:
            if log_callback:
                log_callback("Updating workflow with previous outputs...")
            
            # Try to update with image first
            workflow_prompt = update_workflow_with_previous_output(
                workflow_prompt, previous_outputs, workflow_name, log_callback
            )
            
            # Try to update with video
            workflow_prompt = update_workflow_with_video_input(
                workflow_prompt, previous_outputs, workflow_name, log_callback
            )
        
        # Modify seed for variation (only if seed is 0 or not set)
        for node_id, node_data in workflow_prompt.items():
            if node_data["class_type"] in ["KSampler", "KSamplerAdvanced", "MMAudioSampler"]:
                if "seed" in node_data["inputs"]:
                    current_seed = node_data["inputs"]["seed"]
                    # Only modify seed if it's 0 or not set, preserve original seeds
                    if current_seed == 0 or current_seed is None:
                        node_data["inputs"]["seed"] = 12345 + chain_idx * 1000
        
        # Submit workflow
        if log_callback:
            log_callback(f"Submitting {workflow_name} to ComfyUI...")
        
        result = queue_prompt(workflow_prompt)
        if not result:
            if log_callback:
                log_callback(f"Failed to submit {workflow_name}")
            return None
        
        # Parse response
        response_data = json.loads(result)
        prompt_id = response_data.get("prompt_id")
        
        if log_callback:
            log_callback(f"{workflow_name} submitted, ID: {prompt_id}")
        
        # Wait for completion
        if wait_for_completion(prompt_id, timeout, log_callback):
            if log_callback:
                log_callback(f"{workflow_name} completed successfully")
            
            # Get outputs
            outputs = get_task_outputs(prompt_id)
            if outputs:
                if log_callback:
                    log_callback(f"{workflow_name} outputs retrieved")
                return outputs
            else:
                if log_callback:
                    log_callback(f"Failed to get {workflow_name} outputs")
                return None
        else:
            if log_callback:
                log_callback(f"{workflow_name} timed out")
            return None
    
    except Exception as e:
        if log_callback:
            log_callback(f"Error executing {workflow_name}: {e}")
        return None

def execute_workflow_chain(
    workflow_group: Dict,
    chain_idx: int,
    wait_time: int = 20,
    timeout: int = 30000,
    log_callback: Optional[Callable] = None
) -> bool:
    """Execute a complete workflow chain"""
    
    group_name = workflow_group.get("name", "Unknown")
    workflows = workflow_group.get("workflows", [])
    
    if log_callback:
        log_callback(f"\n{'='*60}")
        log_callback(f"Executing workflow chain: {group_name}")
        log_callback(f"Total workflows: {len(workflows)}")
        log_callback(f"{'='*60}\n")
    
    previous_outputs = None
    
    for idx, workflow_path in enumerate(workflows):
        workflow_name = f"Workflow {idx + 1}/{len(workflows)}"
        
        if log_callback:
            log_callback(f"\n--- {workflow_name}: {os.path.basename(workflow_path)} ---")
        
        # Check if workflow file exists
        if not os.path.exists(workflow_path):
            if log_callback:
                log_callback(f"ERROR: Workflow file not found: {workflow_path}")
            return False
        
        # Execute workflow
        outputs = execute_single_workflow(
            workflow_path,
            workflow_name,
            previous_outputs,
            chain_idx,
            timeout,
            log_callback
        )
        
        if outputs is None:
            if log_callback:
                log_callback(f"ERROR: {workflow_name} failed to execute")
            return False
        
        # Store outputs for next workflow
        previous_outputs = outputs
        
        # Free memory and wait after each workflow
        if log_callback:
            log_callback(f"Calling free_memory_and_wait with wait_time={wait_time}")
        free_memory_and_wait(wait_time, log_callback)
    
    if log_callback:
        log_callback(f"\n{'='*60}")
        log_callback(f"Workflow chain completed: {group_name}")
        log_callback(f"{'='*60}\n")
    
    return True

def process_type_prompts_file(
    type_prompts_path: str,
    config: Dict,
    wait_time: int = 20,
    timeout: int = 30000,
    log_callback: Optional[Callable] = None
) -> bool:
    """Process type_prompts.txt file and execute workflow chains"""
    
    # Parse type prompts
    type_numbers = parse_type_prompts(type_prompts_path)
    
    if not type_numbers:
        if log_callback:
            log_callback("ERROR: No type numbers found in file")
        return False
    
    if log_callback:
        log_callback(f"Found {len(type_numbers)} workflow chains to execute")
        log_callback(f"Type numbers: {type_numbers}\n")
    
    # Execute each workflow chain
    for idx, type_num in enumerate(type_numbers):
        if log_callback:
            log_callback(f"\n\n{'#'*60}")
            log_callback(f"Processing line {idx + 1}/{len(type_numbers)}: Type {type_num}")
            log_callback(f"{'#'*60}\n")
        
        # Get workflow group from config
        workflow_group = config.get(str(type_num))
        
        if not workflow_group:
            if log_callback:
                log_callback(f"ERROR: No workflow group found for type {type_num}")
            continue
        
        # Execute workflow chain
        success = execute_workflow_chain(
            workflow_group,
            idx,
            wait_time,
            timeout,
            log_callback
        )
        
        if not success:
            if log_callback:
                log_callback(f"WARNING: Workflow chain {idx + 1} failed, continuing to next...")
        
        # Free memory between chains
        if idx < len(type_numbers) - 1:
            free_memory_and_wait(wait_time, log_callback)
    
    if log_callback:
        log_callback(f"\n\n{'#'*60}")
        log_callback(f"All workflow chains completed!")
        log_callback(f"{'#'*60}\n")
    
    return True


