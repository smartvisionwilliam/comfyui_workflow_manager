"""
ComfyUI Video Generation Workflow Manager
Smart Vision - Advanced AI Video Generation System
"""

import gradio as gr
import requests
import uuid
import json
import os
import threading
import queue
import time
from typing import Optional, Dict, List, Any
from workflow_engine import (
    load_workflow_config,
    save_workflow_config,
    check_comfyui_status,
    process_type_prompts_file,
    execute_workflow_chain,
    set_comfyui_url,
    set_directories,
    set_video_dimensions,
    set_scene_inputs,
    COMFYUI_URL,
    OUTPUT_DIR,
    INPUT_DIR,
    execute_single_workflow,
    free_memory_and_wait,
    free_memory_async,
    get_memory_usage
)

# Global state
current_process_thread: Optional[threading.Thread] = None
stop_processing = False
log_buffer = []
status_queue = queue.Queue()
execution_status = "idle"

def log_message(message: str):
    """Add message to log buffer"""
    global log_buffer
    log_buffer.append(message)
    print(message)

def update_status(message: str):
    """Update status message"""
    global status_queue
    status_queue.put(message)
    log_message(message)

def get_logs():
    """Get all logs as a single string"""
    global log_buffer
    return "\n".join(log_buffer)

def get_status_messages():
    """Get status messages from queue"""
    global status_queue
    messages = []
    while not status_queue.empty():
        try:
            message = status_queue.get_nowait()
            messages.append(message)
        except queue.Empty:
            break
    return "\n".join(messages)

def unload_all_models_via_prompt(comfyui_url: str, logger=print) -> bool:
    """Trigger ComfyUI to unload all models using FL_UnloadAllModels node via /prompt.
    Falls back silently if node is unavailable.
    """
    try:
        client_id = str(uuid.uuid4())
        payload = {
            "prompt": {
                # Minimal graph: single node that unloads all models
                "1": {
                    "class_type": "FL_UnloadAllModels",
                    "inputs": {"value": "*"}
                }
            },
            "client_id": client_id
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(f"{comfyui_url}/prompt", json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            logger("Models unload requested via FL_UnloadAllModels")
            return True
        else:
            logger(f"Unload via node failed (status {resp.status_code})")
            return False
    except Exception as e:
        logger(f"Unload via node error: {e}")
        return False

def clear_logs():
    """Clear log buffer"""
    global log_buffer
    log_buffer = []

def clear_status():
    """Clear status queue"""
    global status_queue
    while not status_queue.empty():
        try:
            status_queue.get_nowait()
        except queue.Empty:
            break

def get_memory_status():
    """Get current memory status for display"""
    try:
        memory_info = get_memory_usage()
        if "error" in memory_info:
            return f"❌ Memory monitoring error: {memory_info['error']}"
        
        status_lines = []
        status_lines.append(f"📊 Process Memory: {memory_info.get('process_rss', 0):.2f} GB")
        status_lines.append(f"💻 System Memory: {memory_info.get('system_used', 0):.2f} GB used, {memory_info.get('system_available', 0):.2f} GB available")
        status_lines.append(f"📈 System Usage: {memory_info.get('system_percent', 0):.1f}%")
        
        # Add GPU memory info if available
        gpu_info = []
        for key, value in memory_info.items():
            if key.startswith('gpu_'):
                gpu_info.append(f"{key}: {value:.2f} GB")
        
        if gpu_info:
            status_lines.append(f"🎮 GPU Memory: {', '.join(gpu_info)}")
        
        return "\n".join(status_lines)
    except Exception as e:
        return f"❌ Error getting memory status: {e}"

def update_file_path(file_obj):
    """Update file path from file object"""
    if file_obj is None:
        return ""
    return file_obj.name if hasattr(file_obj, 'name') else str(file_obj)

def preview_file_content(file_obj):
    """Preview file content"""
    if file_obj is None:
        return "No file selected"
    
    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Show first 10 lines
        lines = content.split('\n')[:10]
        preview = '\n'.join(lines)
        if len(content.split('\n')) > 10:
            preview += '\n... (showing first 10 lines)'
        
        return f"File: {file_path}\n\nContent preview:\n{preview}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ==================== Multi-Scene Video Generation ====================

def read_prompts_file(file_path: str) -> List[str]:
    """Read prompts from a text file"""
    try:
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        update_status(f"Error reading {file_path}: {e}")
        return []

def execute_multi_scene_workflow(
    character_image_path: str,
    voice_audio_path: str,
    video_width: int,
    video_height: int,
    type_prompts_path: str,
    image_prompts_path: str,
    video_prompts_path: str,
    sound_prompts_path: str,
    audio_prompts_path: str,
    comfyui_url: str,
    wait_time: int = 20,
    timeout: int = 300000
):
    """Execute multi-scene video generation workflow"""
    global execution_status, current_process_thread
    
    if current_process_thread and current_process_thread.is_alive():
        return "⚠️ A task is already running, please wait for completion"
    
    execution_status = "running"
    clear_status()
    update_status("🎬 Starting multi-scene video generation task...")
    
    # Validate file existence
    files_to_check = [type_prompts_path, image_prompts_path, video_prompts_path, 
                     sound_prompts_path, audio_prompts_path]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        error_msg = f"❌ The following files do not exist: {', '.join(missing_files)}"
        update_status(error_msg)
        execution_status = "error"
        return error_msg
    
    # Read prompt files
    type_prompts = read_prompts_file(type_prompts_path)
    image_prompts = read_prompts_file(image_prompts_path)
    video_prompts = read_prompts_file(video_prompts_path)
    sound_prompts = read_prompts_file(sound_prompts_path)
    audio_prompts = read_prompts_file(audio_prompts_path)
    
    # Validate file lengths
    file_lengths = [len(type_prompts), len(image_prompts), len(video_prompts), 
                   len(sound_prompts), len(audio_prompts)]
    if len(set(file_lengths)) > 1:
        error_msg = "❌ All prompt files must have the same number of lines"
        update_status(error_msg)
        execution_status = "error"
        return error_msg
    
    if not type_prompts:
        error_msg = "❌ type_prompts.txt file is empty"
        update_status(error_msg)
        execution_status = "error"
        return error_msg
    
    # Set ComfyUI URL
    set_comfyui_url(comfyui_url)
    
    # Point engine IO to ComfyUI's real input/output folders
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        set_directories(
            output_dir=os.path.join(base_dir, "output"),
            input_dir=os.path.join(base_dir, "input")
        )
    except Exception:
        pass
    
    # Inject UI video dimensions for downstream nodes
    try:
        set_video_dimensions(video_width, video_height)
    except Exception:
        pass
    
    # Check ComfyUI connection
    update_status("🔍 Checking ComfyUI connection...")
    if not check_comfyui_status():
        error_msg = f"❌ Cannot connect to ComfyUI: {comfyui_url}"
        update_status(error_msg)
        execution_status = "error"
        return error_msg
    
    update_status("✅ ComfyUI connection successful")
    
    # Load configuration
    config = load_workflow_config()
    if not config:
        error_msg = "❌ No workflow configuration found, please add workflow groups in the configuration page first"
        update_status(error_msg)
        execution_status = "error"
        return error_msg
    
    total_scenes = len(type_prompts)
    update_status(f"📊 Total scenes to execute: {total_scenes}")
    
    # Execute in a new thread
    def execute_thread():
        try:
            for scene_idx in range(total_scenes):
                scene_num = scene_idx + 1
                workflow_group_num = int(type_prompts[scene_idx])
                group_name = f"Workflow Group {workflow_group_num}"
                
                update_status(f"🎬 Starting scene {scene_num}, using {group_name}")
                
                # Get workflow group
                workflow_group = config.get(str(workflow_group_num))
                if not workflow_group:
                    update_status(f"❌ Workflow group not found: {group_name}")
                    continue
                # Provide scene-level inputs to engine for prompt injection
                try:
                    scene_inputs_dict = {
                        "image_prompt": image_prompts[scene_idx],
                        "video_prompt": video_prompts[scene_idx],
                        "sound_prompt": sound_prompts[scene_idx],
                        "audio_prompt": audio_prompts[scene_idx]
                    }
                    # Add character image if provided
                    if character_image_path:
                        # Copy to input directory first
                        import shutil
                        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                        input_dir = os.path.join(base_dir, "input")
                        if os.path.exists(character_image_path):
                            filename = os.path.basename(character_image_path)
                            dest_path = os.path.join(input_dir, filename)
                            try:
                                shutil.copy2(character_image_path, dest_path)
                                scene_inputs_dict["character_image"] = filename
                            except Exception:
                                scene_inputs_dict["character_image"] = character_image_path
                    
                    # Handle voice audio file if provided
                    if voice_audio_path:
                        import shutil
                        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                        input_dir = os.path.join(base_dir, "input")
                        if os.path.exists(voice_audio_path):
                            filename = os.path.basename(voice_audio_path)
                            dest_path = os.path.join(input_dir, filename)
                            try:
                                shutil.copy2(voice_audio_path, dest_path)
                                voice_audio_filename = filename
                            except Exception:
                                voice_audio_filename = voice_audio_path
                        else:
                            voice_audio_filename = None
                    else:
                        voice_audio_filename = None
                    
                    # Add voice audio if provided
                    if voice_audio_filename:
                        scene_inputs_dict["voice_audio"] = voice_audio_filename
                    
                    set_scene_inputs(scene_inputs_dict)
                except Exception:
                    pass
                
                # Execute workflow group
                success = execute_workflow_group_with_scene_data(
                    workflow_group,
                    scene_num,
                    image_prompts[scene_idx],
                    video_prompts[scene_idx],
                    sound_prompts[scene_idx],
                    audio_prompts[scene_idx],
                    character_image_path,
                    video_width,
                    video_height,
                    wait_time,
                    timeout
                )
                
                if success:
                    update_status(f"✅ Scene {scene_num} executed successfully")
                else:
                    update_status(f"❌ Scene {scene_num} execution failed")
                    break
            
            update_status("🎉 All scenes completed!")
            execution_status = "completed"
            
        except Exception as e:
            update_status(f"❌ Error during execution: {str(e)}")
            execution_status = "error"
    
    current_process_thread = threading.Thread(target=execute_thread)
    current_process_thread.start()
    
    return "🚀 Starting multi-scene video generation task..."

def execute_workflow_group_with_scene_data(
    workflow_group: Dict,
    scene_num: int,
    image_prompt: str,
    video_prompt: str,
    sound_prompt: str,
    audio_prompt: str,
    character_image_path: str,
    video_width: int,
    video_height: int,
    wait_time: int,
    timeout: int
) -> bool:
    """Execute workflow group with scene-specific data"""
    
    group_name = workflow_group.get("name", "Unknown")
    workflows = workflow_group.get("workflows", [])
    
    update_status(f"📋 Executing workflow group: {group_name} ({len(workflows)} workflows)")
    
    previous_outputs = None
    
    for idx, workflow_path in enumerate(workflows):
        workflow_name = f"Workflow {idx + 1}/{len(workflows)}"
        workflow_file = os.path.basename(workflow_path)
        
        update_status(f"🔄 {workflow_name}: {workflow_file}")
        
        # Check if workflow file exists
        if not os.path.exists(workflow_path):
            update_status(f"❌ Workflow file does not exist: {workflow_path}")
            return False
        
        # Prepare workflow input data
        workflow_inputs = prepare_workflow_inputs(
            workflow_file,
            image_prompt,
            video_prompt,
            sound_prompt,
            audio_prompt,
            character_image_path,
            video_width,
            video_height,
            previous_outputs
        )
        
        # Execute workflow
        outputs = execute_single_workflow(
            workflow_path,
            workflow_name,
            previous_outputs,
            scene_num - 1,  # Use scene number as chain_idx
            timeout,
            update_status
        )
        
        if outputs is None:
            update_status(f"❌ {workflow_name} execution failed")
            return False
        
        update_status(f"✅ {workflow_name} executed successfully")
        
        # Store outputs for next workflow
        previous_outputs = outputs
        
        # Free memory and wait (except for last workflow)
        if idx < len(workflows) - 1:
            update_status(f"🧹 Freeing GPU memory and RAM...")
            # Attempt aggressive unload via node first (if available)
            try:
                unload_all_models_via_prompt(COMFYUI_URL, update_status)
            except Exception:
                pass
            # Use enhanced memory cleanup with aggressive mode
            free_memory_and_wait(wait_time, update_status, aggressive=True, retry_count=3)
    
    return True

def prepare_workflow_inputs(
    workflow_file: str,
    image_prompt: str,
    video_prompt: str,
    sound_prompt: str,
    audio_prompt: str,
    character_image_path: str,
    video_width: int,
    video_height: int,
    previous_outputs: Optional[Dict]
) -> Dict[str, Any]:
    """Prepare inputs for different workflow types"""
    
    inputs = {}
    
    # Determine workflow type based on filename and prepare corresponding inputs
    if "Generate_none_character_scene" in workflow_file:
        # Workflow Group 1: Non-character scene generation
        inputs["image_prompt"] = image_prompt
        inputs["video_width"] = video_width
        inputs["video_height"] = video_height
        
    elif "Generate_character_scene" in workflow_file:
        # Workflow Group 2 and 3: Character scene generation
        inputs["image_prompt"] = image_prompt
        inputs["character_image"] = character_image_path
        inputs["video_width"] = video_width
        inputs["video_height"] = video_height
        
    elif "image_to_video" in workflow_file:
        # Image to video workflow - use previous output if available
        inputs["video_prompt"] = video_prompt
        inputs["video_width"] = video_width
        inputs["video_height"] = video_height
        
        
        if previous_outputs and "image" in previous_outputs:
            inputs["character_image"] = previous_outputs["image"]
            update_status(f"🔄 Using generated image from previous workflow: {previous_outputs['image']}")
        else:
            inputs["character_image"] = character_image_path
            update_status(f"🔄 Using original character image: {character_image_path}")
        
    elif "image_to_video_with_audio" in workflow_file:
        # Image to video (with dialogue) workflow - use previous output if available
        inputs["video_prompt"] = video_prompt
        inputs["audio_prompt"] = audio_prompt
        inputs["video_width"] = video_width
        inputs["video_height"] = video_height
        
       
        if previous_outputs and "image" in previous_outputs:
            inputs["character_image"] = previous_outputs["image"]
            update_status(f"🔄 Using generated image from previous workflow: {previous_outputs['image']}")
        else:
            inputs["character_image"] = character_image_path
            update_status(f"🔄 Using original character image: {character_image_path}")
        
    elif "generate_sound_by_video" in workflow_file:
        # Generate background music workflow
        inputs["sound_prompt"] = sound_prompt
    
    return inputs

# ==================== Tab 1: Batch Processing ====================

def upload_type_prompts(file):
    """Handle type_prompts.txt file upload"""
    if file is None:
        return "No file uploaded", ""
    
    try:
        # Read file content
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        return f"File uploaded successfully! Found {len(lines)} workflow chains to execute.", content
    except Exception as e:
        return f"Error reading file: {e}", ""

def start_batch_processing(
    type_prompts_file,
    comfyui_url,
    wait_time,
    timeout,
    output_dir,
    input_dir
):
    """Start batch processing of type_prompts.txt"""
    global current_process_thread, stop_processing
    
    if current_process_thread and current_process_thread.is_alive():
        return "Processing is already running!", get_logs()
    
    if type_prompts_file is None:
        return "Please upload a type_prompts.txt file first!", get_logs()
    
    # Clear previous logs
    clear_logs()
    
    # Update configuration
    set_comfyui_url(comfyui_url)
    set_directories(output_dir, input_dir)
    
    # Check ComfyUI connection
    log_message("Checking ComfyUI connection...")
    if not check_comfyui_status():
        log_message(f"ERROR: Cannot connect to ComfyUI at {comfyui_url}")
        log_message("Please ensure ComfyUI is running and accessible.")
        return "Failed to connect to ComfyUI", get_logs()
    
    log_message("ComfyUI connection successful!")
    
    # Load configuration
    config = load_workflow_config()
    if not config:
        log_message("ERROR: No workflow configuration found!")
        log_message("Please configure workflow groups in the Configuration tab.")
        return "No workflow configuration found", get_logs()
    
    log_message(f"Loaded configuration with {len(config)} workflow groups")
    
    # Start processing in a separate thread
    stop_processing = False
    
    def process_thread():
        try:
            process_type_prompts_file(
                type_prompts_file.name,
                config,
                wait_time,
                timeout,
                log_message
            )
            log_message("\n\nBatch processing completed!")
        except Exception as e:
            log_message(f"\n\nERROR in batch processing: {e}")
    
    current_process_thread = threading.Thread(target=process_thread)
    current_process_thread.start()
    
    return "Batch processing started!", get_logs()

def stop_batch_processing():
    """Stop batch processing"""
    global stop_processing
    stop_processing = True
    log_message("\n\nStop requested by user...")
    return "Stop requested (will finish current workflow)", get_logs()

def refresh_logs():
    """Refresh log display"""
    return get_logs()

# ==================== Tab 2: Configuration Manager ====================

def load_config_display():
    """Load and display current configuration"""
    config = load_workflow_config()
    
    if not config:
        return "No configuration found. Add workflow groups below.", ""
    
    # Format configuration as readable text
    config_text = json.dumps(config, indent=2, ensure_ascii=False)
    
    # Create table data
    table_data = []
    for type_num, group_data in config.items():
        name = group_data.get("name", "Unknown")
        workflows = group_data.get("workflows", [])
        workflow_list = "\n".join(workflows)
        table_data.append([type_num, name, len(workflows), workflow_list])
    
    return config_text, table_data

def add_workflow_group(type_number, group_name, workflow_files):
    """Add a new workflow group"""
    if not type_number or not group_name:
        return "Please provide both type number and group name!", load_config_display()[0]
    
    if not workflow_files:
        return "Please provide at least one workflow file!", load_config_display()[0]
    
    # Parse workflow files (one per line)
    workflow_list = [line.strip() for line in workflow_files.split('\n') if line.strip()]
    
    # Validate workflow files exist
    missing_files = []
    for wf_path in workflow_list:
        if not os.path.exists(wf_path):
            missing_files.append(wf_path)
    
    if missing_files:
        return f"Warning: Some workflow files do not exist:\n" + "\n".join(missing_files), load_config_display()[0]
    
    # Load current config
    config = load_workflow_config()
    
    # Add new group
    config[str(type_number)] = {
        "name": group_name,
        "workflows": workflow_list
    }
    
    # Save config
    if save_workflow_config(config):
        return f"Workflow group {type_number} added successfully!", load_config_display()[0]
    else:
        return "Failed to save configuration!", load_config_display()[0]

def delete_workflow_group(type_number):
    """Delete a workflow group"""
    if not type_number:
        return "Please provide a type number!", load_config_display()[0]
    
    # Load current config
    config = load_workflow_config()
    
    if str(type_number) not in config:
        return f"Type number {type_number} not found in configuration!", load_config_display()[0]
    
    # Delete group
    del config[str(type_number)]
    
    # Save config
    if save_workflow_config(config):
        return f"Workflow group {type_number} deleted successfully!", load_config_display()[0]
    else:
        return "Failed to save configuration!", load_config_display()[0]

def validate_config():
    """Validate all workflow files in configuration"""
    config = load_workflow_config()
    
    if not config:
        return "No configuration to validate."
    
    results = []
    all_valid = True
    
    for type_num, group_data in config.items():
        name = group_data.get("name", "Unknown")
        workflows = group_data.get("workflows", [])
        
        results.append(f"\nType {type_num}: {name}")
        
        for wf_path in workflows:
            if os.path.exists(wf_path):
                results.append(f"  ✓ {wf_path}")
            else:
                results.append(f"  ✗ {wf_path} (NOT FOUND)")
                all_valid = False
    
    if all_valid:
        results.insert(0, "✓ All workflow files are valid!")
    else:
        results.insert(0, "✗ Some workflow files are missing!")
    
    return "\n".join(results)

def load_preset_workflow_groups():
    """Load preset workflow group configurations"""
    preset_groups = {
        "1": {
            "name": "Workflow Group 1 - Generate non-character Scene",
            "workflows": [
                "D:/ComfyUI-hon/workflows/generate_none_character_scene.json",
                "D:/ComfyUI-hon/workflows/image_to_video_wan2.2.json",
                "D:/ComfyUI-hon/workflows/generate_sound_by_video.json"
            ]
        },
        "2": {
            "name": "Workflow Group 2: Generate One Character Scene",
            "workflows": [
                "D:/ComfyUI-hon/workflows/generate_one_character_scene.json",
                "D:/ComfyUI-hon/workflows/image_to_video_wan2.2.json",
                "D:/ComfyUI-hon/workflows/generate_sound_by_video.json"
            ]
        },
        "3": {
            "name": "Workflow Group 3 - Generate One Character Speaking Video",
            "workflows": [
                "D:/ComfyUI-hon/workflows/generate_one_character_scene.json",
                "D:/ComfyUI-hon/workflows/generate_one_character_speaking_video.json"
            ]
        }
    }
    
    # Load current configuration
    config = load_workflow_config()
    
    # Add preset groups
    for type_num, group_data in preset_groups.items():
        config[type_num] = group_data
    
    # Save configuration
    if save_workflow_config(config):
        return "✅ Preset workflow group configuration loaded!\n\n" + load_config_display()[0]
    else:
        return "❌ Failed to save configuration!"

# ==================== Tab 3: Manual Execution ====================

def get_workflow_groups():
    """Get list of workflow groups for dropdown"""
    config = load_workflow_config()
    if not config:
        return []
    
    return [f"{k}: {v.get('name', 'Unknown')}" for k, v in config.items()]

def execute_manual_workflow(
    workflow_group_str,
    num_executions,
    comfyui_url,
    wait_time,
    timeout,
    output_dir,
    input_dir,
    manual_image_prompt,
    manual_video_prompt,
    manual_sound_prompt,
    manual_audio_prompt,
    manual_character_image,
    manual_voice_audio
):
    """Execute a workflow group manually"""
    global current_process_thread
    
    if current_process_thread and current_process_thread.is_alive():
        return "Processing is already running!", get_logs()
    
    if not workflow_group_str:
        return "Please select a workflow group!", get_logs()
    
    # Clear previous logs
    clear_logs()
    
    # Extract type number from selection
    type_num = workflow_group_str.split(':')[0].strip()
    
    # Update configuration
    set_comfyui_url(comfyui_url)
    # Resolve ComfyUI IO dirs: if relative like "output"/"input", point to ComfyUI root
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        resolved_output = output_dir
        resolved_input = input_dir
        if output_dir.strip().lower() == "output":
            resolved_output = os.path.join(base_dir, "output")
        if input_dir.strip().lower() == "input":
            resolved_input = os.path.join(base_dir, "input")
        set_directories(resolved_output, resolved_input)
    except Exception:
        set_directories(output_dir, input_dir)
    
    # Handle character image upload if provided
    character_image_path = None
    if manual_character_image is not None:
        character_image_path = manual_character_image.name if hasattr(manual_character_image, 'name') else str(manual_character_image)
        # Copy to input directory
        import shutil
        if character_image_path and os.path.exists(character_image_path):
            filename = os.path.basename(character_image_path)
            dest_path = os.path.join(resolved_input, filename)
            shutil.copy2(character_image_path, dest_path)
            character_image_path = filename  # Use just filename for workflow

    # Handle voice audio upload if provided
    voice_audio_path = None
    if manual_voice_audio is not None:
        voice_audio_path = manual_voice_audio.name if hasattr(manual_voice_audio, 'name') else str(manual_voice_audio)
        # Copy to input directory
        if voice_audio_path and os.path.exists(voice_audio_path):
            filename = os.path.basename(voice_audio_path)
            dest_path = os.path.join(resolved_input, filename)
            shutil.copy2(voice_audio_path, dest_path)
            voice_audio_path = filename  # Use just filename for workflow

    # Provide optional prompts for manual run
    try:
        scene_inputs = {
            "image_prompt": (manual_image_prompt or ""),
            "video_prompt": (manual_video_prompt or ""),
            "sound_prompt": (manual_sound_prompt or ""),
            "audio_prompt": (manual_audio_prompt or ""),
            "voice_audio": (voice_audio_path or "")
        }
        if character_image_path:
            scene_inputs["character_image"] = character_image_path
        set_scene_inputs(scene_inputs)
    except Exception:
        pass
    
    # Check ComfyUI connection
    log_message("Checking ComfyUI connection...")
    if not check_comfyui_status():
        log_message(f"ERROR: Cannot connect to ComfyUI at {comfyui_url}")
        return "Failed to connect to ComfyUI", get_logs()
    
    log_message("ComfyUI connection successful!")
    
    # Load configuration
    config = load_workflow_config()
    workflow_group = config.get(type_num)
    
    if not workflow_group:
        log_message(f"ERROR: Workflow group {type_num} not found!")
        return f"Workflow group {type_num} not found", get_logs()
    
    # Start execution in a separate thread
    def execute_thread():
        try:
            for i in range(num_executions):
                log_message(f"\n\nExecution {i + 1}/{num_executions}")
                log_message("="*60)
                
                success = execute_workflow_chain(
                    workflow_group,
                    i,
                    wait_time,
                    timeout,
                    log_message
                )
                
                if not success:
                    log_message(f"Execution {i + 1} failed!")
                else:
                    log_message(f"Execution {i + 1} completed successfully!")
            
            log_message("\n\nAll manual executions completed!")
        except Exception as e:
            log_message(f"\n\nERROR in manual execution: {e}")
    
    current_process_thread = threading.Thread(target=execute_thread)
    current_process_thread.start()
    
    return "Manual execution started!", get_logs()

# ==================== Gradio Interface ====================

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Smart Vision - ComfyUI Video Generation Workflow Manager", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 Smart Vision - ComfyUI Video Generation Workflow Manager")
        gr.Markdown("**Smart Vision** - Advanced AI Video Generation System")
        gr.Markdown("Multi-scene video generation tool based on ComfyUI, supporting character setup, scene generation, voice synthesis and background music")
        
        with gr.Tabs():
            # Tab 1: Multi-Scene Video Generation
            with gr.Tab("🎬 Multi-Scene Video Generation"):
                gr.Markdown("## Multi-Scene Video Generation")
                gr.Markdown("Automatically generate multi-scene videos based on prompt files, supporting character setup, voice dialogue and background music")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Basic Settings")
                        
                        character_image = gr.File(
                            label="Character Image",
                            file_types=["image"]
                        )
                        
                        voice_audio_file = gr.File(
                            label="Voice Audio File (for Workflow Group 3 - speaking video)",
                            file_types=["audio"]
                        )
                        
                        with gr.Row():
                            video_width = gr.Number(
                                label="Video Width",
                                value=512,
                                precision=0
                            )
                            video_height = gr.Number(
                                label="Video Height", 
                                value=512,
                                precision=0
                            )
                        
                        gr.Markdown("### 📝 Prompt File Paths")
                        gr.Markdown("Upload your prompt files using the file browsers below. The paths will be automatically filled.")
                        
                        type_prompts_file = gr.File(
                            label="type_prompts.txt",
                            file_types=[".txt"],
                            file_count="single"
                        )
                        type_prompts_path = gr.Textbox(
                            label="type_prompts.txt Path (auto-filled)",
                            interactive=False,
                            visible=False
                        )
                        
                        image_prompts_file = gr.File(
                            label="image_prompts.txt",
                            file_types=[".txt"],
                            file_count="single"
                        )
                        image_prompts_path = gr.Textbox(
                            label="image_prompts.txt Path (auto-filled)",
                            interactive=False,
                            visible=False
                        )
                        
                        video_prompts_file = gr.File(
                            label="video_prompts.txt",
                            file_types=[".txt"],
                            file_count="single"
                        )
                        video_prompts_path = gr.Textbox(
                            label="video_prompts.txt Path (auto-filled)",
                            interactive=False,
                            visible=False
                        )
                        
                        sound_prompts_file = gr.File(
                            label="sound_prompts.txt",
                            file_types=[".txt"],
                            file_count="single"
                        )
                        sound_prompts_path = gr.Textbox(
                            label="sound_prompts.txt Path (auto-filled)",
                            interactive=False,
                            visible=False
                        )
                        
                        audio_prompts_file = gr.File(
                            label="audio_prompts.txt",
                            file_types=[".txt"],
                            file_count="single"
                        )
                        audio_prompts_path = gr.Textbox(
                            label="audio_prompts.txt Path (auto-filled)",
                            interactive=False,
                            visible=False
                        )
                        
                        gr.Markdown("### 📄 File Preview")
                        file_preview = gr.Textbox(
                            label="Selected File Preview",
                            lines=8,
                            interactive=False,
                            placeholder="Select a file to preview its content"
                        )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            multi_scene_comfyui_url = gr.Textbox(
                                label="ComfyUI URL",
                                value="http://127.0.0.1:8000"
                            )
                            multi_scene_wait_time = gr.Number(
                                label="Wait Time Between Workflows (seconds)",
                                value=5,
                                minimum=0
                            )
                            multi_scene_timeout = gr.Number(
                                label="Workflow Timeout (seconds)",
                                value=999999,
                                minimum=30
                            )
                        
                        with gr.Row():
                            execute_multi_scene_btn = gr.Button("▶️ Start Execution", variant="primary", size="lg")
                            stop_multi_scene_btn = gr.Button("⏹️ Stop Execution", variant="stop", size="lg")
                        
                        multi_scene_status = gr.Textbox(
                            label="Execution Status",
                            lines=2,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        multi_scene_logs = gr.Textbox(
                            label="Execution Logs",
                            lines=35,
                            max_lines=35,
                            interactive=False,
                            autoscroll=True
                        )
                        
                        with gr.Row():
                            refresh_multi_scene_btn = gr.Button("🔄 Refresh Logs")
                            clear_multi_scene_btn = gr.Button("🧹 Clear Logs")
                            memory_status_btn = gr.Button("📊 Memory Status")
                        
                        memory_status_display = gr.Textbox(
                            label="Memory Status",
                            lines=4,
                            interactive=False,
                            placeholder="Click 'Memory Status' to check current memory usage"
                        )
                
                # Event handlers for multi-scene generation
                execute_multi_scene_btn.click(
                    execute_multi_scene_workflow,
                    inputs=[
                        character_image,
                        voice_audio_file,
                        video_width,
                        video_height,
                        type_prompts_path,
                        image_prompts_path,
                        video_prompts_path,
                        sound_prompts_path,
                        audio_prompts_path,
                        multi_scene_comfyui_url,
                        multi_scene_wait_time,
                        multi_scene_timeout
                    ],
                    outputs=[multi_scene_status]
                )
                
                # File upload event handlers to auto-fill paths and preview content
                type_prompts_file.change(
                    lambda file_obj: (update_file_path(file_obj), preview_file_content(file_obj)),
                    inputs=[type_prompts_file],
                    outputs=[type_prompts_path, file_preview]
                )
                
                image_prompts_file.change(
                    lambda file_obj: (update_file_path(file_obj), preview_file_content(file_obj)),
                    inputs=[image_prompts_file],
                    outputs=[image_prompts_path, file_preview]
                )
                
                video_prompts_file.change(
                    lambda file_obj: (update_file_path(file_obj), preview_file_content(file_obj)),
                    inputs=[video_prompts_file],
                    outputs=[video_prompts_path, file_preview]
                )
                
                sound_prompts_file.change(
                    lambda file_obj: (update_file_path(file_obj), preview_file_content(file_obj)),
                    inputs=[sound_prompts_file],
                    outputs=[sound_prompts_path, file_preview]
                )
                
                audio_prompts_file.change(
                    lambda file_obj: (update_file_path(file_obj), preview_file_content(file_obj)),
                    inputs=[audio_prompts_file],
                    outputs=[audio_prompts_path, file_preview]
                )
                
                refresh_multi_scene_btn.click(
                    get_status_messages,
                    outputs=[multi_scene_logs]
                )
                
                clear_multi_scene_btn.click(
                    clear_status,
                    outputs=[multi_scene_logs]
                )
                
                memory_status_btn.click(
                    get_memory_status,
                    outputs=[memory_status_display]
                )
            
            # Tab 2: Batch Processing (Original)
            with gr.Tab("📦 Batch Processing", visible=False):
                gr.Markdown("## Batch Processing")
                gr.Markdown("Upload type_prompts.txt file to automatically execute workflow chains")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        type_prompts_upload = gr.File(
                            label="Upload type_prompts.txt",
                            file_types=[".txt"]
                        )
                        upload_status = gr.Textbox(
                            label="Upload Status",
                            lines=2,
                            interactive=False
                        )
                        file_preview = gr.Textbox(
                            label="File Preview",
                            lines=5,
                            interactive=False
                        )
                        
                        with gr.Accordion("Settings", open=True):
                            batch_comfyui_url = gr.Textbox(
                                label="ComfyUI URL",
                                value="http://127.0.0.1:8000"
                            )
                            batch_wait_time = gr.Number(
                                label="Wait Time Between Workflows (seconds)",
                                value=5,
                                minimum=0
                            )
                            batch_timeout = gr.Number(
                                label="Workflow Timeout (seconds)",
                                value=300,
                                minimum=30
                            )
                            batch_output_dir = gr.Textbox(
                                label="Output Directory",
                                value="output"
                            )
                            batch_input_dir = gr.Textbox(
                                label="Input Directory",
                                value="input"
                            )
                        
                        with gr.Row():
                            start_btn = gr.Button("Start Processing", variant="primary")
                            stop_btn = gr.Button("Stop", variant="stop")
                            refresh_btn = gr.Button("Refresh Logs")
                            memory_btn = gr.Button("📊 Memory Status")
                        
                        batch_memory_display = gr.Textbox(
                            label="Memory Status",
                            lines=3,
                            interactive=False,
                            placeholder="Click 'Memory Status' to check current memory usage"
                        )
                    
                    with gr.Column(scale=2):
                        batch_logs = gr.Textbox(
                            label="Processing Logs",
                            lines=30,
                            max_lines=30,
                            interactive=False,
                            autoscroll=True
                        )
                
                # Event handlers
                type_prompts_upload.change(
                    upload_type_prompts,
                    inputs=[type_prompts_upload],
                    outputs=[upload_status, file_preview]
                )
                
                start_btn.click(
                    start_batch_processing,
                    inputs=[
                        type_prompts_upload,
                        batch_comfyui_url,
                        batch_wait_time,
                        batch_timeout,
                        batch_output_dir,
                        batch_input_dir
                    ],
                    outputs=[upload_status, batch_logs]
                )
                
                stop_btn.click(
                    stop_batch_processing,
                    outputs=[upload_status, batch_logs]
                )
                
                refresh_btn.click(
                    refresh_logs,
                    outputs=[batch_logs]
                )
                
                memory_btn.click(
                    get_memory_status,
                    outputs=[batch_memory_display]
                )
            
            # Tab 3: Configuration Manager
            with gr.Tab("⚙️ Workflow Configuration"):
                gr.Markdown("## Workflow Group Configuration Management")
                gr.Markdown("Manage different types of workflow group mappings")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ➕ Add/Edit Workflow Group")
                        
                        config_type_num = gr.Textbox(
                            label="Type Number",
                            placeholder="e.g., 1, 2, 3"
                        )
                        config_group_name = gr.Textbox(
                            label="Group Name",
                            placeholder="e.g., Workflow Group 1 - Non-character Scene Generation"
                        )
                        config_workflows = gr.Textbox(
                            label="Workflow File Paths (one per line)",
                            placeholder="Generate_none_character_scene.json\nimage_to_video.json\ngenerate_sound_by_video.json",
                            lines=5
                        )
                        
                        with gr.Row():
                            add_btn = gr.Button("➕ Add/Update Group", variant="primary")
                            delete_btn = gr.Button("🗑️ Delete Group", variant="stop")
                        
                        config_status = gr.Textbox(
                            label="Operation Status",
                            lines=3,
                            interactive=False
                        )
                        
                        gr.Markdown("### 🔧 Preset Configuration")
                        with gr.Row():
                            load_preset_btn = gr.Button("📁 Load Preset Configuration", variant="secondary")
                            validate_btn = gr.Button("✅ Validate Configuration")
                        
                        validation_result = gr.Textbox(
                            label="Validation Result",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Current Configuration")
                        
                        config_display = gr.Textbox(
                            label="Configuration JSON",
                            lines=20,
                            interactive=False
                        )
                        
                        refresh_config_btn = gr.Button("🔄 Refresh Configuration")
                
                # Event handlers
                add_btn.click(
                    add_workflow_group,
                    inputs=[config_type_num, config_group_name, config_workflows],
                    outputs=[config_status, config_display]
                )
                
                delete_btn.click(
                    delete_workflow_group,
                    inputs=[config_type_num],
                    outputs=[config_status, config_display]
                )
                
                load_preset_btn.click(
                    load_preset_workflow_groups,
                    outputs=[config_display]
                )
                
                validate_btn.click(
                    validate_config,
                    outputs=[validation_result]
                )
                
                refresh_config_btn.click(
                    lambda: load_config_display()[0],
                    outputs=[config_display]
                )
                
                # Load config on startup
                app.load(
                    lambda: load_config_display()[0],
                    outputs=[config_display]
                )
            
            # Tab 4: Manual Execution
            with gr.Tab("🔧 Manual Execution"):
                gr.Markdown("## Manual Workflow Execution")
                gr.Markdown("Manually execute specific workflow groups for testing")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        manual_workflow_group = gr.Dropdown(
                            label="Select Workflow Group",
                            choices=get_workflow_groups(),
                            interactive=True
                        )
                        manual_num_executions = gr.Number(
                            label="Number of Executions",
                            value=1,
                            minimum=1,
                            maximum=100
                        )
                        manual_image_prompt = gr.Textbox(
                            label="Image Prompt (optional)",
                            placeholder="Override CLIPTextEncode text for image workflows"
                        )
                        manual_video_prompt = gr.Textbox(
                            label="Video Prompt (optional)",
                            placeholder="Override CLIPTextEncode text for video workflows (WanImageToVideo)"
                        )
                        manual_sound_prompt = gr.Textbox(
                            label="Sound Prompt (optional)",
                            placeholder="Override sound generation prompt for audio workflows"
                        )
                        manual_audio_prompt = gr.Textbox(
                            label="Audio Prompt (optional)",
                            placeholder="Override character dialogue for audio workflows"
                        )
                        manual_character_image = gr.File(
                            label="Character Image (optional)",
                            file_types=["image"],
                            file_count="single"
                        )
                        manual_voice_audio = gr.File(
                            label="Voice Audio File (optional - for speaking video workflows)",
                            file_types=["audio"],
                            file_count="single"
                        )
                        
                        with gr.Accordion("Settings", open=True):
                            manual_comfyui_url = gr.Textbox(
                                label="ComfyUI URL",
                                value="http://127.0.0.1:8000"
                            )
                            manual_wait_time = gr.Number(
                                label="Wait Time Between Workflows (seconds)",
                                value=5,
                                minimum=0
                            )
                            manual_timeout = gr.Number(
                                label="Workflow Timeout (seconds)",
                                value=300,
                                minimum=30
                            )
                            manual_output_dir = gr.Textbox(
                                label="Output Directory",
                                value="output"
                            )
                            manual_input_dir = gr.Textbox(
                                label="Input Directory",
                                value="input"
                            )
                        
                        with gr.Row():
                            execute_btn = gr.Button("▶️ Start Execution", variant="primary")
                            refresh_manual_btn = gr.Button("🔄 Refresh Logs")
                            manual_memory_btn = gr.Button("📊 Memory Status")
                        
                        manual_memory_display = gr.Textbox(
                            label="Memory Status",
                            lines=3,
                            interactive=False,
                            placeholder="Click 'Memory Status' to check current memory usage"
                        )
                        
                        manual_status = gr.Textbox(
                            label="Execution Status",
                            lines=2,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        manual_logs = gr.Textbox(
                            label="Execution Logs",
                            lines=30,
                            max_lines=30,
                            interactive=False,
                            autoscroll=True
                        )
                
                # Event handlers
                execute_btn.click(
                    execute_manual_workflow,
                    inputs=[
                        manual_workflow_group,
                        manual_num_executions,
                        manual_comfyui_url,
                        manual_wait_time,
                        manual_timeout,
                        manual_output_dir,
                        manual_input_dir,
                        manual_image_prompt,
                        manual_video_prompt,
                        manual_sound_prompt,
                        manual_audio_prompt,
                        manual_character_image,
                        manual_voice_audio
                    ],
                    outputs=[manual_status, manual_logs]
                )
                
                refresh_manual_btn.click(
                    refresh_logs,
                    outputs=[manual_logs]
                )
                
                manual_memory_btn.click(
                    get_memory_status,
                    outputs=[manual_memory_display]
                )
                
                # Refresh dropdown on tab open
                app.load(
                    get_workflow_groups,
                    outputs=[manual_workflow_group]
                )
        
        # Auto-refresh status for multi-scene generation
        # Note: every parameter is not supported in older Gradio versions
        # app.load(
        #     get_status_messages,
        #     outputs=[multi_scene_logs],
        #     every=2
        # )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

