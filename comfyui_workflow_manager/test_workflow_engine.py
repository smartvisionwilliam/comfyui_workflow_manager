"""
Test script for workflow_engine.py
This script verifies the core functionality without requiring ComfyUI to be running.
"""

import os
import json
from workflow_engine import (
    load_workflow_config,
    save_workflow_config,
    parse_type_prompts,
    set_comfyui_url,
    set_directories
)

def test_config_operations():
    """Test configuration loading and saving"""
    print("Testing configuration operations...")
    
    # Test loading config
    config = load_workflow_config("config.json")
    print(f"✓ Loaded config with {len(config)} workflow groups")
    
    for type_num, group_data in config.items():
        name = group_data.get("name", "Unknown")
        workflows = group_data.get("workflows", [])
        print(f"  Type {type_num}: {name} ({len(workflows)} workflows)")
        for wf in workflows:
            exists = "✓" if os.path.exists(wf) else "✗"
            print(f"    {exists} {wf}")
    
    # Test saving config
    test_config = {
        "99": {
            "name": "Test Workflow Group",
            "workflows": ["test/workflow.json"]
        }
    }
    
    if save_workflow_config(test_config, "test_config.json"):
        print("✓ Config save successful")
        
        # Load it back
        loaded = load_workflow_config("test_config.json")
        if loaded == test_config:
            print("✓ Config load/save round-trip successful")
        else:
            print("✗ Config mismatch after load/save")
        
        # Clean up
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")
    else:
        print("✗ Config save failed")
    
    print()

def test_type_prompts_parsing():
    """Test type_prompts.txt parsing"""
    print("Testing type_prompts.txt parsing...")
    
    type_prompts_path = "input/prompts/type_prompts.txt"
    
    if not os.path.exists(type_prompts_path):
        print(f"✗ File not found: {type_prompts_path}")
        return
    
    type_numbers = parse_type_prompts(type_prompts_path)
    print(f"✓ Parsed {len(type_numbers)} type numbers")
    print(f"  Type numbers: {type_numbers}")
    print()

def test_settings():
    """Test settings configuration"""
    print("Testing settings configuration...")
    
    # Test URL setting
    set_comfyui_url("http://127.0.0.1:8000")
    print("✓ ComfyUI URL set")
    
    # Test directory settings
    set_directories("output", "input")
    print("✓ Directories set")
    print()

def test_workflow_file_validation():
    """Test workflow file validation"""
    print("Testing workflow file validation...")
    
    config = load_workflow_config("config.json")
    
    all_valid = True
    missing_files = []
    
    for type_num, group_data in config.items():
        workflows = group_data.get("workflows", [])
        for wf_path in workflows:
            if not os.path.exists(wf_path):
                all_valid = False
                missing_files.append(wf_path)
    
    if all_valid:
        print("✓ All workflow files exist")
    else:
        print(f"✗ {len(missing_files)} workflow file(s) missing:")
        for wf in missing_files:
            print(f"  - {wf}")
    
    print()

def test_integration():
    """Test integration of components"""
    print("Testing integration...")
    
    # Load config
    config = load_workflow_config("config.json")
    
    # Parse type prompts
    type_numbers = parse_type_prompts("input/prompts/type_prompts.txt")
    
    if not type_numbers:
        print("✗ No type numbers found in type_prompts.txt")
        return
    
    # Verify all type numbers have corresponding config
    missing_configs = []
    for type_num in type_numbers:
        if str(type_num) not in config:
            missing_configs.append(type_num)
    
    if missing_configs:
        print(f"✗ {len(missing_configs)} type number(s) missing from config:")
        for tn in missing_configs:
            print(f"  - Type {tn}")
    else:
        print("✓ All type numbers have corresponding configuration")
    
    # Show execution plan
    print("\nExecution Plan:")
    for idx, type_num in enumerate(type_numbers):
        group_data = config.get(str(type_num))
        if group_data:
            name = group_data.get("name", "Unknown")
            workflows = group_data.get("workflows", [])
            print(f"  {idx + 1}. Type {type_num}: {name} ({len(workflows)} workflows)")
    
    print()

def main():
    """Run all tests"""
    print("="*60)
    print("Workflow Engine Test Suite")
    print("="*60)
    print()
    
    test_config_operations()
    test_type_prompts_parsing()
    test_settings()
    test_workflow_file_validation()
    test_integration()
    
    print("="*60)
    print("Test Suite Complete")
    print("="*60)
    print()
    print("Note: These tests verify configuration and file operations only.")
    print("To test actual workflow execution, ComfyUI must be running.")
    print("Use the Gradio application (app.py) for full integration testing.")

if __name__ == "__main__":
    main()


