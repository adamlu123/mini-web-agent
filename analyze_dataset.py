import json
import os
import numpy as np
from transformers import AutoTokenizer
from collections import Counter

def analyze():
    json_path = 'LlamaFactory/data/web_agent_state_debug_latest_0623_bundle/web_agent_state_debug_latest_0623.json'
    manifest_path = 'LlamaFactory/data/web_agent_state_debug_latest_0623_bundle/manifest.json'
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    bundle_dir = os.path.dirname(json_path)

    # 1) Stats
    total_examples = len(data)
    aux_types = []
    image_refs = []
    all_images_in_data = set()
    
    for item in data:
        aux_type = item.get('aux_type', 'agent_interaction')
        aux_types.append(aux_type)
        
        images = item.get('images', [])
        image_refs.extend(images)
        for img in images:
            all_images_in_data.add(img)

    aux_type_counts = Counter(aux_types)
    unique_images = len(all_images_in_data)
    total_image_refs = len(image_refs)
    
    # Missing images check
    missing_images = 0
    for img in all_images_in_data:
        img_path = os.path.join(bundle_dir, img)
        if not os.path.exists(img_path):
            missing_images += 1

    print(f"Total examples: {total_examples}")
    print(f"Aux types count: {dict(aux_type_counts)}")
    print(f"Total image references: {total_image_refs}")
    print(f"Unique images: {unique_images}")
    print(f"Missing images: {missing_images}")

    # 2) Tokenizer stats
    try:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True) # Fallback if 3.5 doesn't exist? No, follow instructions.
    except:
        print("Warning: Qwen/Qwen3.5-9B not found, trying Qwen/Qwen2.5-7B-Instruct as placeholder for template testing if requested fails.")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True)
    except Exception as e:
        print(f"Could not load Qwen/Qwen3.5-9B: {e}")
        # Try to use a similar one just to get the logic if 3.5 is not available
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
        print("Used Qwen/Qwen2.5-7B-Instruct instead.")

    lengths = []
    grouped_lengths = {at: [] for at in aux_type_counts.keys()}

    # 3) System prompt check
    system_check_counts = {
        'Task Reflection Tool': 0,
        '<done>true</done>': 0,
        'open_browser_session': 0
    }

    for item in data:
        aux_type = item.get('aux_type', 'agent_interaction')
        system_prompt = item.get('system', '')
        
        for key in system_check_counts:
            if key in system_prompt:
                system_check_counts[key] += 1
        
        # Prepare messages for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        conversations = item.get('conversations', [])
        for turn in conversations:
            messages.append({"role": turn["from"], "content": turn["value"]})
        
        try:
            # Multi-modal placeholder handling: the prompt says "多模态图片占位可以保留文本 <image>"
            # Most Qwen templates handle it if we just pass as text or use the processor.
            # Here we just treat <image> as text in the content.
            tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            length = len(tokenized)
        except Exception as e:
            # Fallback
            full_text = ""
            for msg in messages:
                full_text += f"{msg['role']}: {msg['content']}\n"
            length = len(tokenizer.encode(full_text))
            
        lengths.append(length)
        grouped_lengths[aux_type].append(length)

    def print_stats(name, data_list):
        if not data_list:
            print(f"{name}: No data")
            return
        print(f"Stats for {name}:")
        print(f"  Min: {np.min(data_list)}")
        print(f"  Median: {np.median(data_list)}")
        print(f"  Max: {np.max(data_list)}")
        print(f"  Mean: {np.mean(data_list):.2f}")
        if name == "Overall":
            print(f"  P90: {np.percentile(data_list, 90)}")
            print(f"  P95: {np.percentile(data_list, 95)}")
            print(f"  P99: {np.percentile(data_list, 99)}")

    print_stats("Overall", lengths)
    for at, l_list in grouped_lengths.items():
        print_stats(f"Aux Type: {at}", l_list)

    print(f"System Prompt keyword counts: {system_check_counts}")

analyze()
