import json
import gzip
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from shapely.geometry import box
import pycocotools.mask as mask_utils
from src.constants import PANELS_IN_SCOPE, SCORE_THRESHOLD

@dataclass
class ImageInstances:
    file_name: str
    height: int
    width: int
    instances: List['Instance']

@dataclass
class Instance:
    bbox: Tuple[int, int, int, int]
    category_id: int
    segmentation: Dict
    id: int
    score: float = 1.0

def process_damage_instances(damage_instances: List[ImageInstances], 
                           panel_instances: List[ImageInstances]) -> Dict:
    """
    Process damage instances by matching them to panels and merging same-type damages.
    Now correctly stores both panel and damage information.
    """
    # Create lookup dictionary for panel instances by filename
    panel_dict = {inst.file_name: inst for inst in panel_instances}
    results = {}
    
    for damage_img in tqdm(damage_instances):
        if damage_img.file_name not in panel_dict:
            continue
            
        panel_img = panel_dict[damage_img.file_name]
        image_matches = {}
        
        # Filter panels and damages
        valid_panels = [
            panel for panel in panel_img.instances 
            if panel.category_id in PANELS_IN_SCOPE
        ]
        
        valid_damages = [
            damage for damage in damage_img.instances 
            if damage.score >= SCORE_THRESHOLD
        ]
        
        # Group damages by category
        damage_by_category = {}
        for damage in valid_damages:
            if damage.category_id not in damage_by_category:
                damage_by_category[damage.category_id] = []
            damage_by_category[damage.category_id].append(damage)
        
        # For each panel, find and merge contained damages
        for panel in valid_panels:
            panel_mask = mask_utils.decode(panel.segmentation)
            panel_damages = {}
            
            # Process each damage category
            for category_id, category_damages in damage_by_category.items():
                contained_damages = []
                
                # Check which damages are contained in this panel
                for damage in category_damages:
                    damage_mask = mask_utils.decode(damage.segmentation)
                    if damage_mask.shape != panel_mask.shape:
                        print(f"Shape mismatch: {damage_mask.shape}, {panel_mask.shape}")
                        continue

                    if is_damage_in_panel(damage_mask, panel_mask):
                        contained_damages.append(damage)
                
                # If we found damages in this category, merge them
                if contained_damages:
                    merged_damage = merge_damage_instances(contained_damages)
                    if merged_damage:
                        panel_damages[category_id] = {
                            'panel': panel,  # Store panel instance
                            'damage': merged_damage  # Store damage instance
                        }
            
            if panel_damages:
                image_matches[panel.id] = panel_damages
        
        if image_matches:
            results[damage_img.file_name] = image_matches
    
    return results

def is_damage_in_panel(damage_mask: np.ndarray, panel_mask: np.ndarray) -> bool:
    """
    Check if damage mask is contained within panel mask.
    Uses 90% overlap threshold for robustness.
    """
    intersection = np.logical_and(damage_mask, panel_mask)
    return np.sum(intersection) >= 0.9 * np.sum(damage_mask)

def merge_damage_instances(damages: List[Instance]) -> Optional[Instance]:
    """
    Merge multiple damage instances of the same category into a single instance.
    
    Args:
        damages: List of damage instances to merge
        
    Returns:
        New Instance with merged mask or None if input is empty
    """
    if not damages:
        return None
    
    # Initialize with first damage mask
    merged_mask = mask_utils.decode(damages[0].segmentation)
    
    # Merge all other masks
    for damage in damages[1:]:
        damage_mask = mask_utils.decode(damage.segmentation)
        merged_mask = np.logical_or(merged_mask, damage_mask)
    
    # Compute bounding box for merged mask
    rows = np.any(merged_mask, axis=1)
    cols = np.any(merged_mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Create new instance with merged data
    return Instance(
        bbox=(x1, y1, x2 - x1 + 1, y2 - y1 + 1),
        category_id=damages[0].category_id,
        segmentation=mask_utils.encode(np.asfortranarray(merged_mask)),
        id=damages[0].id,  # Use first damage's ID
        score=max(d.score for d in damages)  # Use highest confidence score
    )


def convert_to_serializable(obj):
    """
    Convert object to JSON serializable format.
    Handles NumPy types and other non-serializable objects.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return list(obj)
    return obj

def save_results(results: Dict, output_dir: str):
    """
    Save damage analysis results in an efficient format.
    Uses compressed numpy format for masks and JSON for metadata.
    
    Args:
        results: Results dictionary from process_damage_instances
        output_dir: Directory to save the results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a metadata dictionary without the binary mask data
    metadata = {}
    compressed_masks = {}
    
    for image_id, panel_data in results.items():
        metadata[image_id] = {}
        compressed_masks[image_id] = {}
        
        for panel_id, damage_categories in panel_data.items():
            # Convert panel_id to string for JSON compatibility
            panel_id_str = str(panel_id)
            metadata[image_id][panel_id_str] = {}
            compressed_masks[image_id][panel_id_str] = {}
            
            for category_id, damage_inst in damage_categories.items():
                # Convert category_id to string for JSON compatibility
                category_id_str = str(category_id)
                
                # Store metadata with explicit type conversion
                meta_entry = {
                    'bbox': [convert_to_serializable(x) for x in damage_inst.bbox],
                    'category_id': convert_to_serializable(damage_inst.category_id),
                    'id': convert_to_serializable(damage_inst.id),
                    'score': convert_to_serializable(damage_inst.score)
                }
                metadata[image_id][panel_id_str][category_id_str] = meta_entry
                
                # Store compressed mask data
                compressed_masks[image_id][panel_id_str][category_id_str] = damage_inst.segmentation

    # Save metadata as JSON
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save compressed masks using gzip pickle
    with gzip.open(output_path / 'masks.pkl.gz', 'wb') as f:
        pickle.dump(compressed_masks, f)

def load_results(input_dir: str) -> Dict:
    """
    Load damage analysis results saved by save_results.
    
    Args:
        input_dir: Directory containing the saved results
        
    Returns:
        Reconstructed results dictionary
    """
    input_path = Path(input_dir)
    
    # Load metadata
    with open(input_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load compressed masks
    with gzip.open(input_path / 'masks.pkl.gz', 'rb') as f:
        compressed_masks = pickle.load(f)
    
    # Reconstruct the full results dictionary
    results = {}
    for image_id, panel_data in metadata.items():
        results[image_id] = {}
        
        for panel_id, damage_categories in panel_data.items():
            results[image_id][int(panel_id)] = {}
            
            for category_id, meta_entry in damage_categories.items():
                # Create Instance with both metadata and mask
                damage_inst = Instance(
                    bbox=tuple(meta_entry['bbox']),  # Convert list back to tuple
                    category_id=meta_entry['category_id'],
                    segmentation=compressed_masks[image_id][panel_id][category_id],
                    id=meta_entry['id'],
                    score=meta_entry['score']
                )
                results[image_id][int(panel_id)][int(category_id)] = damage_inst
    
    return results