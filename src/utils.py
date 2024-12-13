import re
import os
import json
import cv2 
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pycocotools.mask as mask_utils


def compute_damage_features(
    matched_results: Dict,
    panel_name_map: Dict[int, str],
    damage_type_map: Dict[int, str]
) -> pd.DataFrame:
    """
    Compute features from damage and panel masks.
    Updated to handle new data structure with separate panel and damage info.
    
    Args:
        matched_results: Dictionary of format:
            {image_path: {panel_id: {damage_cat_id: {'panel': panel_inst, 'damage': damage_inst}}}}
        panel_name_map: Dictionary mapping panel category IDs to names
        damage_type_map: Dictionary mapping damage category IDs to names
    """
    data = []
    
    for image_path, panels in matched_results.items():
        for panel_id, damage_categories in panels.items():
            for damage_cat_id, instances in damage_categories.items():
                # Get panel and damage instances
                panel_inst = instances['panel']
                damage_inst = instances['damage']
                
                # Decode masks
                panel_mask = mask_utils.decode(panel_inst.segmentation)
                damage_mask = mask_utils.decode(damage_inst.segmentation)
                
                # Compute features
                damage_size = compute_damage_size(damage_mask, panel_mask)
                damage_on_edge = check_damage_on_edge(damage_mask, panel_mask)
                num_components = count_components(damage_mask)
                rel_x, rel_y = compute_relative_position(damage_mask, panel_mask)
                
                # Check panel visibility
                whole_panel_visible = not touches_image_edge(panel_mask)
                
                data.append({
                    'image_path': image_path,
                    'panel': panel_name_map[panel_inst.category_id],
                    'damage_type': damage_type_map[damage_inst.category_id],
                    'damage_size': damage_size,
                    'damage_on_edge': damage_on_edge,
                    'whole_panel_visible': whole_panel_visible,
                    'num_components': num_components,
                    'relative_x': rel_x,
                    'relative_y': rel_y
                })
    
    return pd.DataFrame(data)

# No changes needed to these helper functions as they work with masks directly
def compute_damage_size(damage_mask: np.ndarray, panel_mask: np.ndarray) -> float:
    """Compute what percentage of panel area is covered by damage."""
    panel_area = np.sum(panel_mask)
    if panel_area == 0:
        return 0.0
    
    damage_area = np.sum(np.logical_and(damage_mask, panel_mask))
    return (damage_area / panel_area) * 100

def check_damage_on_edge(damage_mask: np.ndarray, panel_mask: np.ndarray) -> bool:
    """Check if damage touches the edge of the panel using simple erosion."""
    kernel = np.ones((3, 3), np.uint8)
    eroded_panel = cv2.erode(panel_mask.astype(np.uint8), kernel, iterations=1)
    panel_edge = panel_mask.astype(np.uint8) - eroded_panel
    return np.any(np.logical_and(damage_mask, panel_edge))

def compute_relative_position(damage_mask: np.ndarray, panel_mask: np.ndarray) -> Tuple[float, float]:
    """Compute relative position of damage centroid within panel."""
    damage_y, damage_x = np.where(damage_mask)
    if len(damage_x) == 0:
        return 0.5, 0.5
        
    damage_centroid = (np.mean(damage_x), np.mean(damage_y))
    
    panel_y, panel_x = np.where(panel_mask)
    if len(panel_x) == 0:
        return 0.5, 0.5
        
    x_min, x_max = np.min(panel_x), np.max(panel_x)
    y_min, y_max = np.min(panel_y), np.max(panel_y)
    
    rel_x = (damage_centroid[0] - x_min) / (x_max - x_min + 1e-6)
    rel_y = (damage_centroid[1] - y_min) / (y_max - y_min + 1e-6)
    
    return rel_x, rel_y

def count_components(mask: np.ndarray) -> int:
    """Count number of separate damage components."""
    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return num_labels - 1  # Subtract 1 for background

# Alternative even simpler approach without using cv2:
def check_damage_on_edge_simple(damage_mask: np.ndarray, panel_mask: np.ndarray) -> bool:
    """
    Simpler version using only NumPy operations.
    Checks if any damage pixels are on panel boundary.
    """
    # Find panel boundary pixels
    panel_boundary = np.zeros_like(panel_mask)
    panel_boundary[:-1, :] |= panel_mask[1:, :] != panel_mask[:-1, :]  # Vertical edges
    panel_boundary[:, :-1] |= panel_mask[:, 1:] != panel_mask[:, :-1]  # Horizontal edges
    
    # Check if damage intersects with boundary
    return np.any(np.logical_and(damage_mask, panel_boundary))

def touches_image_edge(mask: np.ndarray) -> bool:
    """
    Check if mask touches the edge of the image.
    Uses efficient numpy operations checking only the border.
    """
    height, width = mask.shape
    border = np.concatenate([
        mask[0, :],     # top edge
        mask[-1, :],    # bottom edge
        mask[1:-1, 0],  # left edge (excluding corners)
        mask[1:-1, -1]  # right edge (excluding corners)
    ])
    return np.any(border)

# Example usage
def process_and_analyze_damages(
    matched_results: Dict,
    panel_name_map: Dict[int, str],
    damage_type_map: Dict[int, str],
    output_csv: str = None
) -> pd.DataFrame:
    """
    Process damage results and optionally save to CSV.
    
    Args:
        matched_results: Results from damage matching
        panel_name_map: Mapping of panel IDs to names
        damage_type_map: Mapping of damage type IDs to names
        output_csv: Optional path to save CSV file
        
    Returns:
        DataFrame with damage analysis
    """
    # Compute features
    df = compute_damage_features(
        matched_results,
        panel_name_map,
        damage_type_map
    )
    
    # Save if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
    
    return df

def visualize_damage_analysis(
    image_path: str,
    matched_results: Dict,
    features_df: Optional[pd.DataFrame] = None,
    figsize: tuple = (20, 10)
) -> None:
    """
    Visualize damage analysis with original image and mask overlays.
    Now correctly handles separate panel and damage masks.
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot original image
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Plot image with masks overlay
    ax2.imshow(img)
    
    # Generate colors for different damage types
    unique_damage_types = {
        damage_dict['damage'].category_id 
        for panel_damages in matched_results.values() 
        for damage_dict in panel_damages.values()
    }
    damage_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_damage_types)))
    color_map = dict(zip(unique_damage_types, damage_colors))
    
    # For each panel
    for panel_id, damage_dict in matched_results.items():
        for damage_cat_id, instances_dict in damage_dict.items():
            # Get panel mask from panel instance
            panel_mask = mask_utils.decode(instances_dict['panel'].segmentation)
            damage_mask = mask_utils.decode(instances_dict['damage'].segmentation)
            
            # Create panel overlay
            panel_overlay = np.zeros_like(img)
            panel_overlay[panel_mask > 0] = [255, 255, 255]  # White for panels
            ax2.imshow(panel_overlay, alpha=0.2)
            
            # Create damage overlay
            damage_overlay = np.zeros_like(img)
            color = (np.array(color_map[instances_dict['damage'].category_id][:3]) * 255).astype(int)
            damage_overlay[damage_mask > 0] = color
            ax2.imshow(damage_overlay, alpha=0.5)
            
            # Add centroid marker
            #y_coords, x_coords = np.where(damage_mask)
            #if len(x_coords) > 0:
            #    centroid_x = np.mean(x_coords)
            #    centroid_y = np.mean(y_coords)
            #    ax2.plot(centroid_x, centroid_y, 'r*', markersize=10)
    
    ax2.set_title("Damage Analysis Overlay")
    ax2.axis('off')
    
    
    plt.tight_layout()
    plt.show()

def analyze_single_image(
    image_path: str,
    matched_results: Dict,
    features_df: pd.DataFrame
) -> None:
    """
    Perform complete analysis visualization for a single image.
    """
    if image_path not in matched_results:
        print(f"No analysis results found for image: {image_path}")
        return
    
    # Visualize
    visualize_damage_analysis(
        image_path,
        matched_results[image_path],  # Pass just the results for this image
        features_df
    )
    
    # Print detailed features
    print_image_features(image_path, features_df)

def print_image_features(image_path: str, features_df: pd.DataFrame) -> None:
    """
    Print detailed features for a specific image.
    
    Args:
        image_path: Path to image to analyze
        features_df: DataFrame containing computed features
    """
    image_features = features_df[features_df['image_path'] == image_path]
    
    if image_features.empty:
        print(f"No features found for image: {image_path}")
        return
    
    print(f"\nAnalysis for image: {image_path}")
    print("-" * 50)
    
    # Group by panel
    for panel, panel_data in image_features.groupby('panel'):
        print(f"\nPanel: {panel}")
        print("=" * 30)
        
        for _, damage in panel_data.iterrows():
            print(f"\nDamage Type: {damage['damage_type']}")
            print(f"Size: {damage['damage_size']:.2f}%")
            print(f"Number of components: {damage['num_components']}")
            print(f"Position: x={damage['relative_x']:.2f}, y={damage['relative_y']:.2f}")
            print(f"On edge: {damage['damage_on_edge']}")
            print("-" * 20)