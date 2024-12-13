import os
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from ast import literal_eval


def aggregate_by_damage_type(line_level_df):
    agg_dict = {
        'damage_size': 'sum',
        'damage_on_edge': 'any',  # True if any instance is True
        'whole_panel_visible': 'any',  # True if any instance is True
    }
    
    # Group by image_id, panel, and damage_type and aggregate
    aggregated = line_level_df.groupby(['image_id', 'panel', 'damage_type']).agg(agg_dict)
        
    # Add count column showing how many rows were merged
    count = line_level_df.groupby(['image_id', 'panel', 'damage_type']).size()
    aggregated['count'] = count
        
    # Reset index to convert groupby columns back to regular columns
    aggregated = aggregated.reset_index()
    return aggregated


def calculate_visible_metrics(group):
        visible_data = group[group['whole_panel_visible'] == True]['damage_size']
        if len(visible_data) > 0:
            return pd.Series({
                'max_damage_size_visible': visible_data.max(),
                'average_damage_size_visible': visible_data.mean()
            })
        return pd.Series({
            'max_damage_size_visible': np.nan,
            'average_damage_size_visible': np.nan
        })


def aggregate_by_imbag(df):
    image_id_lists = df.groupby(['imbag_id', 'panel'])['image_id'].agg(list).reset_index()
    
    # Now perform the main aggregation
    aggregated = df.groupby(['imbag_id', 'panel', 'damage_type']).agg({
            'damage_size': ['max', 'mean'],          # Get max and average damage size
            'damage_on_edge': 'any',                 # True if any are True
            'whole_panel_visible': 'any',            # True if any are True
            'count': 'max'                           # Keep max count
        })
        
    # Flatten column names
    aggregated.columns = ['max_damage_size', 'average_damage_size', 
                            'damage_on_edge', 'whole_panel_visible', 'count']
    aggregated = aggregated.reset_index()

    visible_metrics = df.groupby(['imbag_id', 'panel', 'damage_type']).apply(calculate_visible_metrics)
    visible_metrics = visible_metrics.reset_index()
        
    # Merge all results together
    result = pd.merge(aggregated, visible_metrics, 
                        on=['imbag_id', 'panel', 'damage_type'], 
                        how='left')
        
    # Add the image_id lists
    result = pd.merge(result, image_id_lists, 
                        on=['imbag_id', 'panel'], 
                        how='left')
    return result