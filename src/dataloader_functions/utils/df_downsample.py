import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.log_msgs import *

import pandas as pd


def _downsample_uniform(data_df, target_col, output_rows, seed=0):
    """
    Ensures at least 2 instances per class is included, then performs uniform downsampling.

    No special treatment for regression tasks.
    """
    n_rows = len(data_df)
    
    if n_rows <= output_rows:
        warn_msg(f"Data already has fewer rows than the specified output rows."+
                 f" Returning None.", color='red')
        return None

    # Step 1: Ensure at least one sample per class
    sampled_df = data_df.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(n=2, random_state=seed)
    )

    if len(sampled_df) > output_rows:
        warn_msg("The data has fewer 'classes' than the specified output rows. If the task is classification, this number of rows is infesable!.", color='red')
        return sampled_df.sample(n=output_rows, random_state=seed).reset_index(drop=True)

    # Step 2: Remove these sampled rows from the dataset
    remaining_data = data_df.drop(sampled_df.index)

    # Step 3: Uniformly sample from remaining data to reach `output_rows`
    remaining_needed = output_rows - len(sampled_df)

    if remaining_needed > 0 and not remaining_data.empty:
        additional_samples = remaining_data.sample(
            min(remaining_needed, len(remaining_data)), 
            random_state=seed, 
            replace=False
        )
        sampled_df = pd.concat([sampled_df, additional_samples])

    # Step 4: Ensure exactly `output_rows`
    return sampled_df.sample(n=output_rows, random_state=seed).reset_index(drop=True)

def _downsample_stratified(data_df: pd.DataFrame, target, output_rows, task="clf", seed=0):
    """
    Stratified (keeping the ratios) downsampling for both classification and regression.
    
    - If `task='clf'`, ensures at least 2 samples per class.
    - If `task='reg'`, dynamically determines `q` based on dataset size.
    
    Parameters:
    - data_df (pd.DataFrame): Input dataset.
    - target (str): Name of the target column.
    - output_rows (int): Number of rows after downsampling.
    - task (str): 'clf' for classification, 'reg' for regression.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - pd.DataFrame: Downsampled dataset with `output_rows` rows.
    """
    n_rows = len(data_df)

    if n_rows <= output_rows:
        warn_msg(f"Data already has fewer rows than the specified output rows."+
                 f" Returning None.", color='red')
        return None

    if task == "clf":
        # Count occurrences of each class
        class_counts = data_df[target].value_counts()

        # Ensure every class has at least 2 instances
        if (class_counts < 2).any():
            raise ValueError(f"All classes must have at least 2 instances. Found:\n{class_counts[class_counts < 2]}")

        # Step 1: Ensure at least 2 samples per class
        mandatory_samples = data_df.groupby(target, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), 2), random_state=seed)
        )

        remaining_output_rows = output_rows - len(mandatory_samples)

        # Step 2: Compute original class distribution (excluding mandatory samples)
        class_counts_normalized = data_df[target].value_counts(normalize=True)

        # Calculate the number of additional samples per class based on distribution
        samples_per_class = (class_counts_normalized * remaining_output_rows).astype(int)

        # Ensure at least 1 additional sample per class (if possible)
        samples_per_class = samples_per_class.clip(lower=1)

        # Step 3: Perform stratified sampling (excluding already taken mandatory samples)
        remaining_data = data_df.drop(mandatory_samples.index)

        sampled_df = remaining_data.groupby(target, group_keys=False).apply(
            lambda x: x.sample(min(len(x), samples_per_class[x.name]), random_state=seed)
        )

        # Step 4: Combine both samples
        sampled_df = pd.concat([mandatory_samples, sampled_df])

    elif task == "reg":
        # Dynamic q calculation based on dataset size
        if n_rows <= 5000:
            q = 5
        elif n_rows <= 50_000:
            q = int(5 + (n_rows - 5000) / (50_000 - 5000) * (10 - 5))  # Scale q from 5 → 10
        else:
            q = int(10 + (n_rows - 50_000) / (100_000 - 50_000) * (20 - 10))  # Scale q from 10 → 20

        q = min(max(q, 5), 20)  # Ensure q is between 5 and 20
        info_msg(f"Using q={q} for regression stratification")

        # Step 1: Compute value distribution for regression
        bins = pd.qcut(data_df[target], q=q, duplicates="drop")  # Divide into quantile bins
        bin_counts = bins.value_counts(normalize=True)

        # Step 2: Compute samples per bin
        samples_per_bin = (bin_counts * output_rows).astype(int)

        # Step 3: Perform stratified sampling within each bin
        sampled_df = data_df.groupby(bins, group_keys=False).apply(
            lambda x: x.sample(min(len(x), samples_per_bin[x.name]), random_state=seed)
        )

    else:
        raise ValueError("Invalid task type. Use 'clf' for classification or 'reg' for regression.")

    # Step 5: Fill up any missing rows randomly to match `output_rows`
    remaining_needed = output_rows - len(sampled_df)

    if remaining_needed > 0 and not data_df.empty:
        additional_samples = data_df.sample(
            min(remaining_needed, len(data_df)), random_state=seed, replace=False
        )
        sampled_df = pd.concat([sampled_df, additional_samples])

    # Ensure exactly `output_rows`
    return sampled_df.sample(n=output_rows, random_state=seed).reset_index(drop=True)

def _balanced_downsample_simple(data_df: pd.DataFrame, target, output_rows, seed=0):
    """
    Ensures all classes appear at least as many times as the rarest class,
    then fills up remaining slots with uniform sampling.
    """
    n_rows = len(data_df)
    
    if n_rows <= output_rows:
        warn_msg(f"Data already has fewer rows than the specified output rows."+
                 f" Returning None.", color='red')
        return None

    # Count occurrences of each class
    class_counts = data_df[target].value_counts()
    
    # Find the minimum class frequency
    min_class_size = class_counts.min()

    # Ensure that Step 1 does not exceed output_rows
    num_classes = len(class_counts)
    max_per_class = output_rows // num_classes  # Avoid exceeding total rows
    min_class_size = min(min_class_size, max_per_class)

    # Step 1: Take at least min_class_size samples from every class
    sampled_df = data_df.groupby(target, group_keys=False).apply(
        lambda x: x.sample(min(len(x), min_class_size), random_state=seed)
    )

    info_msg(f"Balance sampled {color_text(len(sampled_df), 'yellow')} rows -> rest is uniform sampled", color='green')

    # Step 2: Remove already sampled rows from the dataset
    remaining_data = data_df.drop(sampled_df.index)

    # Step 3: Uniformly sample from the remaining data until reaching output_rows
    remaining_needed = output_rows - len(sampled_df)

    if remaining_needed > 0 and not remaining_data.empty:
        additional_samples = remaining_data.sample(
            min(remaining_needed, len(remaining_data)), 
            random_state=seed, 
            replace=False
        )
        sampled_df = pd.concat([sampled_df, additional_samples])

    # Return exactly output_rows
    return sampled_df.sample(n=min(len(sampled_df), output_rows), random_state=seed).reset_index(drop=True)

def df_downsample(strat:str, data_df:pd.DataFrame, task:str, target:str, out_rows:int, seed=0):
    """
    Downsample a DataFrame based on the specified strategy.
    Parameters:
    - strat (str): Downsampling strategy ('uniform', 'stratified', 'balanced').
    - data_df (pd.DataFrame): Input DataFrame to downsample.
    - task (str): Task type ('clf' for classification, 'reg' for regression).
    - target (str): Name of the target column.
    - out_rows (int): Desired number of rows after downsampling.
    - seed (int): Random seed for reproducibility.
    Returns:
    - pd.DataFrame: Downsampled DataFrame.
    Raises:
    - ValueError: If the specified strategy is not recognized.
    """

    if strat == 'uniform':
        data_df = _downsample_uniform(data_df, output_rows=out_rows, seed=seed)
    elif strat == 'stratified':
        data_df = _downsample_stratified(data_df, target, task=task, output_rows=out_rows, seed=seed)
    elif strat == 'balanced':
        data_df = _balanced_downsample_simple(data_df, target, output_rows=out_rows, seed=seed)
    else:
        raise ValueError(f"Unknown downsampling strategy: {strat}")

    return data_df