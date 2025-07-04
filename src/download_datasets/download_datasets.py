from nbclient import NotebookClient
from nbformat import read, write
import os, sys
from configs.dataset_configs import get_dataset_list, get_a_dataset_dict
from src.dataloader_functions.utils.log_msgs import info_msg, warn_msg, error_msg
project_root = os.environ["PROJECT_ROOT"]

def build_ntbk_path(dataset_name: str) -> str:
    """
    Build the path to the Jupyter notebook for a given dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
    
    Returns:
        str: Path to the Jupyter notebook.
    """
    datasets_dir = os.path.join(project_root, 'datasets_notebooks')
    dataset_config = get_a_dataset_dict(dataset_name)
    
    if dataset_config['task'] == 'clf':
        task_folder = 'classification'
    elif dataset_config['task'] == 'reg':
        task_folder = 'regression'
    else:
        raise ValueError(f"Unknown task type '{dataset_config['task']}' for dataset '{dataset_name}'.")

    if not dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configurations.")
    
    ntbk_name = dataset_config['ntbk']
    ntbk_path = os.path.join(datasets_dir, task_folder, ntbk_name)
    
    return ntbk_path

def run_notebook(notebook_path, output_path=None):
    """
    Runs a Jupyter notebook and saves the output if specified.
    """
    # Change to notebook's directory
    notebook_dir = os.path.dirname(notebook_path)
    os.chdir(notebook_dir)

    # Load the notebook
    with open(notebook_path) as f:
        nb = read(f, as_version=4)

    # Execute notebook
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    # Save executed notebook if output_path is given
    if output_path:
        with open(output_path, 'w') as f:
            write(nb, f)

def download_datasets(datasets_selection: str) -> bool:
    """
    Run downlaod and default processing of selected datasets via jupiter notebooks.

    Input:
        datasets_selection (str): 'all', 'clf', 'reg', or a specific dataset name.  

    Returns:
        bool: True if successful, False otherwise.
    """

    dataset_name_list = get_dataset_list(datasets_selection)

    print(f"Downloading datasets: {dataset_name_list}")

    for dataset_name in dataset_name_list:
        ntbk_path = build_ntbk_path(dataset_name)

        # Check if the notebook exists
        if not os.path.exists(ntbk_path):
            print(f"Notebook for dataset '{dataset_name}' not found at '{ntbk_path}'.")
            continue

        # Run the notebook
        try:
            info_msg(f"Running notebook for dataset '{dataset_name}' at '{ntbk_path}'...")
            run_notebook(ntbk_path)
            info_msg(f"Successfully ran notebook for dataset '{dataset_name}'.", color='green')
        except Exception as e:
            error_msg(f"Error running notebook for dataset '{dataset_name}': {e}")
            return False


if __name__ == "__main__":
    
    download_selection = 'hs_cards'  # or 'clf', 'reg', <specific_dataset_name>

    # Block = 1 =
    # First Make sure the data is downlaoded and "standardly preprocessed"
    
    download_datasets(download_selection)
