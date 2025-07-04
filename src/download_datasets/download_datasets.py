from nbclient import NotebookClient
from nbformat import read, write
import os, sys
import nbformat.v4 as v4

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.dataset_configs import get_dataset_list, get_a_dataset_dict
from src.dataloader_functions.utils.log_msgs import info_msg, warn_msg, error_msg

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
    
    if not dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configurations.")
    
    ntbk_name = dataset_config['ntbk']

    # Recursively walk through the datasets_dir to find the notebook
    for root, _, files in os.walk(datasets_dir):
        for file in files:
            if file == ntbk_name:
                return os.path.join(root, file)
    
    raise FileNotFoundError(f"Notebook '{ntbk_name}' not found in '{datasets_dir}'.")

def run_notebook(notebook_path, ntbk_params=None, output_path=None):
    """
    Runs a Jupyter notebook and saves the output if specified.
    """
    # Change to notebook's directory
    notebook_dir = os.path.dirname(notebook_path)
    os.chdir(notebook_dir)

    # Load the notebook
    with open(notebook_path) as f:
        nb = read(f, as_version=4)

    # Inject parameter cell if provided
    if ntbk_params:
        injected_code = '\n'.join(f'{key} = {repr(value)}' for key, value in ntbk_params.items())
        param_cell = v4.new_code_cell(injected_code)
        nb.cells.insert(0, param_cell)

    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    if output_path:
        with open(output_path, 'w') as f:
            write(nb, f)

def download_datasets(datasets_selection:str | list='default', task: str='all', path: str=None) -> bool:
    """
    Run downlaod and default processing of selected datasets via jupiter notebooks.

    Input:
        datasets_selection (list or str): List of dataset names or selection criteria.
        - allowed list items for bulk selection: 'default'(default), 'extra', 'other'
        - also accepts specific dataset names as strings, e.g. 'hs_cards'
        task (str): Task type, either 'clf' for classification or 'reg' for regression.
        - options: 'clf', 'reg', 'all'(default)

    Returns:
        bool: True if successful, False otherwise.

    Example:
        download_datasets(['default', 'okcupid'], task='clf')
    """

    dataset_name_list = get_dataset_list(datasets_selection, task=task)

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
            run_notebook(ntbk_path, ntbk_params={'data_path': path})
            info_msg(f"Successfully ran notebook for dataset '{dataset_name}'.", color='green')
        except Exception as e:
            error_msg(f"Error running notebook for dataset '{dataset_name}': {e}")
            return False
    return True

if __name__ == "__main__":
    
    download_selection = 'hs_cards'  # or 'clf', 'reg', <specific_dataset_name>

    download_datasets(download_selection, task='all', path=None)
