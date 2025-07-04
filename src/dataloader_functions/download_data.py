import os
import openml
import requests

import sys
import os
# Add project root to sys.path (two levels up from this file)
current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
dataloader_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if dataloader_root not in sys.path:
    sys.path.insert(0, dataloader_root)

current_dir = os.path.dirname(os.path.realpath(__file__))
RAW_DIR = os.path.join(current_dir, '..', '..', 'datasets_files','raw')

from dataloader_functions.utils.log_msgs import *
from dataloader_functions.utils.data_2_df import run_command, _unzip_if_zipped


def _check_download_parameters(dataset_config):
    """
    Check if the download parameters are available.

    Looking for the following parameters:
    - source: source of the data (huggingface, kaggle, openml)
    - remote_path: path to the data on the source
    - task: task of the data (classification, regression)
    - target: target variable
    - files: files to download
    """

    dataset_name = dataset_config.get('dataset_name', 'unknown')

    if 'source' not in dataset_config.keys():
        error_msg(f"Dataset {dataset_name} does not have a source.")
        return False
    if 'remote_path' not in dataset_config.keys():
        error_msg(f"Dataset {dataset_name} does not have a remote path.")
        return False
    if 'task' not in dataset_config.keys():
        error_msg(f"Dataset {dataset_name} does not have a task.")
        return False
    if 'target' not in dataset_config.keys():
        error_msg(f"Dataset {dataset_name} does not have a target variable.")
        return False
    if 'files' not in dataset_config.keys():
        error_msg(f"Dataset {dataset_name} does not have files to download.")
        return False

    if dataset_config["task"] not in ['reg', 'clf']:
        error_msg(f"Dataset {dataset_name} does not have a valid task.")
        return False

    if dataset_config["source"] not in ['huggingface', 'kaggle', 'openml', 'url']:
        error_msg(f"Dataset {dataset_name} does not have a valid source.")
        return False

    return True

def _create_local_path_raw(local_path, task, dataset_name=None):
    """
    Craft the download/load path if not manually set.
    """
    if local_path is None:

        # use the defualt save path in this repo -> ../../data/raw/task
        local_path = RAW_DIR
        # we can craft our own download path:
        if task is None:
            error_msg("Task is not provided.")
            return None
        elif task == 'reg':
            local_path = os.path.join(local_path, 'regression')
        elif task == 'clf':
            local_path = os.path.join(local_path, 'classification')
        else:
            error_msg(f"Task '{task}' is not supported.")
            return None
        
        if dataset_name is not None:
            # if the dataset name is provided, create a folder for it
            local_path = os.path.join(local_path, dataset_name)
    return local_path

def _check_if_downloaded(download_path):
    """
    Check if the dataset is already downloaded.
    If the path exists and the folder is not empty, return True.
    """
    if os.path.exists(download_path) and os.path.isdir(download_path) and len(os.listdir(download_path)) > 0:
        info_msg(f"Dataset already downloaded in {color_text(download_path)}.")
        return True
    else:
        info_msg(f"Dataset not downloaded yet. Downloading to {color_text(download_path)}.")
        return False

def _rename_files(path, dataset_config):
    if 'rename_files' in dataset_config.keys():
        # check there are the same amount of files
        if not len(dataset_config['rename_files']) > 0 or len(dataset_config['rename_files'][0]) == 0:
            warn_msg(f"No files to rename. Skipping renaming.")
            return False

        if len(dataset_config['files']) != len(dataset_config['rename_files']):
            warn_msg(f"Number of files to rename does not match the number of files to download. Skipping renaming.")
            return False
        for file, new_file in zip(dataset_config['files'], dataset_config['rename_files']):
            file_path = os.path.join(path, file)
            new_file_path = os.path.join(path, new_file)
            if os.path.exists(file_path):
                os.rename(file_path, new_file_path)
                info_msg(f"Renamed {color_text(file)} to {color_text(new_file)}.")
            else:
                error_msg(f"File {color_text(file)} does not exist.")
                return False
    return True

def _clean_folder(path):
    """
    Clean the folder by removing all files in it.
    """
    if os.path.exists(path) and os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        info_msg(f"Cleaned the folder {color_text(path)}.")
    else:
        info_msg(f"Folder {color_text(path)} does not exist - can't be cleaned.")

def _remove_unlisted(dataset_config, download_path):
    """
    Remove files in the download_path that are not listed in dataset_config['files'].
    """
    listed_files = set(dataset_config['files'])

    if not os.path.exists(download_path) or not os.path.isdir(download_path):
        error_msg(f"Download path {color_text(download_path)} does not exist or is not a directory.")
        return False

    removed_any = False
    for file in os.listdir(download_path):
        file_path = os.path.join(download_path, file)
        if os.path.isfile(file_path) and file not in listed_files:
            os.remove(file_path)
            info_msg(f"Removed unlisted file {color_text(file)}.")
            removed_any = True

    if not removed_any:
        info_msg(f"No unlisted files found in {color_text(download_path)}.")

    return True
    
def _tsv_to_csv(path, dataset_config):
    """
    Convert all TSV files in the given path to CSV files.
    """

    rename_files = dataset_config.get('files', None)
    if rename_files is None or len(rename_files) == 0:
        raise ValueError("No files listed for processing.")

    for file in os.listdir(path):
        if file.endswith('.tsv'):
            tsv_file_path = os.path.join(path, file)
            csv_file_path = os.path.join(path, file.replace('.tsv', '.csv'))
            with open(tsv_file_path, 'r') as tsv_file:
                with open(csv_file_path, 'w') as csv_file:
                    for line in tsv_file:
                        csv_file.write(line.replace('\t', ','))
            os.remove(tsv_file_path)
            info_msg(f"Converted {color_text(file)} to CSV.")

    # replace the file name 'rename_files' with the new file name
    rename_files = [file.replace('.tsv', '.csv') for file in rename_files]
    return rename_files

def download_raw_data(dataset_config:dict, download_path:str, force_download:bool=False, remove_unlisted:bool=False):
    """
    Given the dataset name, attempt to download the data from the web.

    Supported sources are:
    - HuggingFace
    - Kaggle
    - OpenML
    - URL (using requests.get)
    """
    def hf_download(dataset_info, download_path):
        """
        Downloads a dataset from HuggingFace and saves it in the provided path.
        Given multuiple files, all will be downloaded.
        """
        hf_path = dataset_info['remote_path']
        files = dataset_info['files']
        base_url = "https://huggingface.co/datasets/" + hf_path + "/resolve/main/"

        if isinstance(files, str):
            files = [files]

        for file in files:
            url = base_url + file
            file_name = file.split("/")[-1]
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(download_path, file_name)
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                info_msg(f"Downloaded: {color_text(file_name)}"+
                         f" from {color_text('HuggingFace')} to {download_path}.", color='green')
            else:
                error_msg(f"Could not download the file '{color_text(file_name)}'.")
                return False
        
        return True

    def kaggle_download(dataset_info, download_path):
        """
        Downloads a dataset from Kaggle and saves it in the provided path.
        Given multuiple files, all will be downloaded.
        """

        if 'competition' in dataset_info.keys():
            print(f"competition: {dataset_info['competition']}")

        if 'competition' in dataset_info.keys() and dataset_info['competition']:
            # we need to download the whole competition
            command = f"kaggle competitions download -c {dataset_info['remote_path']} -p {download_path}"
            run_command(command)
            info_msg(f"Downloaded: {color_text(dataset_info['remote_path'])}"+
                        f" from {color_text('Kaggle')} to {download_path}.", color='green')
            return True

        kaggle_path = dataset_info['remote_path']
        files = dataset_info['files']    
        if isinstance(files, str):
            files = [files]

        for file in files:
            file_name = file.split("/")[-1]
            command = f"kaggle datasets download -d {kaggle_path} --file '{file_name}' -p {download_path}"
            try:
                run_command(command)
                info_msg(f"Downloaded: {color_text(file_name)}"+
                         f" from {color_text('Kaggle')} to {download_path}.", color='green')
            except:
                error_msg(f"Could not download the file '{color_text(file_name)}'.")
                return False
        return True

    def openml_download(dataset_info, download_path):
        """
        Downloads a dataset (single file) from OpenML and saves it as a CSV file.
        Saves the dataset as a CSV file in the given download path.
        """
        dataset_name = dataset_info['dataset_name']
        openml_id = dataset_info['remote_path']
        
        dataset = openml.datasets.get_dataset(openml_id,)
        if dataset is None:
            error_msg(f"Could not download the dataset '{color_text(dataset_name)}'"+
                      f" from {color_text('OpenML')} to {download_path}.")
            return False
        df, *_ = dataset.get_data()

        file_path = os.path.join(download_path, dataset_name + ".csv")
        df.to_csv(file_path, index=False)
        info_msg(f"Downloaded: {color_text(dataset_name)}"+
                 f" from {color_text('OpenML')}.", color='green')
        return True
    
    def url_download(dataset_info, download_path):
        """
        Downloads a !single file! from the given URL.
        Experimental use only, feel free to modify for specific url downloads.
        """
        
        url = dataset_info['remote_path']
        file_name = os.path.basename(url)
        file_path = os.path.join(download_path, file_name)

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            info_msg(f"Downloaded: {color_text(file_name)}"+
                    f" from {color_text('url')} to {download_path}.", color='green')
            return True
        else:
            error_msg(f"Could not download the file '{color_text(file_name)}'"+
                    f" via {color_text('url-download')}.")
            return False

    download_func_dict = {
    'huggingface': hf_download,
    'kaggle': kaggle_download,
    'openml': openml_download,
    'url': url_download
    }

    # check if the dataset is already downloaded
    if not force_download and _check_if_downloaded(download_path):
        return download_path
    elif force_download:
        _clean_folder(download_path)

    # chec if the dataset is supported and has all parameters
    if not _check_download_parameters(dataset_config):
        error_msg(f"Dataset {dataset_config['dataset_name']} is not supported or its configs are not complete.")
        return False
    
    task = dataset_config['task']
    dataset_source = dataset_config['source']

    # in case the download path is not provided, create a local path
    download_path = _create_local_path_raw(download_path, task, dataset_config['dataset_name'])

    # download the dataset according to the source
    download_fc = download_func_dict.get(dataset_source)
    download_fc(dataset_config, download_path)

    # if the dataset has not been downloaded, return False
    if not os.path.exists(download_path) or len(os.listdir(download_path)) == 0:
        error_msg(f"Dataset {dataset_config['dataset_name']} could not be downloaded.")
        return False

    _unzip_if_zipped(download_path)

    if remove_unlisted:
        _remove_unlisted(dataset_config, download_path)


    dataset_config['files'] =_tsv_to_csv(download_path, dataset_config)

    if not _rename_files(download_path, dataset_config):
        # error_msg(f"Could not rename the files in {color_text(download_path)}.")
        ...    

    return download_path


if __name__ == "__main__":
    ## testing download functions
    dataset_config = {
        'dataset_name': 'laptops',
        'source': 'kaggle',
        'remote_path': 'dhanushbommavaram/laptop-dataset',
        'files': ['complete laptop data0.csv'],
        'rename_files': ['laptops.csv'],
        'task': 'reg',
        'target': 'Price',
    }

    download_path = os.path.join(RAW_DIR, 'regression', dataset_config['dataset_name'])
    download_raw_data(dataset_config, download_path, force_download=True)