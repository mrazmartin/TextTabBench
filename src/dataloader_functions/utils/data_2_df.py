import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.log_msgs import *
import subprocess
import urllib.parse

import chardet
import csv
import arff
import pandas as pd

#####
### utils for other for folder management functions:
#####
def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        error_msg(f"Command failed with error:\n{e.stderr.strip()}")
        raise 

#####
## folder management functions
#####

def setup_folder(folder_path):
    """
    Create a folder if it does not exist.
    """
    os.makedirs(folder_path, exist_ok=True)

def check_if_folder_empty(folder_path):
    """
    Check if the folder is empty.
    """
    is_empty = len(os.listdir(folder_path)) == 0
    return is_empty

def _unzip_if_zipped(download_path):
    files_in_dir = os.listdir(download_path)
    for file in files_in_dir:
        if file.endswith('.zip'):
            # first check if the unzipped file is already present
            if file[:-4] in files_in_dir:
                info_msg(f"File {color_text(file[:-4])} already exists. Replacing it.")
                os.remove(os.path.join(download_path, file[:-4]))
                
            unzip_file(os.path.join(download_path, file), download_path)
            # remove the zip file
            os.remove(os.path.join(download_path, file))

def unzip_file(file_path, download_path):
    command = f"unzip {file_path} -d {download_path}"
    run_command(command)
    info_msg(f"Unzipped the file '{color_text(file_path)}'.")


#####
## handle yaml files
#####
def _read_yaml_as_df(file_path):
    """
    Read the yaml file and return the contents as a pandas DataFrame.
    """
    data_dict = read_yaml(file_path)
    if data_dict is None:
        raise ValueError("Error reading the yaml file.")

    data_df = pd.json_normalize(data_dict, sep='.')
    return data_df

def read_yaml(file_path):
    """
    Read the yaml file and return the contents.
    """
    import yaml

    with open(file_path, 'r') as file:
        try:
            data_dict = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            error_msg(f"Error reading the yaml file: {exc}")
    return data_dict

#####
## handle csv data files
#####
def _read_csv_as_df(file_path):
    encoding = detect_encoding(file_path)
    try:
        delimiter = sniff_delimeter(file_path, encoding)
        if delimiter is None:
            delimiter = ','
        data_df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
    except Exception as e:
        error_msg(f"Error reading the csv file: {e}")
        return None
    
    return data_df

def sniff_delimeter(file_path: str, encoding: str):
    """
    Sniff the delimiter of the CSV file.

    Args:
    file_path: str, path to the CSV file
    encoding: str, encoding of the file

    Returns:
    delimiter: str, detected delimiter of the CSV file
    """
    with open(file_path, 'r', encoding=encoding) as file:
        dialect = csv.Sniffer().sniff(file.readline())
        delimiter = dialect.delimiter

    if delimiter not in ",;\t":
        warn_msg(f"Delimiter {delimiter} is not supported. Using ',' as default.")
        return ','

    return delimiter

def detect_encoding(file_path: str):
    """
    Detect the encoding of a file.

    Args:
    file_path: str, path to the file
    
    Returns:
    encoding: str, detected encoding of the file
    """
    with open(file_path, "rb") as file:
        result = chardet.detect(file.read())
        return result['encoding'] 

#####
## handle arff files
#####
def _read_arff_to_dataframe(file_path):
    # Load the ARFF file
    with open(file_path, 'r') as file:
        dataset = arff.load(file)

    # Extract data and attributes
    data = dataset['data']
    attributes = dataset['attributes']

    # Get column names from attributes
    columns = [attr[0] for attr in attributes]

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # String attributes are read as objects; no need for decoding
    return df

#####
## handle supported type files:
#####
def read_any_to_df(file_path):
    """
    Read any file to a pandas DataFrame.
    Support for: .csv, .arff, .yaml, .pkl, .xlxs, .xls

    Args:
    file_path: str, path to the file

    Returns:
    data_df: pd.DataFrame, data in the file as a DataFrame
    """

    if not os.path.exists(file_path):
        error_msg(f"File {color_text(file_path)} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")

    if file_path.endswith('.csv'):
        return _read_csv_as_df(file_path)
    elif file_path.endswith('.arff'):
        return _read_arff_to_dataframe(file_path)
    elif file_path.endswith('.yaml'):
        return _read_yaml_as_df(file_path)
    elif file_path.endswith('pkl'):
        return pd.read_pickle(file_path)
    elif file_path.endswith('.xlxs') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        error_msg(f"File format {file_path.split('.')[-1]} is not supported.")
        raise ValueError("File format not supported.")
