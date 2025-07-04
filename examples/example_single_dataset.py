import os, sys
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# some imports for xgboost pp
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.download_datasets.download_datasets import download_datasets # ups, sorry for the bad naming :)
from configs.dataset_configs import get_a_dataset_dict

def train_eval_xgboost(loaded_df, loaded_config, task='clf'):
    target_col = loaded_config['target']

    X = loaded_df.drop(columns=[target_col])
    y = loaded_df[target_col]

    # Encode target labels if classification
    if task == 'clf' and y.dtype == object:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Separate numeric and categorical
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()

    # Preprocessing
    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Choose model
    if task == 'clf':
        model = XGBClassifier(eval_metric='mlogloss')
    elif task == 'reg':
        model = XGBRegressor()
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Create pipeline and fit
    pipeline = make_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"Model trained. Test score: {score:.4f}")


if __name__ == "__main__":
    # Example usage of the download_datasets function
    datasets_selection = 'hs_cards'  # or specify a list of dataset names
    task = 'clf'  # or specify a specific task
    download = False

    if download:
        success = download_datasets(datasets_selection=datasets_selection, task=task)
        
        if success:
            print("Datasets downloaded successfully.")
        else:
            print("Failed to download datasets.")
    
    # load the data again
    dataset_config = get_a_dataset_dict(datasets_selection)

    # find the file location
    if dataset_config['task'] == 'clf':
        dataset_subfolder = os.path.join('raw', 'classification', dataset_config['dataset_name']) 
    elif dataset_config['task'] == 'reg':
        dataset_subfolder = os.path.join('raw', 'regression', dataset_config['dataset_name'])
    else:
        raise ValueError(f"Unknown task: {dataset_config['task']}")

    # build the path to the processed file
    download_path = os.path.join(current_dir, '..', 'datasets_notebooks', 'datasets_files', dataset_subfolder)

    file_base = dataset_config['dataset_name']
    processed_filename = f"{file_base}_processed.pkl"
    processed_path = os.path.join(download_path, processed_filename)

    # Load the bundled dictionary (data + summary + config)
    bundle = pd.read_pickle(processed_path)

    # Extract components
    loaded_df = bundle['data']
    summary_df = bundle['summary']
    loaded_config = bundle['config']

    print(f"\n=== {file_base.upper()} ===")

    meta_df = pd.DataFrame(loaded_df.columns.tolist(), columns=['Column Name', 'Type', '# Categories'])
    print(f"Meta information:\n{meta_df}\n")

    # Flatten for modeling
    loaded_df.columns = loaded_df.columns.get_level_values(0)
    print(f"loaded_df shape: {loaded_df.shape}")

    # drop the text for now from meta_df
    non_text_cols = meta_df[meta_df['Type'] != 'textual']['Column Name'].tolist()
    loaded_df = loaded_df[non_text_cols]
    print(f"loaded_df shape after dropping text columns: {loaded_df.shape}")

    # split the data into features and target

    # Train a simple XGBoost model
    train_eval_xgboost(loaded_df, loaded_config)
