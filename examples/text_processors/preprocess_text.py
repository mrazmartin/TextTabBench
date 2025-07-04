import fasttext
import numpy as np 
import os, sys
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import pickle
import pandas as pd
from skrub import TableVectorizer
from src.dataloader_functions.utils.log_msgs import info_msg, warn_msg, error_msg, success_msg
from huggingface_hub import hf_hub_download

project_root = os.environ["PROJECT_ROOT"]

def generate_text_embeddings(df, meta_df, emb_path='.', 
                            methods=('fasttext', 'ag', 'skrub'),
                            save_format='npy'):
    """
    Generate text embeddings only (no full-data processing).
    """
    text_columns = meta_df.loc[meta_df['Type'] == 'textual', 'Column Name'].tolist()
    if not text_columns:
        error_msg("No text columns found in metadata")
        raise ValueError("No text columns found in metadata")
    
    for col in text_columns:
        df[col] = df[col].fillna('').astype(str)

    os.makedirs(emb_path, exist_ok=True)
    embeddings = {}
    info_msg(f"Generating embeddings for text columns: {', '.join(text_columns)}")

    if 'fasttext' in methods:
        info_msg("Initializing FastText embeddings...")
        model_path = os.path.join(project_root, 'embedding_models', 'fasttext_model', 'cc.en.300.bin')

        if not os.path.exists(model_path):
            info_msg(f"Downloading FastText model to {model_path}")
            try:
                model_path = hf_hub_download(repo_id="hi-paris/fastText", filename="cc.en.300.bin", local_dir=model_path)
                info_msg("FastText model download completed", color="green")
            except Exception as e:
                error_msg(f"Model download failed: {str(e)}")
                raise
        else:
            info_msg(f"Found existing model at {model_path}")
        
        try:
            model = fasttext.load_model(model_path)
            info_msg("FastText model loaded successfully", color="green")
        except Exception as e:
            error_msg(f"Model loading failed: {str(e)}")
            raise
        
        def embed_text(sentence):
            # Clean text: remove newlines, extra spaces, and ensure proper string format
            text = str(sentence).strip().replace('\n', ' ').replace('\r', ' ')
            text = ' '.join(text.split())  # Remove extra whitespace
            if not text:
                return np.zeros(model.get_dimension())
            try:
                return model.get_sentence_vector(text)
            except Exception as e:
                error_msg(f"Failed to embed text: '{text[:50]}...' (Error: {str(e)})")
                return np.zeros(model.get_dimension())
        
        info_msg(f"Generating Fasttext embeddings...")
        try:
            ft_embs = []
            for col in text_columns:
                ft_embs.append(np.vstack(df[col].apply(embed_text)))
            embeddings['fasttext'] = np.hstack(ft_embs)
            info_msg(f"FastText embeddings created successfully", color="green")
        except Exception as e:
            error_msg(f"Failed to create fasttext embeddings: {str(e)}")
            raise

    if 'ag' in methods:
        info_msg("Generating AG text embeddings...")
        try:
            feature_generator = AutoMLPipelineFeatureGenerator()
            embeddings['ag'] = feature_generator.fit_transform(X=df[text_columns])
            info_msg("AG embeddings created successfully",color="green")
        except Exception as e:
            error_msg(f"Failed to create AG embeddings: {str(e)}")
            raise

    if 'skrub' in methods:
        info_msg("Generating skrub text embeddings...")
        try:
            vectorizer = TableVectorizer()
            embeddings['skrub'] = vectorizer.fit_transform(df[text_columns])
            info_msg("skrub embeddings created successfully",color="green")
        except Exception as e:
            error_msg(f"Failed to create skrub embeddings: {str(e)}")
            raise

    def save_embedding(data, prefix):
        path = os.path.join(emb_path, f"{prefix}_text_embeddings")
        
        try:

            if save_format == 'npy':
                if isinstance(data, pd.DataFrame):
                    data = data.fillna(0.0).values.astype(np.float32) 
                    np.save(f"{path}.npy", data)  
                else: 
                    data = np.nan_to_num(data.astype(np.float32))  # Ensure clean float32
                    np.save(f"{path}.npy", data)
            else:  # pkl
                with open(f"{path}.pkl", 'wb') as f:
                    pickle.dump(data, f)
            info_msg(f"Saved {prefix} embeddings to {path}.{save_format}",color="green")
        except Exception as e:
            error_msg(f"Failed to save {prefix} embeddings: {str(e)}")
            raise
    
    for name, emb in embeddings.items():
        # Check for NaNs
        if isinstance(emb, np.ndarray):
            if np.isnan(emb).any():
                print(f"🚨 {name} embeddings have NaNs! Shape: {emb.shape}")
        elif isinstance(emb, pd.DataFrame):
            if emb.isna().any().any():
                print(f"🚨 {name} (DataFrame) has NaNs!")
        
        save_embedding(emb, name)  # Proceed to save

    for name, emb in embeddings.items():
        save_embedding(emb, name)
    
    return embeddings

def load_text_embeddings(emb_path='.', methods=None):
    """
    Load text embeddings from directory.
    """
    available_methods = ['fasttext', 'ag', 'skrub']
    if methods is None:
        methods = available_methods
    
    embeddings = {}
    info_msg(f"Loading embeddings from {emb_path}")
    
    for method in methods:
        for ext in ['npy', 'pkl']:
            path = os.path.join(emb_path, f"{method}_text_embeddings.{ext}")
            if os.path.exists(path):
                try:
                    if ext == 'npy':
                        data = np.load(path, allow_pickle=True)
                    else:
                        with open(path, 'rb') as f:
                            data = pickle.load(f)
                    embeddings[method] = data
                    info_msg(f"Loaded {method} embeddings from {path}",color="green")
                    break
                except Exception as e:
                    warn_msg(f"Failed to load {method} embeddings from {path}: {str(e)}")
    return embeddings

def embeddings_to_df(embeddings, original_df=None, meta_df=None):
    """
    Convert text embeddings dictionary to pandas DataFrame(s) with non-text original columns.
    """
    result = {}
    info_msg("Converting embeddings to DataFrames...")
    
    def _matrix_to_df(matrix, prefix):
        if isinstance(matrix, pd.DataFrame):
            if any(not col.startswith(prefix) for col in matrix.columns):
                return matrix
            return matrix.add_prefix(f"{prefix}_")
        return pd.DataFrame(
            matrix,
            columns=[f"{prefix}_{i}" for i in range(matrix.shape[1])]
        )
    
    for emb_type in ['fasttext', 'ag', 'skrub']:
        if emb_type in embeddings:
            prefix = 'ft' if emb_type == 'fasttext' else emb_type
            result[f"{emb_type}_df"] = _matrix_to_df(embeddings[emb_type], prefix)
            info_msg(f"Created {emb_type}_df with shape {result[f'{emb_type}_df'].shape}")
    
    if original_df is not None and meta_df is not None:
        try:
            text_columns = meta_df.loc[meta_df['Type'] == 'textual', 'Column Name'].tolist()
            non_text_cols = [col for col in original_df.columns if col not in text_columns]
            original_data = original_df[non_text_cols].copy().reset_index(drop=True)
            
            for key in list(result.keys()):
                result[key] = pd.concat([original_data, result[key]], axis=1)
                info_msg(f"Merged original non-text columns with {key}")
        except Exception as e:
            error_msg(f"Failed to merge original columns: {str(e)}")
            raise
    
    info_msg("Embeddings converted to DataFrames successfully",color="green")
    return result


if __name__ == "__main__":
    df_path = "/work/dlclarge2/dasb-Camvid/tabadap/datasets_files/raw/regression/beer/beer_processed.pkl"
    save_path = "/work/dlclarge2/dasb-Camvid/tabadap/datasets_files/embeddings/regression/beer"
    bundle = pd.read_pickle(df_path)

    # 1. extract components
    loaded_df = bundle['data']
    loaded_df.columns = loaded_df.columns.get_level_values(0)
    summary_df = bundle['summary']
    loaded_config = bundle['config']

    # 2. embedd and save the data
    embeddings = generate_text_embeddings(
        df=loaded_df, 
        meta_df=summary_df, 
        emb_path=save_path,
        methods = 'ag', 
        save_format='npy'
    )

    # 3. load embeddings
    embeddings = load_text_embeddings(emb_path=save_path)

    # 4. convert to df
    dfs = embeddings_to_df(embeddings, original_df=loaded_df, meta_df=summary_df)
    nan_rows = dfs['ag_df'][dfs['ag_df'].isna().any(axis=1)]
    print("Rows with NaN values in AG embeddings:")
    print(nan_rows)
    # 5. aesthetic printing ✨
    for df_name, df in dfs.items():
        print(f"======{df_name}======")
        print(df.shape)
        print(df.head(3))

        
