
import numpy as np
def weighted_loss(y_true, y_pred, class_weights=None):
    if class_weights is None:
        num_classes = y_pred.shape[1]
        class_weights = np.ones(num_classes)
        
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Convert y_true to int type and flatten if needed
    y_true = y_true.astype(int).to_numpy() if hasattr(y_true, "to_numpy") else np.array(y_true).astype(int)
    
    true_class_probs = y_pred[np.arange(len(y_true)), y_true]
    losses = -np.log(true_class_probs)
    weights = class_weights[y_true]
    weighted_losses = weights * losses
    
    return np.mean(weighted_losses)

data_configs = {
    # CLASSIFICATION DATASETS
    'customer_complaints': {
        'dataset_name': 'customer_complaints',
        'short_name': 'complaints',
        'source': 'kaggle',
        'remote_path': 'selener/consumer-complaint-database',
        'files': ['rows.csv'],
        'rename_files': ['customer_complaints.csv'],
        'task': 'clf',
        'target': 'Company response to consumer',

        'ntbk': 'complaints_data.ipynb',
        },
    'diabetes': {
        'dataset_name': 'diabetes',
        'short_name': 'diabetes',
        'source': 'kaggle',
        'remote_path': 'ziya07/diabetes-clinical-dataset100k-rows',
        'files': ['diabetes_dataset_with_notes.csv'],
        'rename_files': ['diabetes.csv'],
        'task': 'clf',
        'target': 'diabetes',

        'task': 'clf',
        'ntbk': 'diabetes_data.ipynb',
        },
    'job_frauds': {
        'dataset_name': 'job_frauds',
        'short_name': 'frauds',
        'source': 'kaggle',
        'remote_path': 'shivamb/real-or-fake-fake-jobposting-prediction',
        'files': ['fake_job_postings.csv'],
        'rename_files': ['fake_job_postings.csv'],
        'task': 'clf',
        'target': 'fraudulent', 

        'ntbk': 'fraud_detec_data.ipynb',
        },
    'hs_cards': {
        'dataset_name': 'hs_cards',
        'short_name': 'hs_cards',
        'source': 'kaggle',
        'remote_path': 'jeradrose/hearthstone-cards',
        'files': ['cards_flat.csv'],
        'rename_files': ['hs_cards.csv'],
        'task': 'clf',
        'target': 'player_class',

        'ntbk': 'HS_cards_data.ipynb',
        },
    'kickstarter': {
        'dataset_name': 'kickstarter',
        'short_name': 'kick',
        'source': 'kaggle',
        'remote_path': 'yashkantharia/kickstarter-campaigns',
        'files': ['Kickstarter_projects_Feb19.csv'],
        'rename_files': ['kickstarter_data.csv'],
        'target': 'status',
        'task': 'clf',
    
        'ntbk': 'kickstarter_data.ipynb',
        },
    'lending_club': {
        'dataset_name': 'lending_club',
        'short_name': 'lending',
        'source': 'kaggle',
        'remote_path': 'imsparsh/lending-club-loan-dataset-2007-2011',
        'files': ['loan.csv'],
        'rename_files': ['ledning_club.csv'],
        'task': 'clf',
        'target': 'loan_status',

        'ntbk': 'lending_club_data.ipynb',
        },
    'okcupid': {
        'dataset_name': 'okcupid',
        'short_name': 'cupid',
        'source': 'kaggle',
        'remote_path': 'andrewmvd/okcupid-profiles',
        'files': ['okcupid_profiles.csv'],
        'rename_files': ['okcupid_data.csv'],
        'task': 'clf',
        'target': 'orientation',

        'ntbk': 'okcupid_attr_data.ipynb',
        },
    'osha_accidents': {
        'dataset_name': 'osha_accidents',
        'short_name': 'osha',
        'source': 'kaggle',
        'remote_path': 'ruqaiyaship/osha-accident-and-injury-data-1517',
        'files': ['OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv'],
        'rename_files': ['osha_data.csv'],
        'task': 'clf',
        'target': 'Task Assigned',
    
        'ntbk': 'OSHA_accidents_data.ipynb',
        },
    'spotify': {
        'dataset_name': 'spotify',
        'short_name': 'spotify',
        'source': 'kaggle',
        'remote_path': 'maharshipandya/-spotify-tracks-dataset',
        'files': ['dataset.csv'],
        'rename_files': ['spotify_data.csv'],
        'task': 'clf',
        'target': 'track_genre',

        'ntbk': 'spotify_genre_data.ipynb',
        },

    # REGRESSION DATASETS
    'airbnb': {
        'dataset_name': 'airbnb',
        'short_name': 'airbnb',
        'source': 'kaggle',
        'remote_path': 'airbnb/seattle',
        'files': ['listings.csv'],
        'rename_files': ['airbnb_data.csv'],
        'task': 'reg',
        'target': 'price',

        'ntbk': 'airbnb_price_data.ipynb',
        },
    'beer': {
        'dataset_name': 'beer',
        'short_name': 'beer',
        'source': 'kaggle',
        'remote_path': 'ruthgn/beer-profile-and-ratings-data-set',
        'files': ['beer_profile_and_ratings.csv'],
        'rename_files': ['beer_rating.csv'],
        'task': 'reg',
        'target': 'review_overall',

        'ntbk': 'beer_rating_data.ipynb',
        },
    'calif_houses': {
        'dataset_name': 'calif_houses',
        'short_name': 'houses',
        'source': 'kaggle',
        'competition': True,
        'remote_path': 'california-house-prices',
        'files': ['train.csv'],
        'rename_files': ['cf_house_train.csv'],
        'task': 'reg',
        'target': 'Total interior livable area', # or 'Listed Price'

        'ntbk': 'calif_houses_data.ipynb',
        },
    'laptops': {
        'dataset_name': 'laptops',
        'short_name': 'laptops',
        'source': 'kaggle',
        'remote_path': 'dhanushbommavaram/laptop-dataset',
        'files': ['complete laptop data0.csv'],
        'rename_files': ['laptops.csv'],
        'task': 'reg',
        'target': 'Price',

        'ntbk': 'laptops_data.ipynb',
        },
    'mercari': {
        'dataset_name': 'mercari',
        'short_name': 'mercari',
        'source': 'kaggle',
        'remote_path': 'elizabethsam/mercari-price-suggestion-challenge',
        'files': ['train.tsv'],
        'rename_files': ['mercari_price.csv'],
        'task': 'reg',
        'target': 'price',

        'ntbk': 'mercari_price_data.ipynb',
        },
    'sf_permits': {
        'dataset_name': 'sf_permits',
        'short_name': 'permits',
        'source': 'kaggle',
        'remote_path': 'aparnashastry/building-permit-applications-data',
        'files': ['Building_Permits.csv'],
        'rename_files': ['sf_permits.csv'],
        'task': 'reg',
        'target': 'time_to_approve',

        'ntbk': 'sf_permit_time_data.ipynb',
        },
    'wine': {
        'dataset_name': 'wine',
        'source': 'kaggle',
        'remote_path': 'elvinrustam/wine-dataset',
        'files': ['WineDataset.csv'],
        'rename_files': ['wine.csv'],
        'task': 'reg',
        'target': 'Price',
        'ntbk': 'wine_cost_data.ipynb',
        },
}

# EXTRA DATASETS
extra_configs = {
    'covid_trials': {
        'dataset_name': 'covid_trials',
        'short_name': 'covid',
        'source': 'kaggle',
        'remote_path': 'parulpandey/covid19-clinical-trials-dataset',
        'files': ['COVID clinical trials.csv'],
        'rename_files': ['covid_trials_data.csv'],
        'task': 'reg',
        'target': 'days_to_complete',

        'ntbk': 'covid_trials_time_data.ipynb',
        },
    'drugs_rating': {
        'dataset_name': 'drugs_rating',
        'short_name': 'drugs',
        'source': 'kaggle',
        'remote_path': 'jithinanievarghese/drugs-side-effects-and-medical-condition',
        'files': ['drugs_side_effects_drugs_com.csv'],
        'rename_files': ['drugs_rating.csv'],
        'task': 'reg',
        'target': 'rating',

        'ntbk': 'drugs_rating_data.ipynb',
        },
    'insurance_complaints': {
        'dataset_name': 'insurance_complaints',
        'short_name': 'complaints',
        'source': 'kaggle',
        'remote_path': 'adelanseur/insurance-company-complaints',
        'files': ['Insurance_Company_Complaints__Resolutions__Status__and_Recoveries.csv'],
        'rename_files': ['inusrance_data.csv'],
        'task': 'reg',
        'target': 'Recovery',

        'ntbk': 'ins_complaint_money_data.ipynb',
        },
    'it_salary': {
        'dataset_name': 'it_salary',
        'short_name': 'salary',
        'source': 'kaggle',
        'remote_path': 'parulpandey/2020-it-salary-survey-for-eu-region',
        'files': ['IT%20Salary%20Survey%20EU%20%202020.csv'],
        'rename_files': ['IT_salary_eu_data.csv'],
        'task': 'reg',
        'target': 'Yearly brutto salary (without bonus and stocks) in EUR',
        'ntbk': 'IT_eu_salary_data_.ipynb',
        },
    'stack_overflow': {
        'dataset_name': 'stack_overflow',
        'short_name': 'stack_of',
        'source': 'kaggle',
        'remote_path': 'berkayalan/stack-overflow-annual-developer-survey-2024',
        'files': ['survey_results_public.csv'],
        'rename_files': ['stackoverflow_salary_data.csv'],
        'task': 'reg',
        'target': 'Yearly_Salary_USD',

        'ntbk': 'stackoverflow_salary_data.ipynb',
        },
}

def get_dataset_list(datasets_selection):
    """
    Return a list of dataset names based on the selection criteria.
    
    Args:
        datasets_selection (str): Selection criteria for datasets. Options are:
            - 'all': All datasets
            - 'clf': All classification datasets
            - 'reg': All regression datasets
            - Specific dataset name: e.g., 'customer_complaints'
    """
    if datasets_selection == 'all':
        dataset_name_list = get_all_datasets()
    elif datasets_selection == 'clf':
        dataset_name_list = get_classification_datasets()
    elif datasets_selection == 'reg':
        dataset_name_list = get_regression_datasets()
    elif datasets_selection in get_all_datasets():
        dataset_name_list = [datasets_selection]
    else:
        raise ValueError(f"Invalid selection '{datasets_selection}'. Choose 'all', 'clf', 'reg', or a specific dataset name.")
    return dataset_name_list

def get_a_dataset_dict(name):
    """
    Return a specific dataset configuration by name. Fake it as nested dict.
    """

    return data_configs.get(name, None)

def get_all_datasets():
    """
    Return a list of all dataset names
    """
    return list(data_configs.keys())

def get_regression_datasets():
    """
    Return a list of regression datasets names.
    """

    return [config_key for config_key in data_configs.keys() if data_configs[config_key]['task'] == 'reg']

def get_classification_datasets():
    """
    Return a list of classification datasets names.
    """
    return [config_key for config_key in data_configs.keys() if data_configs[config_key]['task'] == 'clf']


if __name__ == "__main__":
    # Example usage
    print("All datasets:", get_all_datasets())
    print("Regression datasets:", get_regression_datasets())
    print("Classification datasets:", get_classification_datasets())
    print("Specific dataset:", get_a_dataset_dict('customer_complaints'))