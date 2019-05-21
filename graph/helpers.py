import pandas as pd
import numpy as np
import copy

# Utils
def get_data(network_features_to_keep):
    data_train = pd.read_csv('transfers.csv')
    # Drop club ids because they're irrelevant
    data_train = data_train.drop("from_club_id", axis=1)
    data_train = data_train.drop("to_club_id", axis=1)
    
    # Drop columns
    # Keep features that are not related to network measures
    static_features_col = ['player_nationality', 'value_increase', 'player_stay_in_years', 'club_market_value_from','club_market_value_to', 'transfer_year', 'position', 'birth_date_year']
    static_features_col.extend(network_features_to_keep) # Add network features
    data_train = data_train[static_features_col]
    
    # Transform categorical columns to one hot encoding
    data_train = pd.get_dummies(data_train)

    # Separate features and target
    y_train = data_train['value_increase'].values
    data_train = data_train.drop(columns=['value_increase'], axis=1)
    x_train = data_train.loc[:, data_train.columns != 'value_increase'].values
    
    return x_train, y_train, data_train.columns

## Normalizes each columns of the features
def normalize_feat(features):
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    normalized_feat = (features-mean_feat)/std_feat
    return normalized_feat

# Performs PCA on features and return the first k features of this transformation 
def generate_PCA_features(features, k):
    pca = PCA(n_components=PCA_dim, svd_solver='arpack')
    return pca.fit_transform(features)

def cross_validation(model_ori, input, labels, K=5):
    batch_size = len(input)//K
    errors = np.zeros([K,])
    prev_error = 100
    for k in range(K):
        model = copy.deepcopy(model_ori)
        model.reset()
        test_input, test_labels = input[batch_size*k:batch_size*(k+1)], labels[batch_size*k:batch_size*(k+1)]
        train_input, train_labels = np.concatenate([input[0:batch_size*k], input[batch_size*(k+1):-1]], axis=0), np.concatenate([labels[0:batch_size*k], labels[batch_size*(k+1):-1]], axis=0)

        model.train(train_input, train_labels)
        errors[k] = model.error(test_input, test_labels)
        if errors[k] < prev_error:
            prev_error = errors[k]
            model.save_model()
        #print('Iter {0:d} Percentage Error: {1:.2f}'.format(k, errors[k]))

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error, std_error