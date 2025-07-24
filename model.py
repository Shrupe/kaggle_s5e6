from config import TEST_CSV, TRAIN_CSV, SAMPLE_SUBMISSION

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.samplers import TPESampler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder

# Load and preprocess data
def load_and_preprocess():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    
    # Fix temperature column name
    train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    test.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    # Feature engineering
    def create_features(df):
        # Nutrient balance
        df['NP_balance'] = df['Nitrogen'] - df['Phosphorous']
        df['NK_ratio'] = np.log1p(df['Nitrogen']) / np.log1p(df['Potassium'] + 1)
        df['PK_ratio'] = np.log1p(df['Phosphorous']) / np.log1p(df['Potassium'] + 1)
        
        # Environmental interactions
        df['Temp_Humidity_Index'] = df['Temperature'] * df['Humidity'] / 100
        df['Moisture_Stress'] = np.where(df['Moisture'] < 30, 1, 0)
        df['Temp_Moisture'] = df['Temperature'] * df['Moisture']
        
        # Soil-Crop interactions
        df['Soil_Crop_Interaction'] = df['Soil Type'] + "_" + df['Crop Type']
        return df

    train = create_features(train)
    test = create_features(test)
    
    # Prepare data
    X = train.drop(columns=['id', 'Fertilizer Name'])
    y = train['Fertilizer Name']
    test_ids = test['id']
    X_test = test.drop(columns=['id'])
    
    # Label encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Target encode categoricals
    cat_cols = ['Soil Type', 'Crop Type', 'Soil_Crop_Interaction']
    encoder = TargetEncoder(cols=cat_cols)
    X_encoded = encoder.fit_transform(X, y_encoded)
    X_test_encoded = encoder.transform(X_test)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    return X_encoded, y_encoded, X_test_encoded, test_ids, le_target, train, weight_dict

# Optimization function
def optimize_hyperparameters(X, y, weight_dict):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'tree_method': 'hist',
            'device': 'cuda',  # Fixed GPU setting
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob'
        }
        
        scores = []
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale within fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Create sample weights
            sample_weights = np.array([weight_dict[cls] for cls in y_train])
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_scaled, 
                y_train,
                sample_weight=sample_weights
            )
            
            preds = model.predict_proba(X_val_scaled)
            scores.append(log_loss(y_val, preds))
            
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50, timeout=3600)
    return study.best_params

# Main workflow
def main():
    # Load and preprocess
    X, y, X_test, test_ids, le_target, train_df, weight_dict = load_and_preprocess()
    n_classes = len(le_target.classes_)
    
    # Class distribution analysis
    plt.figure(figsize=(12, 8))
    train_df['Fertilizer Name'].value_counts().plot(kind='bar')
    plt.title('Fertilizer Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Hyperparameter optimization
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(X, y, weight_dict)
    print(f"Best params: {best_params}")
    
    # Cross-validated evaluation
    cv_scores = []
    oof_preds = np.zeros((len(X), n_classes))
    test_preds = np.zeros((len(X_test), n_classes))
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining Fold {fold+1}/5")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale within fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Create sample weights
        sample_weights = np.array([weight_dict[cls] for cls in y_train])
        
        model = xgb.XGBClassifier(
            **best_params,
            eval_metric='mlogloss',
            objective='multi:softprob'
        )
        model.fit(
            X_train_scaled, 
            y_train,
            sample_weight=sample_weights
        )
        
        fold_val_preds = model.predict_proba(X_val_scaled)
        fold_test_preds = model.predict_proba(X_test_scaled)
        
        oof_preds[val_idx] = fold_val_preds
        test_preds += fold_test_preds / kf.n_splits
        
        score = log_loss(y_val, fold_val_preds)
        cv_scores.append(score)
        print(f"Fold {fold+1} Log Loss: {score:.5f}")
    
    print(f"\nCV Log Loss: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")
    
    # Feature importance
    plt.figure(figsize=(12, 10))
    xgb.plot_importance(model, max_num_features=20)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    
    # Train final model
    print("\nTraining final model...")
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X)
    X_test_scaled = final_scaler.transform(X_test)
    
    # Final sample weights
    sample_weights_full = np.array([weight_dict[cls] for cls in y])
    
    final_model = xgb.XGBClassifier(
        **best_params,
        eval_metric='mlogloss',
        objective='multi:softprob'
    )
    final_model.fit(
        X_full_scaled, 
        y,
        sample_weight=sample_weights_full
    )
    
    # Create submission
    test_probs = final_model.predict_proba(X_test_scaled)
    submission = pd.DataFrame(test_probs, columns=le_target.classes_)
    submission.insert(0, 'id', test_ids)
    submission.to_csv('submission.csv', index=False)
    
    # Save artifacts
    joblib.dump(final_scaler, 'scaler.joblib')
    joblib.dump(le_target, 'label_encoder.joblib')
    joblib.dump(final_model, 'xgb_model.joblib')
    
    print("Submission created!")

if __name__ == "__main__":
    main()