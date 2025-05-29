from data_process.data_preprocessing import load_data, preprocess_data
from model_training import train_models
from visualisation import plot_feature_importance
from sklearn.ensemble import RandomForestClassifier
def main():
    # Load and preprocess data
    data = load_data('../data/P1data5117.csv')
    processed_data = preprocess_data(data)
    
    feature_names = data.drop(columns=['Y', 'Ya', 'Yb', 'Yc']).columns
    # Train and evaluate models
    results = train_models(
        processed_data['X_train'],
        processed_data['X_test'],
        processed_data['y_train'],
        processed_data['y_test'],
        feature_names= feature_names
    )
    
    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Performance:")
        for metric_name, value in metrics.items():
            if metric_name != 'Confusion Matrix':
                if metric_name == 'Confusion Matrix':
                    continue
                elif metric_name == 'Best Params':
                    print("Best Parameters:")
                    for param, val in value.items():
                        print(f"  {param}: {val}")
                else:
                    print(f"{metric_name}: {value:.4f}")
        


if __name__ == "__main__":
    
    main()

