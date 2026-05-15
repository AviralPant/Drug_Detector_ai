from kaggle_dataset_integration import KaggleDatasetIntegration

def main():
    print("Setting up Kaggle dataset...")
    
    # Initialize the integration
    integration = KaggleDatasetIntegration()
    
    # Download the dataset
    if integration.download_dataset():
        print("Dataset downloaded and organized successfully!")
    else:
        print("Failed to download dataset.")

if __name__ == "__main__":
    main()