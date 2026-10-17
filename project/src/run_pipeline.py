from ingest_data import main as ingest
from preprocess import main as preprocess
from feature_engineering import main as feature_engineer

def main():
    print("-" * 30)
    print("Start Data Pipeline")
    print("-" * 30)

    print("[1/3] Ingesting Data...")
    ingest()

    print("[2/3] Preprocessing Data...")
    preprocess()

    print("[3/3] Creating Features...")
    feature_engineer()

    print("-" * 30)
    print("Data Pipeline Completed")
    print("-" * 30)

if __name__ == "__main__":
    main()