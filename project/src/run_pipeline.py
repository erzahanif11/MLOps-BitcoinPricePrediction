from ingest_data import main as ingest
from preprocess import main as preprocess

def main():
    print("-" * 30)
    print("Start Data Pipeline")
    print("-" * 30)

    print("Ingesting Data...")
    ingest()

    print("Preprocessing Data...")
    preprocess()

    print("-" * 30)
    print("Data Pipeline Completed")
    print("-" * 30)

if __name__ == "__main__":
    main()