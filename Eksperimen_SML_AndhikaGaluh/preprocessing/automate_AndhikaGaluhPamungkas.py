import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


def preprocess_heart_disease(
    input_path: str,
    output_path: str,
):
    """
    Fungsi untuk melakukan preprocessing dataset Heart Disease secara otomatis.
    Tahapan preprocessing:
    1. Load dataset
    2. Menghapus data duplikat
    3. Memisahkan fitur dan label
    4. Standarisasi fitur
    5. Menggabungkan kembali fitur dan label
    6. Menyimpan dataset hasil preprocessing
    """

    # 1. Load dataset
    df = pd.read_csv(input_path)

    # 2. Menghapus data duplikat
    df = df.drop_duplicates()

    # 3. Memisahkan fitur dan label
    X = df.drop("target", axis=1)
    y = df["target"]

    # 4. Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # 5. Menggabungkan fitur dan label
    processed_df = X_scaled_df.copy()
    processed_df["target"] = y.values

    # 6. Menyimpan dataset hasil preprocessing
    processed_df.to_csv(output_path, index=False)

    print(f"Preprocessing selesai. Dataset disimpan di: {output_path}")


if __name__ == "__main__":
    # Path
    INPUT_DATASET = "../heart.csv"
    OUTPUT_DATASET = "heart_preprocessing.csv"

    preprocess_heart_disease(
        input_path=INPUT_DATASET,
        output_path=OUTPUT_DATASET
    )
