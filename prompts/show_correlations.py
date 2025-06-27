from os.path import join

import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load JSONL into a list of dictionaries
    with open(join("prompts","lesion_features_1.jsonl"), "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Flatten into a DataFrame
    df = pd.json_normalize(data)

    # Step 1: Ensure class is numeric
    df["class_numeric"] = df["class"].map({"nev": 0, "mel": 1})
    # Optional: print mapping
    print(dict(enumerate(df['class'].astype('category').cat.categories)))

    # Step 2: Build full feature list including class
    features = list(df.columns[df.columns.str.startswith("features.")])

    # Drop the 'features.Evolving' entry if it exists
    features = [f for f in features if f != "features.Evolving"]
    features.append('class_numeric')

    # Step 3: Compute correlation and plot heatmap
    corr = df[features].corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
    plt.title("Feature Correlation Matrix with Class")
    plt.show()

