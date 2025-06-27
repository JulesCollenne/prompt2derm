import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Load JSONL into a list of dictionaries
    with open("prompts\lesion_features_1.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Flatten into a DataFrame
    df = pd.json_normalize(data)

    print(df.columns)
    print(df['class'].value_counts())       # Class balance
    print(df.describe())                    # Summary stats

    features = list(df.columns[df.columns.str.startswith("features.")])

    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="class", y=feature, data=df)
        plt.title(feature)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
