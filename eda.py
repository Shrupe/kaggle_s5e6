import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from config import TEST_CSV, TRAIN_CSV

# Load train.csv
df = pd.read_csv(TRAIN_CSV)

# Basic preview of data
def preview_data(df):
    print("====================df.head====================")
    print(df.head(10))
    print("====================df.info====================")
    print(df.info())
    print("====================df.describe====================")
    print(df.describe())
    print("========================================")
    print(df['Fertilizer Name'].value_counts())
    print("==================================")

def separate_columns_by_type(df):
    """
    Separates DataFrame columns into numerical and categorical lists.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical (object/category) column names.
    """
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical_cols, categorical_cols


numerical_cols, categorical_cols = separate_columns_by_type(df)
numerical_cols.remove("id")
categorical_cols.remove("Fertilizer Name")

def plot_numerical_cols(df):
    for feature in numerical_cols:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[feature])
        plt.title(f"Box Plot of {feature}")

        plt.tight_layout()
        plt.show()

        print(f"\nStatistics for {feature}:")
        print(f"Skewness: {df[feature].skew():.2f}")
        print(f"Number of Missing Values: {df[feature].isnull().sum()}")

def plot_categorical_cols(df):
    for feature in categorical_cols:
        counts = df[feature].value_counts()

        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(f"Distribution of {feature}")
        plt.axis("equal")
        plt.show()

        print(f"Number of Unique {feature}: {df[feature].nunique()}")
        print(f"Missing Values in {feature}: {df[feature].isnull().sum()}")


def kde_plot(df):
    colors = sns.color_palette('husl', len(numerical_cols))

    rows = -(-len(numerical_cols) // 4)
    plt.figure(figsize=(20, 5 * rows))

    for i, (col, color) in enumerate(zip(numerical_cols, colors), 1):
        plt.subplot(rows, 4, i)
        sns.kdeplot(data=df, x=col, fill=True, color=color)
        plt.title(f'KDE Plot of {col}', fontsize=14, color=color)
        plt.xlabel(col)
        plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

def scatter_plot(df):
    numeric_df = df.select_dtypes(include='number')

    sns.pairplot(numeric_df, corner=True, plot_kws={'alpha': 0.5})
    plt.suptitle('Pairwise Scatter Plots', y=1.02)
    plt.show()

def nuumericals_vs_label(df):
    for feature in numerical_cols[:-1]:  
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=df_df[feature], y=df["Fertilizer Name"], alpha=0.5
        )
        plt.title(f"{feature} vs. Fertilizer Name")
        plt.xlabel(feature)
        plt.ylabel("Fertilizer Name")
        plt.show()

    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()


def soil_vs_label(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Soil Type", hue="Fertilizer Name", data=df)
    plt.title("Distribution of Fertilizer Name across Soil Types")
    plt.xlabel("Soil Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Fertilizer Name", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    cross_tab = pd.crosstab(df["Soil Type"], df["Fertilizer Name"])

    plt.figure(figsize=(12, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Soil Type vs. Fertilizer Name (Counts)")
    plt.ylabel("Soil Type")
    plt.xlabel("Fertilizer Name")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()    

def crop_vs_label(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Crop Type", hue="Fertilizer Name", data=df)
    plt.title("Distribution of Fertilizer Name across Crop Types")
    plt.xlabel("Crop Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Fertilizer Name", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    cross_tab = pd.crosstab(df["Crop Type"], df["Fertilizer Name"])

    plt.figure(figsize=(12, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Crop Type vs. Fertilizer Name (Counts)")
    plt.ylabel("Crop Type")
    plt.xlabel("Fertilizer Name")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def trend_plot(df):
    colors = sns.color_palette('husl', len(numerical_cols))

    rows = -(-len(numerical_cols) // 4)
    plt.figure(figsize=(20, 5 * rows))

    for i, (col, color) in enumerate(zip(numerical_cols, colors), 1):
        plt.subplot(rows, 4, i)
        sns.lineplot(data=df[col], color=color)
        plt.title(f'Trend Plot of {col}', fontsize=14, color=color)
        plt.xlabel('Index')
        plt.ylabel(col)

    plt.tight_layout()
    plt.show()

def kde_trend_plot(df):
    colors = sns.color_palette('husl', len(numerical_cols))
    rows = -(-len(numerical_cols) // 4)
    plt.figure(figsize=(20, 5 * rows))

    for i, (col, color) in enumerate(zip(numerical_cols, colors), 1):
        plt.subplot(rows, 4, i)
        sns.kdeplot(data=df, x=col, fill=True, color=color)
        sns.lineplot(data=df[col].sort_values().reset_index(drop=True), color='black', linewidth=1)
        plt.title(f'KDE + Trend of {col}', fontsize=14, color=color)
        plt.xlabel(col)
        plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

def violin_plot(df):
    colors = sns.color_palette('husl', len(numerical_cols))
    rows = -(-len(numerical_cols) // 4)
    plt.figure(figsize=(20, 5 * rows))

    for i, (col, color) in enumerate(zip(numerical_cols, colors), 1):
        plt.subplot(rows, 4, i)
        sns.violinplot(data=df, y=col, color=color)
        plt.title(f'Violin Plot of {col}', fontsize=14, color=color)
        plt.xlabel('')
        plt.ylabel(col)

    plt.tight_layout()
    plt.show()



