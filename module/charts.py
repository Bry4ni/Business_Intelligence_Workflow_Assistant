import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_revenue_by_product(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="Product", y="Revenue", estimator=sum, errorbar=None, ax=ax)
    ax.set_title("Total Revenue per Product")
    ax.bar_label(ax.containers[0], fmt="%.0f", label_type="edge", padding=3)
    fig.tight_layout()
    return fig

def plot_monthly_revenue(df):
    df_grouped = df.groupby("Month")[["Revenue"]].sum().reset_index()
    df_grouped["Month"] = df_grouped["Month"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_grouped, x="Month", y="Revenue", marker="o", ax=ax)
    ax.set_title("Monthly Revenue Trend")
    fig.tight_layout()
    return fig

def plot_revenue_by_region(df):
    region_sales = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    region_sales.plot(kind="pie", autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel("")
    ax.set_title("Revenue by Region")
    fig.tight_layout()
    return fig