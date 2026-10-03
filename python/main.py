import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("customer_churn_retention_dataset.csv")
print("FULL DATASET")
print(df.to_string(index=False))

# frist five records
print("FIRST 5 RECORDS")
print(df.head())
# last five records
print("\nLAST 5 RECORDS")
print(df.tail())
# dataset shape
print("\nDATASET SHAPE")
print(df.shape)
# columns names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Clean column names so they are easier to work with
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# Treat empty text cells as missing values
df = df.replace(r"^\s*$", pd.NA, regex=True)

# Convert TotalCharges to numeric when that column exists
if "totalcharges" in df.columns:
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
elif "total_charges" in df.columns:
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

# 7. Check data types
print("\n7. DATA TYPES")
print(df.dtypes)

# 8. Display statistical information
print("\n8. NUMERICAL STATISTICS")
print(df.describe())

print("\nCATEGORICAL STATISTICS")
print(df.describe(include="object"))

# 9. Check missing values
print("\n9. MISSING VALUES")
print(df.isna().sum())

# 10. Check duplicate records
print("\n10. DUPLICATE ROWS")
print("Duplicate count:", df.duplicated().sum())

# 11. Remove duplicate records
df = df.drop_duplicates().copy()
print("Rows after removing duplicates:", len(df))

# 12. Check unique values
print("\n12. UNIQUE VALUES")
for column in df.columns:
    print(f"{column}: {df[column].nunique(dropna=True)} unique values")

# 13. Handle missing values
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        median_value = df[column].median()
        if pd.notna(median_value):
            df[column] = df[column].fillna(median_value)
    else:
        df[column] = df[column].fillna("Unknown")

print("\n13. MISSING VALUES AFTER HANDLING")
print(df.isna().sum())


# 14. Convert required columns to suitable data types
for column in [
    "tenure", "tenuremonths", "tenure_months",
    "monthlycharges", "monthly_charges",
    "totalcharges", "total_charges"
]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

print("\n14. DATA TYPES AFTER CONVERSION")
print(df.dtypes)


# 15. Create customer tenure groups
tenure_col = None

for column in ["tenure", "tenuremonths", "tenure_months"]:
    if column in df.columns:
        tenure_col = column
        break

if tenure_col:
    df["tenure_group"] = pd.cut(
        df[tenure_col],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0–12 months", "13–24 months",
                "25–48 months", "49+ months"]
    )
    print("\n15. CUSTOMERS BY TENURE GROUP")
    print(df["tenure_group"].value_counts().sort_index())
else:
    print("\n15. Tenure column was not found.")
# 16. Verify the cleaned dataset
print("\n16. CLEANED DATASET CHECK")
print("Shape:", df.shape)
print("Remaining missing values:", df.isna().sum().sum())
print("Remaining duplicates:", df.duplicated().sum())
print("\nFirst 5 cleaned records:")
print(df.head())

# Save the cleaned data as a new CSV file
df.to_csv("customer_churn_cleaned.csv", index=False)
print("\nCleaned dataset saved as customer_churn_cleaned.csv")


def find_column(*names):
    """Find a column even if its name uses a slightly different format."""
    for name in names:
        if name in df.columns:
            return name
    return None


customer_col = find_column("customerid", "customer_id")
churn_col = find_column("churn")
monthly_col = find_column("monthly_charges", "monthlycharges")
total_col = find_column("total_charges", "totalcharges")
tenure_col = find_column("tenure", "tenure_months", "tenuremonths")
gender_col = find_column("gender")
contract_col = find_column("contract")
internet_col = find_column("internet_service", "internetservice")
payment_col = find_column("payment_method", "paymentmethod")
senior_col = find_column("senior_citizen", "seniorcitizen")
tenure_group_col = find_column("tenure_group")

# Normalize Churn values for consistent analysis
if churn_col:
    df[churn_col] = df[churn_col].astype(str).str.strip().str.lower()

def require_column(column, description):
    if column is None:
        print(f"Column for {description} was not found.")
        return False
    return True


# 17. Find total customers
print("\n17. TOTAL CUSTOMERS")
if require_column(customer_col, "customer ID"):
    print(df[customer_col].nunique())
else:
    print(len(df))


# 18. Find total churned customers
print("\n18. TOTAL CHURNED CUSTOMERS")
if require_column(churn_col, "churn"):
    print((df[churn_col] == "yes").sum())


# 19. Find total retained customers
print("\n19. TOTAL RETAINED CUSTOMERS")
if require_column(churn_col, "churn"):
    print((df[churn_col] == "no").sum())


# 20. Calculate overall churn rate
print("\n20. OVERALL CHURN RATE")
if require_column(churn_col, "churn"):
    valid_churn = df[df[churn_col].isin(["yes", "no"])]
    if len(valid_churn) > 0:
        churn_rate = (valid_churn[churn_col] == "yes").mean() * 100
        print(f"{churn_rate:.2f}%")


# 21. Calculate average monthly charges
print("\n21. AVERAGE MONTHLY CHARGES")
if require_column(monthly_col, "monthly charges"):
    print(df[monthly_col].mean())


# 22. Calculate average total charges
print("\n22. AVERAGE TOTAL CHARGES")
if require_column(total_col, "total charges"):
    print(df[total_col].mean())


# 23. Calculate average customer tenure
print("\n23. AVERAGE CUSTOMER TENURE")
if require_column(tenure_col, "tenure"):
    print(f"{df[tenure_col].mean():.2f} months")


# 24. Find customers by gender
print("\n24. CUSTOMERS BY GENDER")
if require_column(gender_col, "gender"):
    print(df[gender_col].value_counts())


# 25. Find customers by contract type
print("\n25. CUSTOMERS BY CONTRACT TYPE")
if require_column(contract_col, "contract"):
    print(df[contract_col].value_counts())


# 26. Find customers by internet service
print("\n26. CUSTOMERS BY INTERNET SERVICE")
if require_column(internet_col, "internet service"):
    print(df[internet_col].value_counts())


# 27. Find customers by payment method
print("\n27. CUSTOMERS BY PAYMENT METHOD")
if require_column(payment_col, "payment method"):
    print(df[payment_col].value_counts())


# Helper function for churn rates by category
def print_churn_rate_by(category_col, title):
    print(f"\n{title}")
    if category_col is None or churn_col is None:
        print("Required column was not found.")
        return

    valid_rows = df[df[churn_col].isin(["yes", "no"])]
    result = valid_rows.groupby(category_col, observed=False)[churn_col].apply(
        lambda values: (values == "yes").mean() * 100
    )
    print(result.round(2).astype(str) + "%")


# 28. Calculate churn rate by gender
print_churn_rate_by(gender_col, "28. CHURN RATE BY GENDER")


# 29. Calculate churn rate by contract type
print_churn_rate_by(contract_col, "29. CHURN RATE BY CONTRACT TYPE")


# 30. Calculate churn rate by internet service
print_churn_rate_by(internet_col, "30. CHURN RATE BY INTERNET SERVICE")


# 31. Calculate churn rate by payment method
print_churn_rate_by(payment_col, "31. CHURN RATE BY PAYMENT METHOD")


# 32. Calculate churn rate by senior citizen status
print_churn_rate_by(senior_col, "32. CHURN RATE BY SENIOR CITIZEN STATUS")


# 33. Calculate churn rate by tenure group
print_churn_rate_by(tenure_group_col, "33. CHURN RATE BY TENURE GROUP")


# 34. Compare monthly charges of churned and retained customers
print("\n34. MONTHLY CHARGES: CHURNED VS RETAINED")
if monthly_col and churn_col:
    print(df.groupby(churn_col)[monthly_col].mean().round(2))


# 35. Compare tenure of churned and retained customers
print("\n35. TENURE: CHURNED VS RETAINED")
if tenure_col and churn_col:
    print(df.groupby(churn_col)[tenure_col].mean().round(2))


# 36. Find top 10 customers based on total charges
print("\n36. TOP 10 CUSTOMERS BY TOTAL CHARGES")
if total_col:
    columns_to_show = [total_col]
    if customer_col:
        columns_to_show.insert(0, customer_col)
    print(df.nlargest(10, total_col)[columns_to_show])


# 37. Identify customer segments with higher churn
print("\n37. CUSTOMER SEGMENTS WITH HIGHER CHURN")
for category_col in [
    contract_col, internet_col, payment_col, gender_col,
    senior_col, tenure_group_col
]:
    if category_col and churn_col:
        print_churn_rate_by(
            category_col,
            f"Churn rate by {category_col}"
        )


# 38. Identify major factors associated with churn
print("\n38. FACTORS ASSOCIATED WITH CHURN")
print(
    "Compare the churn-rate results above. "
    "Categories with higher percentages are the customer groups "
    "most associated with churn in this dataset."
)
# Create and save the analysis results

analysis_rows = []

def add_analysis(section, category, value):
    analysis_rows.append({
        "analysis": section,
        "category": str(category),
        "result": value
    })

def pick_column(*names):
    for name in names:
        if name in df.columns:
            return name
    return None

customer_col = pick_column("customerid", "customer_id")
churn_col = pick_column("churn")
monthly_col = pick_column("monthly_charges", "monthlycharges")
total_col = pick_column("total_charges", "totalcharges")
tenure_col = pick_column("tenure", "tenure_months", "tenuremonths")

# Standardize churn labels for the calculations
if churn_col:
    churn_values = df[churn_col].astype(str).str.strip().str.lower()

    add_analysis("Total customers", "All customers", len(df))
    add_analysis("Churned customers", "Yes", int((churn_values == "yes").sum()))
    add_analysis("Retained customers", "No", int((churn_values == "no").sum()))

    valid = churn_values.isin(["yes", "no"])
    if valid.any():
        rate = (churn_values[valid] == "yes").mean() * 100
        add_analysis("Overall churn rate (%)", "All customers", round(rate, 2))

# Average charges and tenure
for column, label in [
    (monthly_col, "Average monthly charges"),
    (total_col, "Average total charges"),
    (tenure_col, "Average tenure in months")
]:
    if column:
        add_analysis(label, "All customers", round(df[column].mean(), 2))

# Customer count and churn rate by category
for column in [
    "gender", "contract", "internet_service", "internetservice",
    "payment_method", "paymentmethod", "senior_citizen",
    "seniorcitizen", "tenure_group"
]:
    if column in df.columns:
        for category, count in df[column].value_counts(dropna=False).items():
            add_analysis(f"Customer count by {column}", category, int(count))

        if churn_col:
            temp = pd.DataFrame({
                "category": df[column],
                "churn": churn_values
            })
            temp = temp[temp["churn"].isin(["yes", "no"])]

            for category, group in temp.groupby("category", observed=False):
                rate = (group["churn"] == "yes").mean() * 100
                add_analysis(
                    f"Churn rate by {column} (%)",
                    category,
                    round(rate, 2)
                )

# Top 10 customers by total charges
if total_col:
    top_ten = df.nlargest(10, total_col)
    for rank, (_, row) in enumerate(top_ten.iterrows(), start=1):
        customer = row[customer_col] if customer_col else f"Rank {rank}"
        add_analysis("Top 10 by total charges", customer, round(row[total_col], 2))

analysis_df = pd.DataFrame(analysis_rows)
analysis_df.to_csv("customer_churn_analysis.csv", index=False)
print("\nAnalysis saved to customer_churn_analysis.csv")
# PART 4 — PYTHON VISUALIZATION

import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Find columns using possible name variations
def find_column(*names):
    for name in names:
        if name in df.columns:
            return name
    return None

churn_col = find_column("churn")
contract_col = find_column("contract")
gender_col = find_column("gender")
internet_col = find_column("internetservice", "internet_service")
payment_col = find_column("paymentmethod", "payment_method")
tenure_col = find_column("tenure", "tenuremonths", "tenure_months")
monthly_col = find_column("monthlycharges", "monthly_charges")
total_col = find_column("totalcharges", "total_charges")
tenure_group_col = find_column("tenure_group")

# Create the folder for the 14 chart images
os.makedirs("visualizations", exist_ok=True)

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", filename), dpi=150)
    plt.close()
    print(f"Saved: visualizations/{filename}")

# Make a lowercase churn column for reliable comparisons
if churn_col:
    df["_churn_label"] = df[churn_col].astype(str).str.strip().str.lower()

# 1. Churn distribution — pie chart
if churn_col:
    counts = df["_churn_label"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            startangle=90)
    plt.title("Churn Distribution")
    save_chart("01_churn_distribution.png")

# 2. Customer distribution by contract — bar chart
if contract_col:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=contract_col)
    plt.title("Customer Distribution by Contract")
    plt.xlabel("Contract")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=20)
    save_chart("02_customers_by_contract.png")

# 3. Churn by contract — bar chart
if contract_col and churn_col:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=contract_col, hue="_churn_label")
    plt.title("Churn by Contract")
    plt.xlabel("Contract")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=20)
    save_chart("03_churn_by_contract.png")

# 4. Churn by gender — bar chart
if gender_col and churn_col:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=gender_col, hue="_churn_label")
    plt.title("Churn by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")
    save_chart("04_churn_by_gender.png")

# 5. Churn by internet service — bar chart
if internet_col and churn_col:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=internet_col, hue="_churn_label")
    plt.title("Churn by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=20)
    save_chart("05_churn_by_internet_service.png")

# 6. Churn by payment method — bar chart
if payment_col and churn_col:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=payment_col, hue="_churn_label")
    plt.title("Churn by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=25, ha="right")
    save_chart("06_churn_by_payment_method.png")

# 7. Tenure distribution — histogram
if tenure_col:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=tenure_col, bins=20, kde=True)
    plt.title("Tenure Distribution")
    plt.xlabel("Tenure in Months")
    plt.ylabel("Number of Customers")
    save_chart("07_tenure_distribution.png")

# 8. Monthly charges distribution — histogram
if monthly_col:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=monthly_col, bins=20, kde=True)
    plt.title("Monthly Charges Distribution")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Number of Customers")
    save_chart("08_monthly_charges_distribution.png")

# 9. Tenure vs monthly charges — scatter plot
if tenure_col and monthly_col:
    plt.figure(figsize=(8, 5))
    if churn_col:
        sns.scatterplot(
            data=df, x=tenure_col, y=monthly_col, hue="_churn_label",
            alpha=0.6
        )
    else:
        sns.scatterplot(data=df, x=tenure_col, y=monthly_col, alpha=0.6)
    plt.title("Tenure vs Monthly Charges")
    plt.xlabel("Tenure in Months")
    plt.ylabel("Monthly Charges")
    save_chart("09_tenure_vs_monthly_charges.png")

# 10. Monthly charges by churn — box plot
if monthly_col and churn_col:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="_churn_label", y=monthly_col)
    plt.title("Monthly Charges by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    save_chart("10_monthly_charges_by_churn.png")

# 11. Total charges by churn — box plot
if total_col and churn_col:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="_churn_label", y=total_col)
    plt.title("Total Charges by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Total Charges")
    save_chart("11_total_charges_by_churn.png")

# 12. Churn rate by tenure group — bar chart
if tenure_group_col and churn_col:
    valid = df[df["_churn_label"].isin(["yes", "no"])]
    churn_rates = valid.groupby(tenure_group_col, observed=False)["_churn_label"].apply(
        lambda values: (values == "yes").mean() * 100
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(x=churn_rates.index.astype(str), y=churn_rates.values)
    plt.title("Churn Rate by Tenure Group")
    plt.xlabel("Tenure Group")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=20)
    save_chart("12_churn_rate_by_tenure_group.png")

# 13. Service usage — count plot
service_columns = [
    column for column in [
        find_column("phoneservice", "phone_service"),
        find_column("multiplelines", "multiple_lines"),
        internet_col,
        find_column("onlinesecurity", "online_security"),
        find_column("onlinebackup", "online_backup"),
        find_column("deviceprotection", "device_protection"),
        find_column("techsupport", "tech_support"),
        find_column("streamingtv", "streaming_tv"),
        find_column("streamingmovies", "streaming_movies")
    ] if column is not None
]

if service_columns:
    service_data = df[service_columns].melt(
        var_name="Service",
        value_name="Usage"
    )
    plt.figure(figsize=(12, 6))
    sns.countplot(data=service_data, x="Service", hue="Usage")
    plt.title("Service Usage")
    plt.xlabel("Service")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=35, ha="right")
    save_chart("13_service_usage.png")

# 14. Correlation heatmap
numeric_df = df.select_dtypes(include="number")
if not numeric_df.empty:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    save_chart("14_correlation_heatmap.png")

print("\nVisualization code finished.")
print("Open the visualizations folder in Explorer to see the chart images.")