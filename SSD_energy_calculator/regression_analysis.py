import pandas as pd
import statsmodels.api as sm
from pathlib import Path


# ============================================================
# 1. 输入文件
# ============================================================

INPUT_FILE = Path("regression_data.xlsx")

# 如果你的列名叫 BVF，则改成：
# TARGET_COL = "BVF"

TARGET_COL = "gamma"


# ============================================================
# 2. 读取数据
# ============================================================

if INPUT_FILE.suffix.lower() == ".csv":
    df = pd.read_csv(INPUT_FILE)
else:
    df = pd.read_excel(INPUT_FILE)




df["Gap"] = (
    df["写TLC速率"]
    -
    df["折叠时写入速率"]
)


# ============================================================
# 4. 准备五个因素和 BVF
# ============================================================

data = pd.DataFrame({

    "SLC":
        df["slc cache"],

    "Folding":
        df["折叠时写入速率"],

    "Gap":
        df["Gap"],

    "OP":
        df["OP"],

    "Parallelism":
        df["理论最大并行数"],

    "BVF":
        df[TARGET_COL],

})


# 删除存在空值的设备

data = data.dropna().copy()



factor_names = [
    "SLC",
    "Folding",
    "Gap",
    "OP",
    "Parallelism",
]


X = data[factor_names].astype(float)

y = data["BVF"].astype(float)




X_std = (
    X - X.mean()
) / X.std(ddof=0)


y_std = (
    y - y.mean()
) / y.std(ddof=0)


# 加截距项

X_model = sm.add_constant(X_std)




model = sm.OLS(
    y_std,
    X_model
).fit()




coefficients = pd.DataFrame({

    "Factor": factor_names,

    "Standardized Beta": [
        model.params[factor]
        for factor in factor_names
    ]

})



F_statistic = model.fvalue

F_pvalue = model.f_pvalue

R_squared = model.rsquared




print()
print("=" * 70)
print("Five-factor Regression Analysis for BVF")
print("=" * 70)




print()
print("1. Standardized Regression Coefficients")
print("-" * 70)

for _, row in coefficients.iterrows():

    print(
        f"{row['Factor']:<15s}"
        f"{row['Standardized Beta']:+.4f}"
    )



print()
print("2. Overall Regression Statistic")
print("-" * 70)

print(
    f"F-statistic = {F_statistic:.4f}"
)

print(
    f"F-test p-value = {F_pvalue:.6g}"
)




print()
print("3. Model Explanatory Power")
print("-" * 70)

print(
    f"R-squared = {R_squared:.4f}"
)

print(
    f"Explained BVF variation = "
    f"{R_squared * 100:.2f}%"
)



print()
print("=" * 70)
print("Interpretation")
print("=" * 70)


if F_pvalue < 0.05:

    print(
        "Overall regression: statistically significant "
        "(p < 0.05)"
    )

else:

    print(
        "Overall regression: not statistically significant "
        "(p >= 0.05)"
    )


print(
    f"The five factors explain "
    f"{R_squared * 100:.2f}% "
    f"of the BVF variation."
)




coefficients.to_csv(
    "regression_coefficients.csv",
    index=False,
    encoding="utf-8-sig"
)


summary = pd.DataFrame({

    "Metric": [
        "F-statistic",
        "F-test p-value",
        "R-squared",
        "Explained variation (%)",
    ],

    "Value": [
        F_statistic,
        F_pvalue,
        R_squared,
        R_squared * 100,
    ]

})


summary.to_csv(
    "regression_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


print()
print("Saved:")
print("  regression_coefficients.csv")
print("  regression_summary.csv")
print()
