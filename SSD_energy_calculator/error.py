from pathlib import Path

import pandas as pd


INPUT_FILE = Path("result/Tow_ssd_gamma_filtered.csv")
OUTPUT_FILE = Path("result/two_ssd_error.xlsx")


# 设备对应的实测最大能耗
TRUE_VALUES = {
     "Pichau Aldrin Pro 2TB (TLC)": 7.4,
    "Crucial T700 Pro 4 TB": 11.2,
    "Inland Performance Plus 2 TB":8.3,
    "Fanxiang S770 1 TB": 5.7,
}


def read_csv_auto_encoding(file_path: Path) -> pd.DataFrame:
    """尝试使用常见编码读取CSV文件。"""
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("无法识别CSV文件编码。")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到文件：{INPUT_FILE.resolve()}")

    df = read_csv_auto_encoding(INPUT_FILE)

    # 清理列名
    df.columns = df.columns.astype(str).str.strip()

    required_columns = {"设备", "对应E_tot"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV缺少必要列：{sorted(missing_columns)}，"
            f"当前列为：{df.columns.tolist()}"
        )

    # 清理设备名称前后的空格
    df["设备"] = df["设备"].astype(str).str.strip()

    # 将计算值转换为数字
    df["对应E_tot"] = pd.to_numeric(
        df["对应E_tot"],
        errors="coerce"
    )

    # 检查是否有无法转换的计算值
    invalid_value_rows = df["对应E_tot"].isna()

    if invalid_value_rows.any():
        invalid_rows = df.loc[
            invalid_value_rows,
            ["设备", "对应E_tot"]
        ]

        raise ValueError(
            "以下行的“对应E_tot”不是有效数字：\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    # 根据每一行的设备名称匹配真实值
    df["真实值"] = df["设备"].map(TRUE_VALUES)

    # 检查CSV中是否存在没有真实值的设备
    unmatched_devices = (
        df.loc[df["真实值"].isna(), "设备"]
        .drop_duplicates()
        .tolist()
    )

    if unmatched_devices:
        device_list = "\n".join(
            f"- {device}" for device in unmatched_devices
        )

        raise ValueError(
            "以下设备没有匹配到真实值，请检查设备名称是否完全一致：\n"
            f"{device_list}"
        )

    # 对文件中的每一行计算error
    # # error = |1 - 计算值 / 真实值|
    df["error"] = (
        1 - df["对应E_tot"] / df["真实值"]
    )

    df["error"] = df["error"].round(6)

    output = df[["设备", "error"]].copy()

    if len(output) != len(df):
        raise RuntimeError("输出行数与输入行数不一致。")

    output.to_excel(
        OUTPUT_FILE,
        index=False,
        sheet_name="error结果",
        engine="openpyxl"
    )

    print(f"输入数据行数：{len(df)}")
    print(f"输出结果行数：{len(output)}")
    print(f"结果已保存到：{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
