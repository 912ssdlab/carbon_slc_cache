
import math
import os
from dataclasses import dataclass
from typing import Optional


def read_float(
    prompt: str,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_blank: bool = False,
) -> Optional[float]:
    while True:
        raw = input(prompt).strip()

        if allow_blank and raw == "":
            return None

        try:
            value = float(raw)
        except ValueError:
            print("输入无效，请输入数字。")
            continue

        if min_value is not None and value < min_value:
            print(f"输入值不能小于 {min_value}。")
            continue

        if max_value is not None and value > max_value:
            print(f"输入值不能大于 {max_value}。")
            continue

        return value


@dataclass
class SSDParameters:
    # Workload / cache parameters
    N_r: float
    N_w: float
    h_SLC: float
    r_SLC: float
    rho_fold: float
    OP: float
    k: float
    N_GC: float
    N_fold: float
    gamma: float

    # Controller
    P_ctrl: float
    T_ctrl: Optional[float]

    # NAND read
    P_r_SLC: float
    T_r_SLC: float
    P_r_TLC: float
    T_r_TLC: float

    # NAND program/write
    P_w_SLC: float
    T_w_SLC: float
    P_w_TLC: float
    T_w_TLC: float

    # NAND erase
    P_e_SLC: float
    T_e_SLC: float
    P_e_TLC: float
    T_e_TLC: float

    # Idle / baseline
    P_idle: float
    t_ac_base: float


def collect_parameters() -> SSDParameters:
    print("=" * 68)
    print("SSD 能耗计算器")
    print("功率请输入 W，时间请输入 s，最终能量单位为 J。")
    print("只需输入工作负载与缓存参数；硬件参数已经固定。")
    print("比例请使用小数，例如 20% 输入 0.20。")
    print("=" * 68)

    print("\n[1] 工作负载与缓存参数")
    N_r = read_float("主机读取页数 N_r: ", min_value=0)
    N_w = read_float("主机写入页数 N_w: ", min_value=0)
    r_SLC = read_float("逻辑 SLC 缓存比例 r_SLC (0~1): ", min_value=0, max_value=1)

    rho_fold = read_float(
        "该 r_SLC 下的 SLC 折叠页数 rho_fold(r_SLC): ",
        min_value=0,
    )
    t_ac_base = read_float("baseline完成时间: ", min_value=0)
    OP = read_float("预留空间比例 OP，例如 20% 输入 0.20: ", min_value=0)
    # k = read_float("有效页比例 k (0~1): ", min_value=0, max_value=1)
    # N_GC = read_float("垃圾回收操作次数 N_GC: ", min_value=0)
    N_fold = read_float("SLC 折叠操作次数 N_fold: ", min_value=0)
    gamma = read_float("后台操作有效影响系数 gamma (0~1): ", min_value=0, max_value=1)

    # 固定硬件参数：无需用户重复输入
    P_ctrl = 0.0
    T_ctrl = 0.0

    P_r_SLC = 0.01102
    T_r_SLC = 0.00003
    P_r_TLC = 0.0139
    T_r_TLC = 0.00014

    P_w_SLC = 0.01529
    T_w_SLC = 0.00016
    P_w_TLC = 0.0168
    T_w_TLC = 0.0031

    P_e_SLC = 0.01538
    T_e_SLC = 0.003
    P_e_TLC = 0.032
    T_e_TLC = 0.0035
    h_SLC = 0.5
    P_idle = 0.03

    # t_ac_base = 60
    #
    # rho_fold =79102
    k = 0
    N_GC = 0
    # N_fold = 32
    # gamma= 0.14

    print("\n固定硬件参数已自动加载，无需再次输入。")

    return SSDParameters(
        N_r=N_r,
        N_w=N_w,
        h_SLC=h_SLC,
        r_SLC=r_SLC,
        rho_fold=rho_fold,
        OP=OP,
        k=k,
        N_GC=N_GC,
        N_fold=N_fold,
        gamma=gamma,
        P_ctrl=P_ctrl,
        T_ctrl=T_ctrl,
        P_r_SLC=P_r_SLC,
        T_r_SLC=T_r_SLC,
        P_r_TLC=P_r_TLC,
        T_r_TLC=T_r_TLC,
        P_w_SLC=P_w_SLC,
        T_w_SLC=T_w_SLC,
        P_w_TLC=P_w_TLC,
        T_w_TLC=T_w_TLC,
        P_e_SLC=P_e_SLC,
        T_e_SLC=T_e_SLC,
        P_e_TLC=P_e_TLC,
        T_e_TLC=T_e_TLC,
        P_idle=P_idle,
        t_ac_base=t_ac_base,
    )


def calculate(p: SSDParameters) -> dict[str, Optional[float]]:
    # Equation (8): additional pages migrated during GC
    denominator = 1.0 + p.OP - p.k
    if denominator <= 0:
        raise ValueError(
            "公式 eta_GC 的分母 1 + OP - k 必须大于 0；"
            f"当前值为 {denominator}。"
        )

    eta_GC = p.rho_fold * p.k / denominator

    # Equation (7): NAND read energy
    slc_read_pages = p.N_r * p.h_SLC + p.rho_fold
    tlc_read_pages = p.N_r * (1.0 - p.h_SLC) + eta_GC

    E_r = (
        slc_read_pages * p.P_r_SLC * p.T_r_SLC
        + tlc_read_pages * p.P_r_TLC * p.T_r_TLC
    )

    # Equation (9): NAND program/write energy
    E_w = (
        p.N_w * p.P_w_SLC * p.T_w_SLC
        + (p.rho_fold+ eta_GC) * p.P_w_TLC * p.T_w_TLC
    )

    # Equation (10): write amplification factor
    WAF = None
    if p.N_w > 0:
        WAF = (p.N_w + p.rho_fold + eta_GC) / p.N_w

    # Equation (11): NAND erase energy
    E_e = (
        p.N_GC * p.P_e_TLC * p.T_e_TLC
        + p.N_fold * p.P_e_SLC * p.T_e_SLC
    )

    # Equation (6): total NAND energy
    E_NAND = E_r + E_w + E_e

    # Equation (13): foreground service time
    t_fg = (
        p.N_r
        * (
            p.h_SLC * p.T_r_SLC
            + (1.0 - p.h_SLC) * p.T_r_TLC
        )
        + p.N_w * p.T_w_SLC
    )

    # Equation (14): background operation time
    t_bg = (
        eta_GC * p.T_r_TLC
        + p.rho_fold * p.T_r_SLC
        + (p.rho_fold + eta_GC) * p.T_w_TLC
        + p.N_GC * p.T_e_TLC
        + p.N_fold * p.T_e_SLC
    )

    # Equation (12): SSD active time
    t_ac = t_fg + p.gamma * t_bg

    # Equation (5): controller energy
    # If T_ctrl is blank, assume controller operates throughout t_ac.
    T_ctrl_used = t_ac if p.T_ctrl is None else p.T_ctrl
    E_ctrl = p.P_ctrl * T_ctrl_used

    # Equation (4): active-mode energy
    E_active = E_ctrl + E_NAND

    # Equation (15): differential idle energy
    E_idle = p.P_idle * max(p.t_ac_base - t_ac, 0.0)
    #E_idle=0
    # Equation (3): total energy
    E_tot = E_active + E_idle

    return {
        "eta_GC": eta_GC,
        "slc_read_pages": slc_read_pages,
        "tlc_read_pages": tlc_read_pages,
        "WAF": WAF,
        "t_fg": t_fg,
        "t_bg": t_bg,
        "t_ac": t_ac,
        "T_ctrl_used": T_ctrl_used,
        "E_r": E_r,
        "E_w": E_w,
        "E_e": E_e,
        "E_NAND": E_NAND,
        "E_ctrl": E_ctrl,
        "E_active": E_active,
        "E_idle": E_idle,
        "E_tot": E_tot,
    }



import csv



SSD_TABLE = [

    ("Pichau Aldrin Pro 2TB (TLC)", 0.074, 256.928, 82944, 36.0, 55232),
   (
        "Fanxiang S770 1 TB ",
        0.100, 198.9952, 47424, 20.58333333, 48384
    ),
    (
        "Inland Performance Plus 2 TB",
        0.100,
        251.3066667,
        51200,
        14.81481481,
        64000,
    ),

    (
        "Crucial T700 Pro 4 TB",
        0.100, 350.5066667, 147200, 53.56622999, 64000
    ),
(
        "ASMI70",
        0.074, 396.0064, 103488, 24.7816092, 396.0064
    ),
(
        "SNNEM-PA",
        0.100, 156.8021333, 23104, 17.19047619, 42880

    ),


]
DEVICE_NAME_MAP = {
    "S770": "Fanxiang S770 1 TB",
    "InlandPP": "Inland Performance Plus 2 TB",
    "T700P": "Crucial T700 Pro 4 TB",
    "PAP2TB": "Pichau Aldrin Pro 2TB (TLC)",

}

MEASURED_ENERGY = {
    "Inland Performance Plus 2 TB": 8.3,
    "Pichau Aldrin Pro 2TB (TLC)": 7.4,
    "Crucial T700 Pro 4 TB": 11.2,
    "Fanxiang S770 1 TB ": 5.7,

}


N_R_DEFAULT = 0.0    # 主机读取页数，表格未给出，默认按纯写入场景处理
R_SLC_DEFAULT = 0.0  # calculate() 中未实际使用，仅保留字段
K_DEFAULT = 0.0
N_GC_DEFAULT = 0.0


def build_parameters(op, t_ac_base, rho_fold, n_fold, n_w, gamma) -> SSDParameters:
    return SSDParameters(
        N_r=N_R_DEFAULT,
        N_w=n_w,
        h_SLC=0.5,
        r_SLC=R_SLC_DEFAULT,
        rho_fold=rho_fold,
        OP=op,
        k=K_DEFAULT,
        N_GC=N_GC_DEFAULT,
        N_fold=n_fold,
        gamma=gamma,
        P_ctrl=0.0,
        T_ctrl=0.0,
        P_r_SLC=0.01102,
        T_r_SLC=0.00003,
        P_r_TLC=0.0139,
        T_r_TLC=0.00014,
        P_w_SLC=0.01529,
        T_w_SLC=0.00016,
        P_w_TLC=0.0168,
        T_w_TLC=0.0031,
        P_e_SLC=0.01538,
        T_e_SLC=0.003,
        P_e_TLC=0.032,
        T_e_TLC=0.0035,
        P_idle=0.03,
        t_ac_base=t_ac_base,
    )

def load_gamma_ranges(csv_path="gamma_ranges.csv"):
    gamma_ranges = {}

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            device = row["Device"].strip()
            gamma_min = float(row["Gamma_Min"])
            gamma_max = float(row["Gamma_Max"])

            gamma_ranges[device] = (gamma_min, gamma_max)

    return gamma_ranges
def generate_gammas(gamma_min, gamma_max):
    start = math.floor(gamma_min * 100 + 1e-9)
    end = math.ceil(gamma_max * 100 - 1e-9)

    return [i / 100 for i in range(start, end + 1)]
def main() -> None:
    gamma_ranges = load_gamma_ranges("gamma_ranges.csv")

    # 完整 SSD 名称 -> gamma_range.csv 中的 Device 名称
    name_to_device = {
        full_name: device
        for device, full_name in DEVICE_NAME_MAP.items()
    }

    rows = []

    for name, op, t_ac_base, rho_fold, n_fold, n_w in SSD_TABLE:

        # 防止 SSD_TABLE 里的名字末尾有空格
        clean_name = name.strip()

        device = name_to_device.get(clean_name)

        if device is None:
            print(f"[跳过] 没有找到设备映射: {clean_name}")
            continue

        if device not in gamma_ranges:
            print(f"[跳过] gamma_range.csv 中没有 {device}")
            continue

        gamma_min, gamma_max = gamma_ranges[device]

        gammas = generate_gammas(gamma_min, gamma_max)

        for gamma in gammas:
            params = build_parameters(
                op,
                t_ac_base,
                rho_fold,
                n_fold,
                n_w,
                gamma,
            )

            result = calculate(params)

            rows.append({
                "设备": clean_name,
                "gamma取值": gamma,
                "对应E_tot": result["E_tot"],
            })

    os.makedirs("result", exist_ok=True)

    out_path = "result/Tow_ssd_gamma_filtered.csv"

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["设备", "gamma取值", "对应E_tot"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"共计算 {len(rows)} 条结果，已写入 {out_path}")
#

if __name__ == "__main__":
    main()