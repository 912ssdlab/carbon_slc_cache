#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
from typing import Dict



BASE_GAMMA = 0.59

BASE_TLC = 2069.0
BASE_FOLD = 857.0
BASE_SLC = 688.0
BASE_OP = 0.074
BASE_PAR = 128.0
BASE_GAP = BASE_TLC - BASE_FOLD  # 1212 MB/s




EFFECT_CSV = "gamma_effect_range.csv"
OUTPUT_CSV = "gamma_ranges.csv"




SSD_TABLE = [
    ("S770",        1497,  756,  341, 0.100,  64),
    ("InlandPP",    1800, 1000,  225, 0.1, 64),
    ("T700P",       3300, 1000,  406, 0.100, 192),
    ("PAP2TB",   2159, 863,  691, 0.074, 128),
    ("BBOX570",   6850, 1750,  1600, 0.1, 384),
    ("KX16",   2627, 1031,  231, 0.1, 96),
]




def parse_change_value(text):
    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        str(text)
    )

    if not match:
        raise ValueError(
            f"Cannot parse Parameter Change: {text}"
        )

    return float(match.group())




def load_gamma_effect_ranges(
    csv_path: str,
) -> Dict[str, Dict[str, float]]:

    parameter_info = {
        "SLC Cache": 100.0,
        "Folding Write Speed": 100.0,
        "TLC-Folding Speed Gap": 100.0,
        "Over-Provisioning (OP)": 0.01,
        "Maximum Parallelism": 32.0,
    }

    effects = {}

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "Factor",
            "Range",
        }

        if not required_columns.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "CSV must contain columns: Factor, Range\n"
                f"Actual columns: {reader.fieldnames}"
            )

        for row in reader:

            factor = row["Factor"].strip()
            range_text = row["Range"].strip()

            if factor not in parameter_info:
                raise ValueError(
                    f"Unknown factor: {factor}"
                )

            # Example:
            # [+0.0317, +0.0391]
            range_text = (
                range_text
                .replace("[", "")
                .replace("]", "")
            )

            parts = range_text.split(",")

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid Range for {factor}: "
                    f"{row['Range']}"
                )

            gamma_low = float(
                parts[0].strip()
            )

            gamma_high = float(
                parts[1].strip()
            )

            effects[factor] = {
                "step": parameter_info[factor],
                "low": min(
                    gamma_low,
                    gamma_high,
                ),
                "high": max(
                    gamma_low,
                    gamma_high,
                ),
            }

    return effects




def interval_multiply(scale, low, high):

    v1 = scale * low
    v2 = scale * high

    return min(v1, v2), max(v1, v2)




def parameter_contribution(
    parameter_delta,
    effect
):
    scale = parameter_delta / effect["step"]

    return interval_multiply(
        scale,
        effect["low"],
        effect["high"],
    )




def calculate_gamma_range(
    tlc,
    fold,
    slc,
    op,
    par,
    effects,
):

    gap = tlc - fold



    parameter_deltas = {
        "SLC Cache":
            slc - BASE_SLC,

        "Folding Write Speed":
            fold - BASE_FOLD,

        "TLC-Folding Speed Gap":
            gap - BASE_GAP,

        "Over-Provisioning (OP)":
            op - BASE_OP,

        "Maximum Parallelism":
            par - BASE_PAR,
    }

    gamma_min = BASE_GAMMA
    gamma_max = BASE_GAMMA

    contributions = {}



    for parameter, delta in parameter_deltas.items():

        if parameter not in effects:
            raise KeyError(
                f"Missing parameter in {EFFECT_CSV}: "
                f"{parameter}"
            )

        contribution_min, contribution_max = (
            parameter_contribution(
                delta,
                effects[parameter],
            )
        )

        contributions[parameter] = (
            contribution_min,
            contribution_max,
        )

        gamma_min += contribution_min
        gamma_max += contribution_max

    return gamma_min, gamma_max, contributions




def main():

    # Read parameter-change rules directly from CSV
    effects = load_gamma_effect_ranges(
        EFFECT_CSV
    )

    output_rows = []

    for (
        name,
        tlc,
        fold,
        slc,
        op,
        par,
    ) in SSD_TABLE:

        (
            raw_gamma_min,
            raw_gamma_max,
            contributions,
        ) = calculate_gamma_range(
            tlc=tlc,
            fold=fold,
            slc=slc,
            op=op,
            par=par,
            effects=effects,
        )

        # Clamp gamma to valid [0, 1]
        gamma_min = max(
            0.0,
            min(1.0, raw_gamma_min)
        )

        gamma_max = max(
            0.0,
            min(1.0, raw_gamma_max)
        )

        output_rows.append({
            "Device": name,
            "Gamma_Min": f"{gamma_min:.4f}",
            "Gamma_Max": f"{gamma_max:.4f}",
        })

    # ========================================================
    # Output gamma range CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Device",
                "Gamma_Min",
                "Gamma_Max",
            ],
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"Gamma range CSV written to: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
