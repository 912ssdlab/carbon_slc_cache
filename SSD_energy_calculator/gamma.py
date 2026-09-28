import csv
from typing import Dict, List, Tuple

import numpy as np


BASELINE_DEVICE = "NV2000ME5"

BASELINE_GAMMA = 0.59

MEASURED_WINDOW_S = 1.0


GAMMA_SWEEP_STEP = 0.001


EFFECT_SEARCH_STEP = 0.0001


MEDIAN_ERROR_MARGIN = 0.005
MEAN_ERROR_MARGIN = 0.010



DEVICE_MATCH_TOLERANCE = 0.05


ANCHOR_INFERRED_GAMMAS = True


MAX_ITERATIVE_ROUNDS = 20

CONVERGENCE_TOLERANCE = 1e-6



ARCH_TABLE = {

    "KC3000": {
        "tlc_write": 1900.0,
        "folding_write": 1015.0,
        "slc_cache": 369.0,
        "OP": 0.074,
        "max_parallelism": 64.0,
    },

    "SN850X-1": {
        "tlc_write": 1500.0,
        "folding_write": 900.0,
        "slc_cache": 300.0,
        "OP": 0.100,
        "max_parallelism": 64.0,
    },

    "RedB": {
        "tlc_write": 840.0,
        "folding_write": 390.0,
        "slc_cache": 83.0,
        "OP": 0.0740,
        "max_parallelism": 48.0,
    },

    "PM981": {
        "tlc_write": 1200.0,
        "folding_write": 750.0,
        "slc_cache": 50.0,
        "OP": 0.074,
        "max_parallelism": 128.0,
    },

    "Z540": {
        "tlc_write": 3697.0,
        "folding_write": 1800.0,
        "slc_cache": 210.0,
        "OP": 0.100,
        "max_parallelism": 96.0,
    },
"MP600": {
        "tlc_write": 3500.0,
        "folding_write": 1500.0,
        "slc_cache": 220.0,
        "OP": 0.100,
        "max_parallelism": 64.0,
    },

    "R4PG": {
        "tlc_write": 3960.0,
        "folding_write": 1620.0,
        "slc_cache": 226.0,
        "OP": 0.100,
        "max_parallelism": 64.0,
    },

    "SF530": {
        "tlc_write": 3850.0,
        "folding_write": 1600.0,
        "slc_cache": 225.0,
        "OP": 0.100,
        "max_parallelism": 64.0,
    },

    "SN850X-8": {
        "tlc_write": 2850.0,
        "folding_write": 1200.0,
        "slc_cache": 2350.0,
        "OP": 0.100,
        "max_parallelism": 256.0,
    },

    "MP700Pro": {
        "tlc_write": 4000.0,
        "folding_write": 1500.0,
        "slc_cache": 440.0,
        "OP": 0.100,
        "max_parallelism": 192.0,
    },

    "ADL960": {
        "tlc_write": 3590.0,
        "folding_write": 1280.0,
        "slc_cache": 200.0,
        "OP": 0.074,
        "max_parallelism": 256.0,
    },

    "NV2000ME5": {
        "tlc_write": 2069.0,
        "folding_write": 857.0,
        "slc_cache": 688.0,
        "OP": 0.074,
        "max_parallelism": 128.0,
    },

    "GAG5": {
        "tlc_write": 3500.0,
        "folding_write": 1400.0,
        "slc_cache": 220.0,
        "OP": 0.100,
        "max_parallelism": 96.0,
    },
}


ENERGY_TABLE = {

    "KC3000": {
        "OP": 0.074,
        "t_ac_base": 259.904,
        "rho_fold": 56640.0,
        "N_fold": 26.81818182,
        "N_w": 64960.0,
    },

    "SN850X-1": {
        "OP": 0.100,
        "t_ac_base": 218.24,
        "rho_fold": 38400.0,
        "N_fold": 28.57142857,
        "N_w": 57600.0,
    },

    "RedB": {
        "OP": 0.0740,
        "t_ac_base": 107.136,
        "rho_fold": 28800.0,
        "N_fold": 7.0,
        "N_w": 24960.0,
    },

    "PM981": {
        "OP": 0.074,
        "t_ac_base": 178.56,
        "rho_fold": 28800.0,
        "N_fold": 18.75,
        "N_w": 48000.0,
    },

    "Z540": {
        "OP": 0.100,
        "t_ac_base": 482.5749333,
        "rho_fold": 121408.0,
        "N_fold": 43.6091954,
        "N_w": 115200.0,
    },
"MP600": {
        "OP": 0.100,
        "t_ac_base": 429.86,
        "rho_fold": 12800.0,
        "N_fold": 60,
        "N_w": 96000.0,
    },

    "R4PG": {
        "OP": 0.100,
        "t_ac_base": 476.16,
        "rho_fold": 149760.0,
        "N_fold": 70.9,
        "N_w": 103680.0,
    },

    "SF530": {
        "OP": 0.100,
        "t_ac_base": 466.24,
        "rho_fold": 144000.0,
        "N_fold": 68.18181818,
        "N_w": 102400.0,
    },


    "SN850X-8": {
        "OP": 0.100,
        "t_ac_base": 347.2,
        "rho_fold": 105600.0,
        "N_fold": 28.85245902,
        "N_w": 76800.0,
    },

    "MP700Pro": {
        "OP": 0.100,
        "t_ac_base": 462.9333333,
        "rho_fold": 160000.0,
        "N_fold": 57.47126437,
        "N_w": 96000.0,
    },

    "ADL960": {
        "OP": 0.074,
        "t_ac_base": 406.72,
        "rho_fold": 147840.0,
        "N_fold": 70.0,
        "N_w": 81920.0,
    },

    "NV2000ME5": {
        "OP": 0.074,
        "t_ac_base": 250.1824,
        "rho_fold": 77568.0,
        "N_fold": 33.67,
        "N_w": 54848.0,
    },

    "GAG5": {
        "OP": 0.100,
        "t_ac_base": 416.64,
        "rho_fold": 134400.0,
        "N_fold": 48.27,
        "N_w": 86900.0,
    },
}


# ================================================================
# 4. Measured maximum power
# ================================================================

MEASURED_POWER_W = {

    "KC3000": 6.0,

    "SN850X-1": 6.3,

    "RedB": 2.9,

    "PM981": 5.9,

    "Z540": 12.6,
    "MP600":7.5,

    "R4PG": 9.4,

    "SF530": 7.8,

    "SN850X-8": 7.4,

    "MP700Pro": 12.3,

    "ADL960": 8.3,

    "NV2000ME5": 7.11,

    "GAG5": 11.3,
}



N_R = 0.0

H_SLC = 0.5

K = 0.0

N_GC = 0.0


# Controller
P_CTRL = 0.0
T_CTRL = 0.0


# NAND read
P_R_SLC = 0.01102
T_R_SLC = 0.00003

P_R_TLC = 0.0139
T_R_TLC = 0.00014


# NAND write
P_W_SLC = 0.01529
T_W_SLC = 0.00016

P_W_TLC = 0.0168
T_W_TLC = 0.0031


# NAND erase
P_E_SLC = 0.01538
T_E_SLC = 0.003

P_E_TLC = 0.032
T_E_TLC = 0.0035


# Idle
P_IDLE = 0.03


# ================================================================
# 6. Energy model
# ================================================================

def calculate_energy(
    device: str,
    gamma: float,
) -> Dict[str, float]:

    d = ENERGY_TABLE[device]

    op = d["OP"]
    t_ac_base = d["t_ac_base"]
    rho_fold = d["rho_fold"]
    n_fold = d["N_fold"]
    n_w = d["N_w"]

    # ------------------------------------------------------------
    # GC
    # ------------------------------------------------------------

    denominator = 1.0 + op - K

    if denominator <= 0:

        raise ValueError(
            f"{device}: 1 + OP - K must be > 0."
        )

    eta_gc = (
        rho_fold
        * K
        / denominator
    )

    # ------------------------------------------------------------
    # NAND read energy
    # ------------------------------------------------------------

    slc_read_pages = (
        N_R * H_SLC
        + rho_fold
    )

    tlc_read_pages = (
        N_R * (1.0 - H_SLC)
        + eta_gc
    )

    e_r = (
        slc_read_pages
        * P_R_SLC
        * T_R_SLC

        +

        tlc_read_pages
        * P_R_TLC
        * T_R_TLC
    )

    # ------------------------------------------------------------
    # NAND write energy
    # ------------------------------------------------------------

    e_w = (
        n_w
        * P_W_SLC
        * T_W_SLC

        +

        (
            rho_fold
            + eta_gc
        )
        * P_W_TLC
        * T_W_TLC
    )

    # ------------------------------------------------------------
    # NAND erase energy
    # ------------------------------------------------------------

    e_e = (
        N_GC
        * P_E_TLC
        * T_E_TLC

        +

        n_fold
        * P_E_SLC
        * T_E_SLC
    )

    # ------------------------------------------------------------
    # Total NAND energy
    # ------------------------------------------------------------

    e_nand = (
        e_r
        + e_w
        + e_e
    )

    # ------------------------------------------------------------
    # Foreground time
    # ------------------------------------------------------------

    t_fg = (
        N_R
        * (
            H_SLC
            * T_R_SLC

            +

            (1.0 - H_SLC)
            * T_R_TLC
        )

        +

        n_w
        * T_W_SLC
    )

    # ------------------------------------------------------------
    # Background time
    # ------------------------------------------------------------

    t_bg = (
        eta_gc
        * T_R_TLC

        +

        rho_fold
        * T_R_SLC

        +

        (
            rho_fold
            + eta_gc
        )
        * T_W_TLC

        +

        N_GC
        * T_E_TLC

        +

        n_fold
        * T_E_SLC
    )

    # ------------------------------------------------------------
    # Active time
    #
    # Gamma enters here
    # ------------------------------------------------------------

    t_ac = (
        t_fg
        + gamma * t_bg
    )

    # ------------------------------------------------------------
    # Controller energy
    # ------------------------------------------------------------

    e_ctrl = (
        P_CTRL
        * T_CTRL
    )

    # ------------------------------------------------------------
    # Active energy
    # ------------------------------------------------------------

    e_active = (
        e_ctrl
        + e_nand
    )

    # ------------------------------------------------------------
    # Differential idle energy
    # ------------------------------------------------------------

    e_idle = (
        P_IDLE
        *
        max(
            t_ac_base
            - t_ac,
            0.0,
        )
    )

    # ------------------------------------------------------------
    # Total energy
    # ------------------------------------------------------------

    e_tot = (
        e_active
        + e_idle
    )

    return {

        "eta_GC": eta_gc,

        "t_fg": t_fg,

        "t_bg": t_bg,

        "t_ac": t_ac,

        "E_NAND": e_nand,

        "E_idle": e_idle,

        "E_tot": e_tot,
    }


# ================================================================
# 7. Build Gamma grid
# ================================================================

def gamma_grid() -> np.ndarray:

    count = int(
        round(
            1.0
            / GAMMA_SWEEP_STEP
        )
    )

    values = [

        round(
            i * GAMMA_SWEEP_STEP,
            10,
        )

        for i in range(
            count + 1
        )
    ]

    return np.array(
        values,
        dtype=float,
    )


# ================================================================
# 8. Infer Gamma from measured E_tot
# ================================================================

def infer_gamma_from_measurement(
    device: str,
) -> Dict:

    target_energy = (
        MEASURED_POWER_W[device]
        * MEASURED_WINDOW_S
    )

    gammas = gamma_grid()

    energies = np.array(

        [
            calculate_energy(
                device,
                gamma,
            )["E_tot"]

            for gamma in gammas
        ],

        dtype=float,
    )

    min_energy = float(
        np.min(energies)
    )

    max_energy = float(
        np.max(energies)
    )

    closest_index = int(
        np.argmin(
            np.abs(
                energies
                - target_energy
            )
        )
    )

    closest_gamma = float(
        gammas[closest_index]
    )

    status = "OK"

    gamma_interpolated = closest_gamma

    bracket_low = closest_gamma
    bracket_high = closest_gamma

    # ------------------------------------------------------------
    # Measured target outside gamma=[0,1] model range
    # ------------------------------------------------------------

    if (
        target_energy < min_energy
        or
        target_energy > max_energy
    ):

        status = "OUT_OF_MODEL_RANGE"

    else:

        found = False

        for i in range(
            len(gammas) - 1
        ):

            e1 = float(
                energies[i]
            )

            e2 = float(
                energies[i + 1]
            )

            if (
                (e1 - target_energy)
                *
                (e2 - target_energy)
                <= 0
            ):

                g1 = float(
                    gammas[i]
                )

                g2 = float(
                    gammas[i + 1]
                )

                bracket_low = min(
                    g1,
                    g2,
                )

                bracket_high = max(
                    g1,
                    g2,
                )

                if abs(
                    e2 - e1
                ) > 1e-15:

                    gamma_interpolated = (
                        g1
                        +
                        (
                            target_energy
                            - e1
                        )
                        *
                        (
                            g2 - g1
                        )
                        /
                        (
                            e2 - e1
                        )
                    )

                else:

                    gamma_interpolated = g1

                found = True

                break

        if not found:

            gamma_interpolated = (
                closest_gamma
            )

    gamma_interpolated = float(
        min(
            1.0,
            max(
                0.0,
                gamma_interpolated,
            )
        )
    )

    energy_at_gamma = calculate_energy(
        device,
        gamma_interpolated,
    )["E_tot"]

    return {

        "measured_power_W":
            MEASURED_POWER_W[device],

        "target_energy_J":
            target_energy,

        "gamma_grid_closest":
            closest_gamma,

        "gamma_bracket_low":
            bracket_low,

        "gamma_bracket_high":
            bracket_high,

        "gamma_inferred_raw":
            gamma_interpolated,

        "E_tot_at_gamma":
            energy_at_gamma,

        "energy_abs_error":
            abs(
                energy_at_gamma
                - target_energy
            ),

        "model_E_min":
            min_energy,

        "model_E_max":
            max_energy,

        "status":
            status,
    }


# ================================================================
# 9. Infer Gamma for all SSDs
# ================================================================

def infer_all_target_gammas():

    details = {}

    raw_gamma = {}

    for device in ARCH_TABLE:

        result = (
            infer_gamma_from_measurement(
                device
            )
        )

        details[device] = result

        raw_gamma[device] = (
            result[
                "gamma_inferred_raw"
            ]
        )

    # ------------------------------------------------------------
    # Anchor:
    # NV2000ME5 = 0.59
    # ------------------------------------------------------------

    if ANCHOR_INFERRED_GAMMAS:

        offset = (
            BASELINE_GAMMA
            -
            raw_gamma[
                BASELINE_DEVICE
            ]
        )

    else:

        offset = 0.0

    target_gamma = {}

    for device in raw_gamma:

        target_gamma[device] = (
            raw_gamma[device]
            + offset
        )

        details[
            device
        ][
            "gamma_anchor_offset"
        ] = offset

        details[
            device
        ][
            "gamma_target_aligned"
        ] = target_gamma[
            device
        ]

    return (
        target_gamma,
        details,
    )


# ================================================================
# 10. Parameter definitions
# ================================================================

FACTOR_INFO = {

    "slc": {
        "name":
            "SLC Cache",

        "change_text":
            "+100 GB",
    },

    "fold": {
        "name":
            "Folding Write Speed",

        "change_text":
            "+100 MB/s",
    },

    "gap": {
        "name":
            "TLC-Folding Speed Gap",

        "change_text":
            "+100 MB/s",
    },

    "op": {
        "name":
            "Over-Provisioning (OP)",

        "change_text":
            "+0.01",
    },

    "parallel": {
        "name":
            "Maximum Parallelism",

        "change_text":
            "+32",
    },
}


FACTOR_ORDER = [
    "slc",
    "fold",
    "gap",
    "op",
    "parallel",
]


# ================================================================
# 11. Initial effects
# ================================================================

INITIAL_EFFECTS = {

    "slc": 0.0,

    "fold": 0.0,

    "gap": 0.0,

    "op": 0.0,

    "parallel": 0.0,
}


SEARCH_RANGES = {

    "slc": (
        -0.1000,
        0.1000,
    ),

    "fold": (
        -0.100,
        0.100,
    ),

    # Only constrain Gap
    "gap": (
        -0.100,
        0.100,
    ),

    "op": (
        -0.100,
        0.100,
    ),

    "parallel": (
        -0.3000,
        0.100,
    ),
}

# ================================================================
# 13. TLC-Folding Speed Gap
# ================================================================

def speed_gap(
    p: Dict[str, float]
) -> float:

    return (
        p["tlc_write"]
        -
        p["folding_write"]
    )


# ================================================================
# 14. Normalize architecture changes relative to baseline
# ================================================================

def normalized_changes(
    device: str,
) -> Dict[str, float]:

    p = ARCH_TABLE[
        device
    ]

    base = ARCH_TABLE[
        BASELINE_DEVICE
    ]

    return {

        # +100 GB -> +1
        "slc":
            (
                p["slc_cache"]
                -
                base["slc_cache"]
            )
            / 100.0,

        # +100 MB/s -> +1
        "fold":
            (
                p["folding_write"]
                -
                base["folding_write"]
            )
            / 100.0,

        # +100 MB/s -> +1
        "gap":
            (
                speed_gap(p)
                -
                speed_gap(base)
            )
            / 100.0,

        # +0.01 -> +1
        "op":
            (
                p["OP"]
                -
                base["OP"]
            )
            / 0.01,

        # +32 -> +1
        "parallel":
            (
                p["max_parallelism"]
                -
                base["max_parallelism"]
            )
            / 32.0,
    }



def gamma_from_effects(
    device: str,
    effects: Dict[str, float],
) -> float:

    x = normalized_changes(
        device
    )

    gamma = BASELINE_GAMMA

    for factor in FACTOR_ORDER:

        gamma += (
            effects[factor]
            *
            x[factor]
        )

    return float(
        gamma
    )


# ================================================================
# 16. Candidate generator
# ================================================================

def candidate_values(
    low: float,
    high: float,
    step: float,
) -> np.ndarray:

    count = int(
        round(
            (
                high
                - low
            )
            / step
        )
    )

    values = [

        round(
            low
            + i * step,
            10,
        )

        for i in range(
            count + 1
        )
    ]

    return np.array(
        values,
        dtype=float,
    )



def evaluate_effects(
    effects: Dict[str, float],
    target_gamma: Dict[str, float],
) -> Dict:

    errors = []

    details = []

    for device in ARCH_TABLE:

        if device == BASELINE_DEVICE:
            continue

        gamma_pred = (
            gamma_from_effects(
                device,
                effects,
            )
        )

        gamma_target = (
            target_gamma[
                device
            ]
        )

        error = abs(
            gamma_pred
            -
            gamma_target
        )

        errors.append(
            error
        )

        details.append({

            "Device":
                device,

            "Gamma Target":
                gamma_target,

            "Gamma Predicted":
                gamma_pred,

            "Absolute Error":
                error,

            "Matched":
                (
                    error
                    <=
                    DEVICE_MATCH_TOLERANCE
                ),
        })

    errors_np = np.array(
        errors,
        dtype=float,
    )

    return {

        "median_error":
            float(
                np.median(
                    errors_np
                )
            ),

        "mean_error":
            float(
                np.mean(
                    errors_np
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        errors_np ** 2
                    )
                )
            ),

        "max_error":
            float(
                np.max(
                    errors_np
                )
            ),

        "match_count":
            int(
                np.sum(
                    errors_np
                    <=
                    DEVICE_MATCH_TOLERANCE
                )
            ),

        "n":
            len(
                errors
            ),

        "details":
            details,
    }


# ================================================================
# 18. Scan ONE factor while keeping all other factors fixed
# ================================================================

def scan_one_factor(
    factor: str,
    current_effects: Dict[str, float],
    target_gamma: Dict[str, float],
) -> Dict:

    low, high = (
        SEARCH_RANGES[
            factor
        ]
    )

    candidates = (
        candidate_values(
            low,
            high,
            EFFECT_SEARCH_STEP,
        )
    )

    records = []

    for candidate in candidates:

        # Other four factors remain fixed
        effects = dict(
            current_effects
        )

        # Only current factor changes
        effects[
            factor
        ] = float(
            candidate
        )

        evaluation = (
            evaluate_effects(
                effects,
                target_gamma,
            )
        )

        records.append({

            "candidate":
                float(
                    candidate
                ),

            "mean_error":
                evaluation[
                    "mean_error"
                ],

            "median_error":
                evaluation[
                    "median_error"
                ],

            "rmse":
                evaluation[
                    "rmse"
                ],

            "max_error":
                evaluation[
                    "max_error"
                ],

            "match_count":
                evaluation[
                    "match_count"
                ],

            "n":
                evaluation[
                    "n"
                ],
        })

    # ------------------------------------------------------------
    # Overall error is the primary goal.
    #
    # mean error first,
    # median and max used as tie-breakers.
    # ------------------------------------------------------------

    best_index = min(

        range(
            len(records)
        ),

        key=lambda i: (

            records[i][
                "mean_error"
            ],

            records[i][
                "median_error"
            ],

            records[i][
                "max_error"
            ],
        ),
    )

    best = records[
        best_index
    ]

    boundary_warning = ""

    if best_index == 0:

        boundary_warning = (
            "BEST_AT_LOWER_BOUND"
        )

    elif (
        best_index
        ==
        len(records) - 1
    ):

        boundary_warning = (
            "BEST_AT_UPPER_BOUND"
        )

    return {

        "factor":
            factor,

        "best":
            best[
                "candidate"
            ],

        "mean_error":
            best[
                "mean_error"
            ],

        "median_error":
            best[
                "median_error"
            ],

        "rmse":
            best[
                "rmse"
            ],

        "max_error":
            best[
                "max_error"
            ],

        "match_count":
            best[
                "match_count"
            ],

        "n":
            best[
                "n"
            ],

        "boundary_warning":
            boundary_warning,

        "records":
            records,
    }


# ================================================================
# 19. Iterative one-factor-at-a-time fitting
# ================================================================

def fit_iterative_effects(
    target_gamma: Dict[str, float],
    max_rounds: int = MAX_ITERATIVE_ROUNDS,
    convergence_tol: float = CONVERGENCE_TOLERANCE,
) -> Dict:

    effects = dict(
        INITIAL_EFFECTS
    )

    history = []

    previous_mean_error = None

    for round_index in range(
        1,
        max_rounds + 1,
    ):

        round_changes = {}

        # --------------------------------------------------------
        # Each step changes only ONE factor
        # --------------------------------------------------------

        for factor in FACTOR_ORDER:

            result = scan_one_factor(
                factor,
                effects,
                target_gamma,
            )

            old_value = effects[
                factor
            ]

            new_value = result[
                "best"
            ]

            effects[
                factor
            ] = new_value

            round_changes[
                factor
            ] = {

                "old":
                    old_value,

                "new":
                    new_value,

                "change":
                    new_value
                    - old_value,
            }

        evaluation = (
            evaluate_effects(
                effects,
                target_gamma,
            )
        )

        history.append({

            "round":
                round_index,

            "effects":
                dict(
                    effects
                ),

            "evaluation":
                evaluation,

            "changes":
                round_changes,
        })

        print(
            f"Round {round_index:>2d}: "
            f"mean={evaluation['mean_error']:.6f}, "
            f"median={evaluation['median_error']:.6f}, "
            f"rmse={evaluation['rmse']:.6f}, "
            f"max={evaluation['max_error']:.6f}"
        )

        # --------------------------------------------------------
        # Convergence check
        # --------------------------------------------------------

        if previous_mean_error is not None:

            improvement = (
                previous_mean_error
                -
                evaluation[
                    "mean_error"
                ]
            )

            if (
                abs(
                    improvement
                )
                <
                convergence_tol
            ):

                break

        previous_mean_error = (
            evaluation[
                "mean_error"
            ]
        )

    final_evaluation = (
        evaluate_effects(
            effects,
            target_gamma,
        )
    )

    return {

        "effects":
            effects,

        "evaluation":
            final_evaluation,

        "history":
            history,
    }


def get_final_factor_range(
    factor: str,
    best_effects: Dict[str, float],
    target_gamma: Dict[str, float],
    best_evaluation: Dict,
) -> Dict:

    low, high = (
        SEARCH_RANGES[
            factor
        ]
    )

    candidates = candidate_values(
        low,
        high,
        EFFECT_SEARCH_STEP,
    )

    records = []

    for candidate in candidates:

        # --------------------------------------------------------
        # Keep four other factors fixed at final optimum
        # --------------------------------------------------------

        effects = dict(
            best_effects
        )

        # Only current factor changes
        effects[
            factor
        ] = float(
            candidate
        )

        evaluation = evaluate_effects(
            effects,
            target_gamma,
        )

        records.append({

            "candidate":
                float(
                    candidate
                ),

            "median_error":
                evaluation[
                    "median_error"
                ],

            "mean_error":
                evaluation[
                    "mean_error"
                ],

            "rmse":
                evaluation[
                    "rmse"
                ],

            "max_error":
                evaluation[
                    "max_error"
                ],
        })



    accepted_mask = []

    for r in records:

        accepted = (

            r[
                "median_error"
            ]
            <=
            best_evaluation[
                "median_error"
            ]
            +
            MEDIAN_ERROR_MARGIN

            and

            r[
                "mean_error"
            ]
            <=
            best_evaluation[
                "mean_error"
            ]
            +
            MEAN_ERROR_MARGIN
        )

        accepted_mask.append(
            accepted
        )

    # ------------------------------------------------------------
    # Find candidate closest to final best
    # ------------------------------------------------------------

    best_value = (
        best_effects[
            factor
        ]
    )

    center_index = int(

        np.argmin(

            np.abs(
                candidates
                -
                best_value
            )
        )
    )

    # ------------------------------------------------------------
    # Find contiguous accepted region around best
    # ------------------------------------------------------------

    if accepted_mask[
        center_index
    ]:

        left = center_index

        right = center_index

        while (
            left > 0
            and
            accepted_mask[
                left - 1
            ]
        ):

            left -= 1

        while (
            right
            <
            len(records) - 1

            and

            accepted_mask[
                right + 1
            ]
        ):

            right += 1

        range_low = float(
            records[
                left
            ][
                "candidate"
            ]
        )

        range_high = float(
            records[
                right
            ][
                "candidate"
            ]
        )

    else:

        # Grid resolution fallback
        left = center_index

        right = center_index

        range_low = float(
            best_value
        )

        range_high = float(
            best_value
        )

    # ------------------------------------------------------------
    # Search-bound warning
    # ------------------------------------------------------------

    boundary_warning = ""

    if left == 0:

        boundary_warning = (
            "RANGE_TOUCHES_LOWER_BOUND"
        )

    elif (
        right
        ==
        len(records) - 1
    ):

        boundary_warning = (
            "RANGE_TOUCHES_UPPER_BOUND"
        )

    return {

        "factor":
            factor,

        "best":
            float(
                best_value
            ),

        "range_low":
            range_low,

        "range_high":
            range_high,

        "boundary_warning":
            boundary_warning,

        "records":
            records,
    }


# ================================================================
# 21. Build ranges for all factors
# ================================================================

def get_all_final_factor_ranges(
    best_effects: Dict[str, float],
    target_gamma: Dict[str, float],
    best_evaluation: Dict,
) -> List[Dict]:

    results = []

    for factor in FACTOR_ORDER:

        result = (
            get_final_factor_range(
                factor,
                best_effects,
                target_gamma,
                best_evaluation,
            )
        )

        results.append(
            result
        )

    return results


# ================================================================
# 22. Formatting
# ================================================================

def signed(
    value: float,
    digits: int = 4,
) -> str:

    threshold = (
        0.5
        *
        (
            10
            **
            (-digits)
        )
    )

    if abs(
        value
    ) < threshold:

        value = 0.0

    return (
        f"{value:+.{digits}f}"
    )




# ================================================================
# 24. Save final factor Gamma ranges
# ================================================================
def save_device_gamma_prediction_ranges(
    range_results,
    best_effects,
    path="predicted_gamma_ranges_12_devices.csv",
):

    # 每个因素最终得到的 Gamma effect range
    effect_ranges = {
        result["factor"]: (
            result["range_low"],
            result["range_high"],
        )
        for result in range_results
    }

    fieldnames = [
        "Device",
        "Gamma_Predicted",
        "Gamma_Min",
        "Gamma_Max",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for device in ARCH_TABLE:

            # 不保存 baseline
            if device == BASELINE_DEVICE:
                continue

            # --------------------------------------------
            # 最佳点预测
            # --------------------------------------------
            gamma_pred = gamma_from_effects(
                device,
                best_effects,
            )

            # --------------------------------------------
            # Gamma prediction range
            # --------------------------------------------
            x = normalized_changes(device)

            gamma_min = BASELINE_GAMMA
            gamma_max = BASELINE_GAMMA

            for factor in FACTOR_ORDER:

                effect_low, effect_high = (
                    effect_ranges[factor]
                )

                # 注意 x[factor] 可能为负数
                # 所以乘完以后必须重新判断 min/max
                v1 = x[factor] * effect_low
                v2 = x[factor] * effect_high

                contribution_low = min(v1, v2)
                contribution_high = max(v1, v2)

                gamma_min += contribution_low
                gamma_max += contribution_high

            # Gamma 限制在 [0, 1]
            gamma_min = max(
                0.0,
                min(1.0, gamma_min)
            )

            gamma_max = max(
                0.0,
                min(1.0, gamma_max)
            )

            gamma_pred = max(
                0.0,
                min(1.0, gamma_pred)
            )

            writer.writerow({
                "Device":
                    device,

                "Gamma_Predicted":
                    f"{gamma_pred:.6f}",

                "Gamma_Min":
                    f"{gamma_min:.6f}",

                "Gamma_Max":
                    f"{gamma_max:.6f}",
            })

    print(
        f"Device Gamma prediction ranges saved to: {path}"
    )

def save_final_effects_csv(
    results: List[Dict],
) -> None:

    path = "gamma_effect_range.csv"

    fieldnames = [
        "Factor",
        "Range",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            factor = result["factor"]

            writer.writerow({

                "Factor":
                    FACTOR_INFO[factor]["name"],

                "Range":
                    (
                        f"[{signed(result['range_low'])}, "
                        f"{signed(result['range_high'])}]"
                    ),
            })




# ================================================================
# 26. Validate tables
# ================================================================

def validate_tables() -> None:

    arch_keys = set(
        ARCH_TABLE
    )

    energy_keys = set(
        ENERGY_TABLE
    )

    measured_keys = set(
        MEASURED_POWER_W
    )

    if not (
        arch_keys
        ==
        energy_keys
        ==
        measured_keys
    ):

        raise ValueError(
            "ARCH_TABLE, ENERGY_TABLE and "
            "MEASURED_POWER_W must contain "
            "exactly the same SSD devices."
        )

    if (
        BASELINE_DEVICE
        not in
        ARCH_TABLE
    ):

        raise ValueError(
            f"Baseline device "
            f"{BASELINE_DEVICE} "
            f"is missing."
        )


# ================================================================
# 27. Main
# ================================================================

def main() -> None:

    validate_tables()

    # ============================================================
    # STEP 1
    # Infer Gamma from measured 1-second energy
    # ============================================================

    (
        target_gamma,
        inference_details,
    ) = (
        infer_all_target_gammas()
    )

    print(
        "=" * 100
    )

    print(
        "STEP 1: INFER GAMMA FROM "
        "MEASURED 1-SECOND ENERGY"
    )

    print(
        "=" * 100
    )

    print(

        f"{'Device':<18}"

        f"{'Measured W':>12}"

        f"{'Raw Gamma':>14}"

        f"{'Aligned Gamma':>16}"

        f"{'Status':>22}"
    )

    print(
        "-" * 100
    )

    for device in ARCH_TABLE:

        d = (
            inference_details[
                device
            ]
        )

        print(

            f"{device:<18}"

            f"{d['measured_power_W']:>12.3f}"

            f"{d['gamma_inferred_raw']:>14.6f}"

            f"{target_gamma[device]:>16.6f}"

            f"{d['status']:>22}"
        )

    raw_baseline_gamma = (
        inference_details[
            BASELINE_DEVICE
        ][
            "gamma_inferred_raw"
        ]
    )

    anchor_offset = (
        inference_details[
            BASELINE_DEVICE
        ][
            "gamma_anchor_offset"
        ]
    )

    print(
        "-" * 100
    )

    print(
        f"Raw inferred "
        f"{BASELINE_DEVICE} gamma = "
        f"{raw_baseline_gamma:.6f}"
    )

    print(
        f"Known baseline gamma          = "
        f"{BASELINE_GAMMA:.6f}"
    )

    print(
        f"Applied common anchor offset  = "
        f"{anchor_offset:+.6f}"
    )

    # ============================================================
    # STEP 2
    # Iterative one-factor-at-a-time fitting
    # ============================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STEP 2: ITERATIVE "
        "ONE-FACTOR-AT-A-TIME FITTING"
    )

    print(
        "=" * 100
    )

    fit_result = (
        fit_iterative_effects(
            target_gamma,
            max_rounds=
                MAX_ITERATIVE_ROUNDS,
            convergence_tol=
                CONVERGENCE_TOLERANCE,
        )
    )

    best_effects = (
        fit_result[
            "effects"
        ]
    )

    evaluation = (
        fit_result[
            "evaluation"
        ]
    )

    # ============================================================
    # STEP 3
    # Final factor Gamma-change ranges
    # ============================================================

    range_results = (
        get_all_final_factor_ranges(
            best_effects,
            target_gamma,
            evaluation,
        )
    )
    save_device_gamma_prediction_ranges(
        range_results,
        best_effects,
    )
    print(
        "\n"
        + "=" * 110
    )

    print(
        "STEP 3: FINAL FACTOR "
        "-> GAMMA CHANGE RANGE"
    )

    print(
        "=" * 110
    )

    print(

        f"{'Factor':<32}"

        f"{'Factor Change':<22}"

        f"{'Best Gamma':>14}"

        f"{'Gamma Change Range':>28}"
    )

    print(
        "-" * 110
    )

    for result in range_results:

        factor = (
            result[
                "factor"
            ]
        )

        info = (
            FACTOR_INFO[
                factor
            ]
        )

        range_text = (
            f"["
            f"{signed(result['range_low'])}, "
            f"{signed(result['range_high'])}"
            f"]"
        )

        warning = ""

        if (
            result[
                "boundary_warning"
            ]
        ):

            warning = (
                "  ["
                +
                result[
                    "boundary_warning"
                ]
                +
                "]"
            )

        print(

            f"{info['name']:<32}"

            f"{info['change_text']:<22}"

            f"{signed(result['best']):>14}"

            f"{range_text:>28}"

            f"{warning}"
        )

    # ============================================================
    # STEP 4
    # Final fitting quality
    # ============================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STEP 4: FINAL FITTING QUALITY"
    )

    print(
        "=" * 100
    )

    print(
        f"Median absolute error : "
        f"{evaluation['median_error']:.6f}"
    )

    print(
        f"Mean absolute error   : "
        f"{evaluation['mean_error']:.6f}"
    )

    print(
        f"RMSE                  : "
        f"{evaluation['rmse']:.6f}"
    )

    print(
        f"Maximum error         : "
        f"{evaluation['max_error']:.6f}"
    )

    print(
        f"Matched devices       : "
        f"{evaluation['match_count']}"
        f"/"
        f"{evaluation['n']}"
    )

    # ============================================================
    # STEP 5
    # Device-level comparison
    # ============================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "STEP 5: DEVICE-LEVEL "
        "GAMMA COMPARISON"
    )

    print(
        "=" * 100
    )

    print(

        f"{'Device':<20}"

        f"{'Target':>14}"

        f"{'Predicted':>14}"

        f"{'Abs Error':>14}"

        f"{'Matched':>12}"
    )

    print(
        "-" * 100
    )

    for d in evaluation[
        "details"
    ]:

        print(

            f"{d['Device']:<20}"

            f"{d['Gamma Target']:>14.6f}"

            f"{d['Gamma Predicted']:>14.6f}"

            f"{d['Absolute Error']:>14.6f}"

            f"{str(d['Matched']):>12}"
        )
    save_final_effects_csv(
        range_results
    )
    print(
        "\n"
        + "=" * 100
    )

    print(
        "Saved CSV files:"
    )



# ================================================================
# Run
# ================================================================

if __name__ == "__main__":

    main()
