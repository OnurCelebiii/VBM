"""
generate_data.py
----------------
Realistic GNSS measurement simulation for VBM604 Statistical Data Analysis.

Statistical characteristics based on:
  - Google Smartphone Decimeter Challenge (ION GNSS+ 2021-2023)
  - ITU-R P.1546 signal propagation model

Target summary statistics (reproduced from README):
  Overall mean  : 2.91 m     Std : 1.96 m
  Open Sky mean : 1.40 m     Urban Canyon mean : 4.77 m
  Skewness : +0.36

Produces: data/gnss_measurements.csv  (3000 rows, 8 columns)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

N = 3000

# ── Sampling weights ──────────────────────────────────────────────────────────
# 55/45 split gives overall mean ≈ 0.55*1.40 + 0.45*4.77 = 2.92 m ≈ 2.91 m
CONSTELLATIONS = ["GPS", "Galileo", "GLONASS", "BeiDou"]
CON_WEIGHTS    = [0.35, 0.25, 0.20, 0.20]

ENVIRONMENTS   = ["Open Sky", "Urban Canyon"]
ENV_WEIGHTS    = [0.55, 0.45]

# ── Constellation-level error offsets (metres) ────────────────────────────────
# Galileo < GPS < BeiDou < GLONASS  (matches README ranking)
CONSTELLATION_OFFSET = {
    "GPS":      0.00,
    "Galileo": -0.37,
    "GLONASS":  0.62,
    "BeiDou":   0.16,
}

# ── Error model calibration constants ─────────────────────────────────────────
# Derived analytically so that:
#   E[error | Open Sky , GPS] ≈ 1.40 m
#   E[error | Urban Canyon, GPS] ≈ 4.77 m
BASE_INTERCEPT   = 0.823   # overall intercept
BETA_PDOP        = 0.40    # positive: higher PDOP → more error
BETA_CN0         = 0.025   # positive coeff on (cn0 - 38); applied as -BETA_CN0
BETA_NSATS       = 0.060   # positive coeff on (nsats - 9); applied as -BETA_NSATS
BETA_MULTI       = 0.25    # positive: higher multipath → more error
URBAN_OFFSET     = 2.077   # partial direct urban-canyon effect
NOISE_STD        = 0.90    # residual Gaussian noise (σ); tuned for std ≈ 1.96 m

# ── Physical signal parameter samplers ───────────────────────────────────────

def sample_pdop(env: str, n: int) -> np.ndarray:
    """Log-normal PDOP.  Open Sky: mean≈1.87  Urban: mean≈3.71."""
    if env == "Open Sky":
        # log-normal: μ_log=0.58, σ_log=0.30 → mean≈1.87
        vals = rng.lognormal(mean=0.58, sigma=0.30, size=n)
        return np.clip(vals, 1.0, 5.0)
    else:
        # log-normal: μ_log=1.25, σ_log=0.35 → mean≈3.71
        vals = rng.lognormal(mean=1.25, sigma=0.35, size=n)
        return np.clip(vals, 1.5, 8.0)


def sample_cn0(env: str, n: int) -> np.ndarray:
    """Carrier-to-noise ratio [dB-Hz]."""
    if env == "Open Sky":
        return np.clip(rng.normal(42.0, 4.0, size=n), 25.0, 55.0)
    else:
        return np.clip(rng.normal(35.0, 5.0, size=n), 18.0, 50.0)


def sample_num_satellites(env: str, n: int) -> np.ndarray:
    """Visible satellite count."""
    if env == "Open Sky":
        vals = rng.normal(11.0, 2.0, size=n).round().astype(int)
        return np.clip(vals, 5, 18)
    else:
        vals = rng.normal(7.0, 2.0, size=n).round().astype(int)
        return np.clip(vals, 4, 14)


def sample_elevation(env: str, n: int) -> np.ndarray:
    """Satellite elevation angle [deg]."""
    if env == "Open Sky":
        return np.clip(rng.normal(45.0, 15.0, size=n), 5.0, 85.0)
    else:
        return np.clip(rng.normal(28.0, 12.0, size=n), 5.0, 70.0)


def sample_multipath(env: str, n: int) -> np.ndarray:
    """Multipath index [0, 1]."""
    if env == "Open Sky":
        return np.round(rng.beta(1.5, 8.0, size=n), 4)   # mean ≈ 0.16
    else:
        return np.round(rng.beta(5.0, 3.0, size=n), 4)    # mean ≈ 0.63


# ── Position error model ──────────────────────────────────────────────────────

def compute_position_error(
    pdop:      np.ndarray,
    cn0:       np.ndarray,
    nsats:     np.ndarray,
    multipath: np.ndarray,
    constellation: str,
    environment:   str,
    n:             int,
) -> np.ndarray:
    """
    Calibrated linear model:

        error = BASE_INTERCEPT
              + BETA_PDOP  * pdop
              - BETA_CN0   * (cn0  - 38)
              - BETA_NSATS * (nsats - 9)
              + BETA_MULTI * multipath
              + URBAN_OFFSET  [if Urban Canyon]
              + CONSTELLATION_OFFSET[constellation]
              + Normal(0, NOISE_STD)

    Calibration ensures:
        E[error | Open Sky,    GPS] ≈ 1.40 m
        E[error | Urban Canyon, GPS] ≈ 4.77 m
        Overall E[error] ≈ 2.91 m  (at 55/45 env split)
    """
    env_offset = URBAN_OFFSET if environment == "Urban Canyon" else 0.0
    con_offset = CONSTELLATION_OFFSET[constellation]
    noise      = rng.normal(0, NOISE_STD, size=n)

    error = (
        BASE_INTERCEPT
        + BETA_PDOP  * pdop
        - BETA_CN0   * (cn0  - 38)
        - BETA_NSATS * (nsats.astype(float) - 9)
        + BETA_MULTI * multipath
        + env_offset
        + con_offset
        + noise
    )
    return np.clip(error, 0.10, 15.0)


# ── Build dataset ─────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    con_draws = rng.choice(CONSTELLATIONS, size=N, p=CON_WEIGHTS)
    env_draws = rng.choice(ENVIRONMENTS,   size=N, p=ENV_WEIGHTS)

    records = []
    for con, env in zip(con_draws, env_draws):
        pdop   = float(sample_pdop(env, 1)[0])
        cn0    = float(sample_cn0(env, 1)[0])
        nsats  = int(sample_num_satellites(env, 1)[0])
        elev   = float(sample_elevation(env, 1)[0])
        multi  = float(sample_multipath(env, 1)[0])
        error  = float(compute_position_error(
            np.array([pdop]),
            np.array([cn0]),
            np.array([nsats]),
            np.array([multi]),
            con, env, 1,
        )[0])

        records.append({
            "constellation":    con,
            "environment":      env,
            "elevation_deg":    round(elev,  2),
            "cn0_dbhz":         round(cn0,   2),
            "num_satellites":   nsats,
            "pdop":             round(pdop,  3),
            "position_error_m": round(error, 4),
            "multipath_index":  multi,
        })

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating GNSS measurement dataset …")
    df = build_dataset()

    out_path = Path("data") / "gnss_measurements.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows → {out_path}\n")

    err = df["position_error_m"]
    print("position_error_m — summary statistics:")
    print(err.describe().to_string())
    print(f"Skewness : {err.skew():.4f}")
    print(f"Kurtosis : {err.kurt():.4f}")

    print("\nMean error by environment:")
    print(df.groupby("environment")["position_error_m"].mean().to_string())

    print("\nMean error by constellation:")
    print(df.groupby("constellation")["position_error_m"].mean().to_string())


if __name__ == "__main__":
    main()
