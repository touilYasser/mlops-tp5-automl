from pathlib import Path

import numpy as np
import pandas as pd


def build_transactions(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    montant = np.round(rng.lognormal(mean=4.2, sigma=1.0, size=n_rows), 2)
    anciennete = rng.integers(0, 15, size=n_rows)
    freq_24h = rng.poisson(2.2, size=n_rows)
    nuit = rng.binomial(1, 0.22, size=n_rows)
    score = (
        0.00055 * montant
        + 0.32 * freq_24h
        + 1.15 * nuit
        - 0.10 * anciennete
        + rng.normal(0, 0.9, size=n_rows)
    )
    fraude = (score > np.quantile(score, 0.84)).astype(int)
    return pd.DataFrame(
        {
            "montant": montant,
            "anciennete": anciennete,
            "freq_24h": freq_24h,
            "nuit": nuit,
            "fraude": fraude,
        }
    )


if __name__ == "__main__":
    output = Path("data/transactions.csv")
    output.parent.mkdir(exist_ok=True)
    build_transactions().to_csv(output, index=False)
    print(f"Donnees generees : {output} ({len(build_transactions())} lignes)")
