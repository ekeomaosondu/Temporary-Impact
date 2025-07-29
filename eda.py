import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional

class MarketImpactAnalyzer:

    def __init__(self, data_dir, tickers=None, depth=5):
        self.data_dir = Path(data_dir)
        self.tickers = tickers or ['CRWV', 'FROG', 'SOUN']
        self.depth = depth
        self.data = {}
        self.load_data()

    def load_data(self) -> None:
        for ticker in self.tickers:
            ticker_dir = self.data_dir / ticker
            csv_files = sorted(ticker_dir.glob("*.csv"))
        
            frames = []
            for fp in csv_files:
                df = pd.read_csv(fp)
                frames.append(df)

            df_all = pd.concat(frames, ignore_index=True)

            if 'ts_event' not in df_all.columns:
                raise KeyError(f"'ts_event' column not found in {ticker} data.")

            df_all = df_all.sort_values('ts_event').reset_index(drop=True)
            self.data[ticker] = df_all

    @staticmethod
    def _get_levels(snapshot: pd.Series, side: str, depth: int) -> Tuple[np.ndarray, np.ndarray]:
        px_list, sz_list = [], []
        for i in range(depth):
            px_col = f'{side}_px_{i:02d}'
            sz_col = f'{side}_sz_{i:02d}'
            if px_col in snapshot.index and sz_col in snapshot.index:
                px = snapshot[px_col]
                sz = snapshot[sz_col]
                if np.isfinite(px) and np.isfinite(sz) and sz > 0:
                    px_list.append(px)
                    sz_list.append(sz)
        return np.array(px_list, dtype=float), np.array(sz_list, dtype=float)

    @staticmethod
    def _vwap_execution(order_size: float, prices: np.ndarray, sizes: np.ndarray) -> float:
        take = np.minimum(order_size, sizes)
        vwap = (prices * take).sum() / take.sum()
        return vwap

    def calculate_slippage(
        self,
        order_size: float,
        snapshot: pd.Series,
        side: str = 'buy'
    ) -> float:
        bids_px, bids_sz = self._get_levels(snapshot, 'bid', self.depth)
        asks_px, asks_sz = self._get_levels(snapshot, 'ask', self.depth)

        if len(bids_px) == 0 and len(asks_px) == 0:
            return 0.0

        best_bid = bids_px[0] if len(bids_px) else np.nan
        best_ask = asks_px[0] if len(asks_px) else np.nan
        if np.isnan(best_bid) or np.isnan(best_ask):
            mid = np.nanmean([best_bid, best_ask])
        else:
            mid = (best_bid + best_ask) / 2

        if not np.isfinite(mid) or mid <= 0 or order_size <= 0:
            return 0.0

        if side.lower() == 'buy':
            if len(asks_px) == 0:
                return 0.0
            vwap = self._vwap_execution(order_size, asks_px, asks_sz)
            slip = (vwap - mid) / mid
        else:
            if len(bids_px) == 0:
                return 0.0
            vwap = self._vwap_execution(order_size, bids_px, bids_sz)
            slip = (mid - vwap) / mid

        return float(max(slip, 0))

    def analyze_temporary_impact(
        self,
        order_sizes=None,
        n_snapshots=1000,
        side='buy',
        plot=True,
        out_dir=None
    ):
        if order_sizes is None:
            order_sizes = np.linspace(1, 1000, 100)

        results = {}
        for ticker in self.tickers:
            df = self.data[ticker]

            if len(df) > n_snapshots:
                snap_idx = np.linspace(0, len(df)-1, n_snapshots, dtype=int)
                sampled = df.iloc[snap_idx]
            else:
                sampled = df

            slips = []
            for size in order_sizes:
                vals = [
                    self.calculate_slippage(size, row, side=side)
                    for _, row in sampled.iterrows()
                ]
                slips.append(np.nanmean(vals))

            slips = np.array(slips)
            results[ticker] = {'order_sizes': order_sizes, 'slippages': slips}

            if plot:
                plt.figure(figsize=(8, 5))
                plt.plot(order_sizes, slips)
                plt.xlabel('Order Size')
                plt.ylabel('Avg Slippage (fraction of mid)')
                plt.title(f'{ticker} Temporary Impact ({side} side)')
                plt.tight_layout()
                if out_dir:
                    out_dir = Path(out_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    plt.savefig(out_dir / f'{ticker}_impact.png', dpi=150)
                    plt.close()
                else:
                    plt.show()

        return results

    @staticmethod
    def fit_impact_model(order_sizes: np.ndarray, slippages: np.ndarray) -> Tuple[float, float]:
        def power_law(x, a, b):
            return a * (x ** b)

        mask = (order_sizes > 0) & np.isfinite(slippages)
        popt, _ = curve_fit(power_law, order_sizes[mask], slippages[mask], p0=[1e-6, 0.5])
        return tuple(popt)

    @staticmethod
    def optimize_trading_schedule(total_shares: float, N: int = 390) -> np.ndarray:
        try:
            from cvxopt import matrix, solvers
            solvers.options['show_progress'] = False

            P = matrix(np.eye(N))
            q = matrix(np.zeros(N))
            G = matrix(-np.eye(N))
            h = matrix(np.zeros(N))
            A = matrix(np.ones((1, N)))
            b = matrix([total_shares], (1, 1))

            sol = solvers.qp(P, q, G, h, A, b)
            x = np.array(sol['x']).flatten()
            return x
        except Exception:
            return np.full(N, total_shares / N)

def main():
    data_dir = '/Users/ekeomaosondu/Desktop/Temporary Impact/Temporary-Impact/Data'
    out_dir = Path('./figs')
    analyzer = MarketImpactAnalyzer(data_dir)

    results = analyzer.analyze_temporary_impact(
        order_sizes=np.linspace(10, 5000, 50),
        n_snapshots=500,
        side='buy',
        plot=True,
        out_dir=out_dir
    )

if __name__ == '__main__':
    main()
