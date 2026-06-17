STRATEGY_CONFIG = {

    "data": {

        "lookback_years": 4,
        "providers": {
            "equities": "yfinance",
            "etfs": "yfinance",
            "crypto": "binance"   # or "coingecko"
        },     
        "assets": {
            "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"],
            "equities": ["SPY", "QQQ", "IWM"],
            "etfs": ["TLT", "GLD"]
        
        },

        "execution": {
            "cost_bps_retail": 10,
            "cost_bps_institutional": 5
        },

        "benchmark": "SPY"
    },
    "strategies": {

"IntradayStrategy": {
    "enabled": True,
    "lookback": 20,
    "volume_window": 20,
    "vol_window": 20,
    "signal_threshold": 1.5,
    "plot_enabled": True,
    "tab": "strategies",
    "title": "Intraday Liquidity Dislocations",

    "chart": {
        "type": "line",
        "mode": "multi",

        "title": "Intraday Liquidity Dislocation Intensity (Rolling)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Normalized Signal",
            "scale": "linear"
        },

        "series": [
            {
                "name": "Intraday Signal",
                "source": "signal",
                "style": "line"
            },
            {
                "name": "Volume Stress",
                "source": "volume_stress",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": True,
            "source": "dislocation_events",
            "style": "circle"
        },

        "normalize": True,
        "benchmark": None
    }
},
      
      "AlphaStrategy" : {"enabled": True, 
    "lookback": 252,
    "rebalance": 21,
    "top_quantile": 0.2,

    "chart": {
        "type": "bar",

        "series": [
            {
                "key": "price",
                "source": "close",
                "label": "Price"
            },
            {
                "key": "signal",
                "source": "signal",
                "label": "Alpha Signal"
            },
            {
                "key": "benchmark",
                "source": "SPY",
                "label": "Benchmark"
            }
        ],
        "series": {
  "key": "signal",
  "source": "signal",
  "label": "Alpha Signal"
},
        "axes": {
            "x": "date",
            "y": "normalized_value"
        },

        "title": "Cross-sectional Alpha (4Y)"
    }
},      
"BetaNeutralStrategy": {
    "enabled": True,
    "lookback": 252,
    "beta_window": 63,
    "rebalance": 21,
    "target_beta": 0.0,

    "plot_enabled": True,
    "tab": "strategies",
    "title": "Beta-neutral portfolio performance",

    "chart": {
        "type": "line",

        "title": "Beta Neutral Strategy (4Y Performance)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Normalized Value",
            "scale": "linear"
        },

        "series": [
            {
                "name": "Beta-Neutral PnL",
                "source": "pnl",
                "style": "line"
            },
            {
                "name": "SPY Benchmark",
                "source": "SPY",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": False,
            "source": "rebalance_events",
            "style": "circle"
        },

        "normalize": True,
        "benchmark": "SPY"
    }
},
"UMDMomentum": {
    "enabled": True,
    "formation": 252,
    "skip_month": 21,
    "holding": 21,
    "top_quantile": 0.2,
    "bottom_quantile": 0.2,
    "plot_enabled": True,
    "tab": "momentum",
    "title": "Cross-Sectional Momentum (UMD)",

    "chart": {
        "type": "time_series_multi",

        "title": "UMD Momentum (Cross-Sectional 4Y Equity Curve)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Cumulative Return",
            "scale": "linear"
        },

        "series": [
            {
                "name": "UMD Portfolio",
                "source": "portfolio",
                "style": "line"
            },
            {
                "name": "SPY Benchmark",
                "source": "benchmark",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": False
        },

        "normalize": False,
        "benchmark": "SPY"
    }
},
 "TimeSeriesMomentum": {
    "enabled": True,
    "formation": 252,
    "vol_target": 0.15,
    "lookback_vol": 63,
    "plot_enabled": True,
    "tab": "momentum",
    "title": "Trend Persistence Capture",

    "chart": {
        "type": "time_series",

        "title": "Time Series Momentum (Trend Following, 4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Signal Strength (Risk Adjusted)",
            "scale": "linear"
        },

        "series": [
            {
                "name": "Trend Signal",
                "source": "signal",
                "style": "line"
            },
            {
                "name": "SPY Benchmark",
                "source": "benchmark",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": True,
            "source": "regime_switches",
            "style": "circle"
        },

        "normalize": False,
        "benchmark": "SPY"
    }
},

"STREV": {
    "enabled": True,
    "lookback": 20,
    "holding": 5,
    "zscore_window": 60,

    "plot_enabled": True,
    "tab": "reversal",

    "title": "STREV Short-Term Mean Reversion",

    "chart": {
        "type": "time_series",

        "mode": "line+markers",

        "title": "STREV Mean Reversion Strategy (4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Portfolio Value",
            "scale": "linear"
        },

        "series": [
            {
                "name": "STREV Portfolio",
                "source": "portfolio_curve",
                "style": "line"
            },
            {
                "name": "Signal Strength",
                "source": "signal_curve",
                "style": "line"
            },
            {
                "name": "SPY Benchmark",
                "source": "benchmark",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": True,
            "source": "entry_exit_events",
            "style": "circle"
        },

        "normalize": True,
        "benchmark": "SPY"
    }
},
  "IntradayReversal": {

    "enabled": True,

    "lookback": 5,
    "volume_window": 20,
    "threshold": 2.0,

    "plot_enabled": True,
    "tab": "reversal",
    "title": "Intraday Overreaction Reversal",
    "description": "Volatility and volume intraday reversal",

    "merge_mode": "overlay",
    "output_mode": "merged",

    "models": {
      "volatility": {
        "enabled": True
      },
      "volume": {
        "enabled": True
      }
    },

    "chart": [
      {
        "name": "volatility",
        "enabled": True,

        "title": "Volatility Reversal Signal",

        "type": "line",

        "xaxis": {
          "source": "date",
          "type": "datetime"
        },

        "yaxis": {
          "label": "Z-Score (Volatility Model)",
          "scale": "linear"
        },

        "series": [
          {
            "name": "Z-Score (Volatility)",
            "source": "z_vol",
            "style": "line"
          },
          {
            "name": "Rolling Volatility",
            "source": "volatility",
            "style": "line"
          }
        ],

        "markers": {
          "enabled": True,
          "source": "reversal_event_vol",
          "style": "triangle-down"
        },

        "normalize": True
      },

      {
        "name": "volume",
        "enabled": True,

        "title": "Volume Reversal Signal",

        "type": "line",

        "xaxis": {
          "source": "date",
          "type": "datetime"
        },

        "yaxis": {
          "label": "Z-Score (Volume Model)",
          "scale": "linear"
        },

        "series": [
          {
            "name": "Z-Score (Volume)",
            "source": "z_volume",
            "style": "line"
          },
          {
            "name": "Rolling Volatility",
            "source": "volatility",
            "style": "line"
          }
        ],

        "markers": {
          "enabled": True,
          "source": "reversal_event_volume",
          "style": "triangle-down"
        },

        "normalize": False
      }
    ]
  },

   "PairTrading": {
    "enabled": True,
    "lookback": 252,
    "entry_zscore": 2.0,
    "exit_zscore": 0.5,
    "max_pairs": 10,
    "min_half_life": 5,
    "plot_enabled": True,
    "tab": "strategies",
    "title": "Pairs Trading - Cointegration Spread Reversion",

    "chart": {
        "type": "multi_line",

        "title": "Pairs Trading Spread & Z-Score Dynamics (4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Value",
            "scale": "linear"
        },

        "series": [
            {
                "name": "spread",
                "source": "signals.spread",
                "style": "line",
                "grouped": True
            },
            {
                "name": "signal",
                "source": "signals.signal",
                "style": "step",
                "grouped": True
            }
        ],

        "normalize": False,
        "benchmark": None,

        "legend": {
            "enabled": True,
            "group_by": "pair"
        }
    }
},

  "BreadthStrategy": {
    "enabled": True,
    "lookback": 20,
    "ma_short": 20,
    "ma_long": 50,

    "plot_enabled": True,
    "tab": "strategies",
    "title": "Breadth-Market Participation Strength",

    "chart": {
      "type": "line",
      "title": "Market Breadth & Participation Strength",

      "xaxis": {
        "source": "index",
        "type": "datetime"
      },

      "yaxis": {
        "label": "Breadth Index (0–1)",
        "scale": "linear"
      },

      "series": [
        {
          "name": "Breadth",
          "source": "breadth",
          "style": "line"
        },
        {
          "name": "MA Short (20)",
          "source": "ma_short",
          "style": "line"
        },
        {
          "name": "MA Long (50)",
          "source": "ma_long",
          "style": "line"
        },
        {
          "name": "Breadth Spread",
          "source": "breadth_spread",
          "style": "line"
        }
      ],

      "normalize": True,
      "benchmark": "SPY"
    }
  }
,

"CorrelationStrategy": {
    "enabled": True,

    "lookback": 252,
    "corr_window": 63,
    "signal_threshold": 0.7,

    "plot_enabled": True,
    "tab": "strategies",
    "title": "Regime correlation shifts",

    "chart": {
        "type": "line",
        "title": "Market Correlation Regimes (4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Correlation Level",
            "scale": "linear"
        },

        "series": [
            {
                "name": "Avg Market Correlation",
                "source": "avg_correlation_series",
                "style": "line"
            },
            {
                "name": "Signal Threshold",
                "source": "signal_threshold_series",
                "style": "line"
            }
        ],

        "normalize": False,
        "benchmark": {}
    }
},

 "DispersionStrategy": {
    "enabled": True,
    "lookback": 63,
    "cross_sectional_window": 63,
    "plot_enabled": True,
    "tab": "strategies",
    "title": "Cross-sectional opportunity dispersion",

    "chart": {
        "type": "line",

        "title": "Market Dispersion Regimes (4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Dispersion Level",
            "scale": "linear"
        },

        "series": [
            {
                "name": "Cross-sectional Dispersion",
                "source": "dispersion",
                "style": "line"
            },
            {
                "name": "MA (63D)",
                "source": "ma_63",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": True,
            "source": "regime_switches",
            "style": "circle"
        },

        "normalize": True,
        "benchmark": {}
    }
},

  "VolatilityStrategy": {
    "enabled": True,
    "lookback": 63,
    "vol_window": 21,
    "target_vol": 0.15,
    "plot_enabled": True,
    "tab": "strategies",
    "title": "Volatility Adaptive Risk Targeting",

    "chart": {
        "type": "line",
        "title": "Volatility Targeting & Risk Stability (4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Volatility Level",
            "scale": "linear"
        },

        "series": [
            {
                "name": "Realized Volatility",
                "source": "volatility",
                "style": "line"
            },
            {
                "name": "Target Volatility",
                "source": "target_vol",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": False,
            "source": "vol_regime_shifts",
            "style": "circle"
        },

        "normalize": False,
        "benchmark": {}
    }
},

  "ForecastStrategy": {
    "enabled": True,
    "lookback": 252,
    "forecast_horizon": 21,
    "train_window": 504,
    "plot_enabled": True,
    "tab": "strategies",
    "title": "Forecast Predictive return modeling",

    "chart": {
        "type": "line",

        "title": "Forecast Model Performance (4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Return / Prediction Value",
            "scale": "linear"
        },

        "series": [
            {
                "name": "Predicted Return",
                "source": "forecast",
                "style": "line"
            },
            {
                "name": "Realized Return",
                "source": "actual",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": False,
            "source": "forecast_events",
            "style": "circle"
        },

        "normalize": False,
        "benchmark": "SPY"
    }
},"IndustryMomentumStrategy": {
    "enabled": True,

    "formation": 252,
    "holding": 21,
    "industry_window": 252,
    "top_quantile": 0.30,

    "plot_enabled": True,
    "tab": "momentum",

    "title": "Industry Momentum - Sector Leadership Persistence",

    "chart": {

        "type": "line",

        "mode": "overlay",

        "title": "Industry Momentum vs Stock Momentum (4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Normalized Performance (Base = 1.0)",
            "scale": "linear"
        },

        "series": [

            {
                "name": "Industry Momentum Portfolio",
                "source": "portfolio_momentum",
                "style": "line"
            },

            {
                "name": "Industry Strength Index",
                "source": "industry_strength",
                "style": "line"
            },

            {
                "name": "Smoothed Momentum",
                "source": "momentum_ma",
                "style": "line"
            },

            {
                "name": "Stock Momentum Portfolio",
                "source": "stock_momentum",
                "style": "line"
            },

            {
                "name": "SPY Benchmark",
                "source": "benchmark",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": True,
            "source": "rebalance_events",
            "style": "circle"
        },

        "normalize": True,

        "benchmark": "SPY"
    }
},
"IndustryMomentumStrategyBase": {
    "enabled": False,

    "formation": 252,
    "holding": 21,
    "industry_window": 252,
    "top_quantile": 0.30,

    "plot_enabled": True,
    "tab": "momentum",

    "title": "Industry Momentum - Sector Leadership Persistence",

    "chart": {

        "type": "line",
        "mode": "overlay",

        "title": "Industry Momentum (Sector Leadership, 4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Normalized Performance",
            "scale": "linear"
        },

        "series": [

            {
                "name": "Industry Momentum Portfolio",
                "source": "portfolio_momentum",
                "style": "line"
            },

            {
                "name": "Industry Strength",
                "source": "industry_strength",
                "style": "line"
            },

            {
                "name": "Momentum MA",
                "source": "momentum_ma",
                "style": "line"
            },

            {
                "name": "SPY Benchmark",
                "source": "benchmark",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": True,
            "source": "rebalance_events",
            "style": "circle"
        },

        "normalize": True,
        "benchmark": "SPY"
    }
},
"PairTradingFallback": {
    "enabled": True,

    "lookback": 126,
    "corr_window": 63,
    "min_corr": 0.6,

    "entry_zscore": 1.5,
    "exit_zscore": 0.3,

    "max_pairs": 5,
    "rebalance": 21,

    "use_spread_returns": True,
    "vol_filter": 0.2,

    "plot_enabled": True,
    "chart_enabled": True,

    "tab": "strategies",
    "title": "Pair Trading Fallback - Correlation-based proxy convergence",

    "chart": {
        "type": "line",
        "title": "Pair Trading Fallback (Correlation Proxy, 4Y)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Spread Proxy / Z-Score",
            "scale": "linear"
        },

        "series": [
            {
                "name": "Correlation-Based Spread Proxy",
                "source": "spread_proxy",
                "style": "line"
            },
            {
                "name": "Z-Score",
                "source": "zscore",
                "style": "line"
            },
            {
                "name": "Correlation Regime",
                "source": "correlation",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": True,
            "source": "trade_events",
            "style": "circle"
        },

        "normalize": False,
        "benchmark": None
    }
},

  "CorrelationFallback": {
    "enabled": True,

    "lookback": 42,
    "corr_window": 42,

    "signal_threshold": 0.6,
    "low_corr_threshold": 0.3,

    "dispersion_window": 21,

    "risk_on_when_corr_low": True,
    "risk_off_when_corr_high": True,

    "signal_smooth": 5,

    "plot_enabled": True,
    "tab": "strategies",
    "title": "Correlation Fallback Robust Regime Detection",

    "chart": {
      "type": "line",
      "mode": "overlay",
      "title": "Correlation Regime Instability Detector (4Y)",

      "xaxis": {
        "source": "date",
        "type": "datetime"
      },

      "yaxis": {
        "label": "Regime Score",
        "scale": "linear"
      },

      "series": [
        {
          "name": "Smoothed Correlation Signal",
          "source": "signal",
          "style": "line"
        },
        {
          "name": "Rolling Correlation",
          "source": "correlation",
          "style": "line"
        },
        {
          "name": "Dispersion",
          "source": "dispersion",
          "style": "line"
        },
        {
          "name": "High Correlation Threshold",
          "source": "threshold_high",
          "style": "line"
        },
        {
          "name": "Low Correlation Threshold",
          "source": "threshold_low",
          "style": "line"
        },
        {
          "name": "Equity Curve",
          "source": "equity_curve",
          "style": "line"
        }
      ],

      "markers": {
        "enabled": False,
        "source": "regime_switches",
        "style": "circle"
      },

      "normalize": False,
      "benchmark": {}
    }
  },"PortfolioConstruction": {

    "enabled": False,

    "formation": 252,
    "rebalance": 21,

    "benchmark": "SPY",

    "plot_enabled": True,
    "tab": "portfolio",
    "title": "Portfolio Construction Engine",
    "description": "Multi-model portfolio optimization framework",

    "merge_mode": "overlay",
    "output_mode": "merged",

    "models": {

        "risk_parity": { "enabled": True },
        "minimum_variance": { "enabled": True },
        "maximum_sharpe": { "enabled": True },
        "mean_variance": { "enabled": True },
        "black_litterman": { "enabled": True },
        "factor_portfolio": { "enabled": True },
        "quant_finance": { "enabled": True }
    },

    "chart": [

        {
            "name": "risk_parity",
            "enabled": True,
            "title": "Risk Parity Portfolio",
            "type": "line",

            "xaxis": {
                "source": "date",
                "type": "datetime"
            },

            "yaxis": {
                "label": "Cumulative Return",
                "scale": "linear"
            },

            "series": [
                {
                    "name": "Risk Parity",
                    "source": "risk_parity",
                    "style": "line"
                }
            ],

            "markers": {
                "enabled": False
            },

            "normalize": False
        },

        {
            "name": "minimum_variance",
            "enabled": True,
            "title": "Minimum Variance Portfolio",
            "type": "line",

            "xaxis": {
                "source": "date",
                "type": "datetime"
            },

            "yaxis": {
                "label": "Cumulative Return",
                "scale": "linear"
            },

            "series": [
                {
                    "name": "Min Variance",
                    "source": "minimum_variance",
                    "style": "line"
                }
            ],

            "markers": {
                "enabled": False
            },

            "normalize": False
        },

        {
            "name": "maximum_sharpe",
            "enabled": True,
            "title": "Maximum Sharpe Portfolio",

            "type": "line",

            "xaxis": {
                "source": "date",
                "type": "datetime"
            },

            "yaxis": {
                "label": "Cumulative Return",
                "scale": "linear"
            },

            "series": [
                {
                    "name": "Max Sharpe",
                    "source": "maximum_sharpe",
                    "style": "line"
                }
            ]
        },

        {
            "name": "benchmark",
            "enabled": True,
            "title": "Benchmark",

            "type": "line",

            "series": [
                {
                    "name": "SPY",
                    "source": "benchmark",
                    "style": "line"
                }
            ]
        }
    ]
},
     "InstitutionEngine": {
    "enabled": False,
    "plot_enabled": True,
    "tab": "institution",
    "title": "Institution Engine Multi-factor signal fusion",

    "chart": {
        "type": "line",

        "title": "Institutional Composite Signal (4Y Decision Layer)",

        "xaxis": {
            "source": "date",
            "type": "datetime"
        },

        "yaxis": {
            "label": "Composite Signal Strength",
            "scale": "linear"
        },

        "series": [
            {
                "name": "Institutional Signal",
                "source": "composite_signal",
                "style": "line"
            },
            {
                "name": "Risk-Adjusted Signal",
                "source": "risk_adjusted_signal",
                "style": "line"
            }
        ],

        "markers": {
            "enabled": False,
            "source": "allocation_decisions",
            "style": "circle"
        },

        "normalize": True,
        "benchmark": "SPY"
    }
}
}   
}