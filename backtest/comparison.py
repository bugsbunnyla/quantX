class Compare:

  def compare(self, raw, opt):
    def safe_sharpe(x):
        std = x.std()
        return 0 if (std == 0 or np.isnan(std)) else x.mean() / std

    return {
        "raw_sharpe": safe_sharpe(raw),
        "opt_sharpe": safe_sharpe(opt)
    }