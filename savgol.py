import numpy as np
from scipy.signal import savgol_filter
from statsmodels.tsa.statespace.tools import diff

# I wrote how smoothing done in DySMHO (based on inaccurate understanding of the paper)
def iterative_savgol_filter(sig, ws=3, phi=2, delta=11, thres=0.5, verbose=True):
    min_val = sig.min()
    max_val = sig.max()
    norm_sig = (sig - min_val) / (max_val - min_val)

    sig_now = savgol_filter(norm_sig, window_length=ws, polyorder=phi)
    sigma_now = (sig_now - sig).std()
    ws = ws + delta

    while 1:
        sig_new = savgol_filter(norm_sig, window_length=ws, polyorder=phi)
        sigma_new = (sig_new - sig_now).std()
        measure = sigma_new / sigma_now
        if verbose:
            print(measure)
        if measure < thres:
            break
        else:
            ws = ws + delta
            sig_now = sig_new
            sigma_now = sigma_new

    if verbose:
        print(ws)
    return (max_val - min_val) * sig_new + min_val


def iterative_savgol_gpt(
    data, window_size, poly_order, delta, max_window_size, threshold, verbose=True
):
    prev_smoothed = data.copy()
    smoothed = savgol_filter(data, window_size, poly_order)
    mean_diff = ((smoothed - prev_smoothed) ** 2).mean()

    if mean_diff <= 0:
        print("Increase window_size to get iterative denoising effects")
        assert mean_diff > 0

    if verbose:
        print(window_size, mean_diff)
    while mean_diff > threshold:
        next_window_size = window_size + delta
        if next_window_size > max_window_size:
            break
        prev_smoothed = smoothed.copy()
        smoothed = savgol_filter(data, next_window_size, poly_order)
        next_mean_diff = ((smoothed - prev_smoothed) ** 2).mean()
        if next_mean_diff > mean_diff:
            break
        else:
            window_size = next_window_size
            mean_diff = next_mean_diff
        if verbose:
            print(window_size, mean_diff)

    return smoothed


class DySMHO:
    def __init__(self, y, div=4):
        self.y = y # of dimension (len(x), len(t))
        self.div = div
        if len(y.shape) > 1:
            self.max_window_size = y.shape[-1]//self.div+1
        else:
            self.max_window_size = len(y)//self.div+1
        print("Max windown size:", self.max_window_size)
        self.best_window_size = None

    """
    Smoothing function applies the Savitzky-Golay filter to the state measurements
        Inputs:
        - window_size: (interger) The length of the filter window (i.e., the number of coefficients)
        - poly_order: (interger) The order of the polynomial used to fit the samples
        - verbose: (True/False) display information regarding smoothing iterations
    """

    def smooth(self, window_size=1, poly_order=2, delta=10, verbose=True):
        if verbose:
            print("\n")
            print(
                "--------------------------- Smoothing data ---------------------------"
            )
            print("\n")

        # Automatic tunning of the window size
        y_norm0 = (self.y - self.y.min()) / (self.y.max() - self.y.min())
        self.smoothed_vec_0 = [y_norm0]
        if len(y_norm0.shape) > 1:
            std_prev = np.std(diff(y_norm0.T, 1).T)
        else:
            std_prev = np.std(diff(y_norm0, 1))
        window_size_used = window_size
        std1 = []
        while True:
            std1.append(std_prev)
            next_window_size_used = window_size_used + delta
            if next_window_size_used > self.max_window_size:
                print("next_window_size_used > self.max_window_size")
                break
            window_size_used += delta
            y_norm0 = savgol_filter(y_norm0, window_size_used, poly_order)
            if len(y_norm0.shape) > 1:
                std_new = np.std(diff(y_norm0.T, 1).T)
            else:
                std_new = np.std(diff(y_norm0, 1))
            if verbose:
                print(
                    "Prev STD: %.5f - New STD: %.5f - Percent change: %.5f"
                    % (std_prev, std_new, 100 * (std_new - std_prev) / std_prev)
                )
            if abs((std_new - std_prev) / std_prev) < 0.1:
                window_size_used -= delta
                break
            else:
                std_prev = std_new
                self.smoothed_vec_0.append(y_norm0)
                y_norm0 = (self.y - self.y.min()) / (self.y.max() - self.y.min())

        if window_size_used > 1:
            self.best_window_size = int(window_size_used)
            print(
                "Smoothing window size: " + str(window_size_used),
                "\n",
            )

            self.y = savgol_filter(self.y, window_size_used, poly_order)
        else:
            print("No smoothing applied")
            print("\n")

        return self.y
