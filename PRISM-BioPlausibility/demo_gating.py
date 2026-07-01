"""Demonstrates the multiplicative gating of FVS across edit types."""
import numpy as np
from fvs import fvs, rasp_ddg_stub

wt = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
ca = {i + 1: np.array([i * 3.8, 0.0, 0.0]) for i in range(len(wt))}
active = {20: 1.0}  # catalytic residue at index 20

cases = {
    "conservative, far from site (I5V)": wt[:5] + "V" + wt[6:],
    "radical AT catalytic site (Q20D)":  wt[:20] + "D" + wt[21:],
    "destabilizing core (Q11W, hi ddG)": wt[:11] + "W" + wt[12:],
}
for name, var in cases.items():
    ddg = rasp_ddg_stub(wt, var)
    r = fvs(wt, var, ca, active, ddg=ddg)
    facs = "  ".join(f"{k}={v:.2f}" for k, v in r.factors.items())
    print(f"{name:38s} FVS={r.score:.3f}   {facs}")

# truncation that deletes the catalytic residue at 20
r = fvs(wt, wt, ca, active, ddg=0.0, edit_kind="nonsense", stop_k=15)
print(f"{'nonsense @15 (site@20 lost)':38s} FVS={r.score:.3f}   "
      f"P_prod={r.factors['P_prod']:.2f}")
