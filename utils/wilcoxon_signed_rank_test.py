import numpy as np
from scipy.stats import wilcoxon

sample1 = np.array([
0.904840291,
0.919975877,
0.936373293,
0.874439657,
0.870863199,
0.919999003,
0.91898483,
0.886047304,
0.885762036,
0.880276978,
0.887319267,
0.799337983,
0.807626724,
0.827961624,
0.780640602,
0.872183979,
0.901806891,
0.892409384,
0.926825047,
0.920895338,
0.838932693,
0.913243175,
0.837514877,
0.919722974,
0.902779698,
0.877882957,
0.91415751,
0.881974876,
0.901899576,
0.947143201,
0.957322698,
0.965612775,
0.929423777,
0.93038394,
0.957041044,
0.956862462,
0.938630861,
0.936980759,
0.93061683,
0.9378839,
0.86643477,
0.883926402,
0.902342458,
0.87286745,
0.924319928,
0.944612486,
0.937143562,
0.961101312,
0.956045724,
0.907624046,
0.952332948,
0.908530592,
0.957775656,
0.944847158,
0.930101687,
0.951239035,
0.9359746,
0.945620289
])

sample2 = np.array([
0.86991483,
0.914802432,
0.768950582,
0.796377122,
0.900274694,
0.853266299,
0.829665661,
0.827110171,
0.899233758,
0.898270309,
0.897456706,
0.915844619,
0.89464736,
0.848915994,
0.766913712,
0.875151277,
0.820103288,
0.922821999,
0.889311969,
0.859852254,
0.869225025,
0.910867572,
0.867971182,
0.900144696,
0.929923117,
0.792060733,
0.905314386,
0.885985911,
0.792674363,
0.926289309,
0.954171245,
0.849601014,
0.879504061,
0.944232576,
0.919874957,
0.900733257,
0.900579572,
0.942265166,
0.942770587,
0.942663369,
0.954963001,
0.943879788,
0.913761175,
0.861799564,
0.932833998,
0.880892157,
0.95782702,
0.935135924,
0.923567139,
0.927015546,
0.95222921,
0.927357198,
0.941744635,
0.961609413,
0.862268544,
0.946479514,
0.934744258,
0.880486269
])

### Check for the same distribution
stat, p_value = wilcoxon(sample1, sample2)
print(f"statistic: {stat}, p value: {p_value}")

# Set the significance level
alpha = 0.05

if p_value < alpha:
    print("There is a significant difference, leading to the rejection of the null hypothesis.")
else:
    print("There is no significant difference and the null hypothesis cannot be rejected")

### Wilcoxon signed rank test (one-tailed test)
difference = sample1 - sample2
stat, p_value = wilcoxon(difference, alternative='greater')
print(f"statistic: {stat}, p value: {p_value}")
