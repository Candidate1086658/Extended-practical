import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import csv

# Ensure toolbar/programmatic saves use a tight bbox so axis labels and titles
# at the figure edges aren't clipped on export.
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.2
plt.rcParams['savefig.dpi'] = 150

# 1. Fetch the CIE 1964 10-degree XYZ CMFs (1nm resolution) from CVRL
url = "http://www.cvrl.org/database/data/cmfs/ciexyz64_1.csv"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
csv_data = response.read().decode('utf-8')

wl_data, x_data, y_data, z_data = [], [], [], []
for row in csv.reader(csv_data.splitlines()):
    if row and len(row) == 4:
        wl_data.append(float(row[0]))
        x_data.append(float(row[1]))
        y_data.append(float(row[2]))
        z_data.append(float(row[3]))
        
wl_data = np.array(wl_data)
x_data = np.array(x_data)
y_data = np.array(y_data)
z_data = np.array(z_data)

# Filter out UV and IR to match the standard 380 - 780 nm visible range
mask = (wl_data >= 380) & (wl_data <= 780)
wl_data, x_data, y_data, z_data = wl_data[mask], x_data[mask], y_data[mask], z_data[mask]

# 2. Evaluate the analytic fits provided in the image
wl_fit = np.linspace(380, 780, 401)

x_fit = 0.398 * np.exp(-1250 * (np.log((wl_fit + 570.1) / 1014))**2) + \
        1.132 * np.exp(-234 * (np.log((1338 - wl_fit) / 743.5))**2)
y_fit = 1.011 * np.exp(-0.5 * ((wl_fit - 556.1) / 46.14)**2)
z_fit = 2.060 * np.exp(-32 * (np.log((wl_fit - 265.8) / 180.4))**2)


'''x_fit = np.exp(-0.5 * ((wl_fit - 595.8) / 33.33)**2) + 0.3 * np.exp(-0.5 * ((wl_fit - 446.8) / 19.44)**2)
y_fit = np.exp(-0.5 * ((wl_fit - 556.3) / 46.14)**2)
z_fit = np.exp(-0.5 * ((wl_fit - 449.8) / 29.83)**2)'''

# 3. Plot the comparison
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                         constrained_layout=True)

# X Channel
axes[0].plot(wl_data, x_data, 'r-', linewidth=4, alpha=0.4, label='CIE 1964 10° $\\bar{x}_{64}$ (Data Table)')
axes[0].plot(wl_fit, x_fit, 'k--', linewidth=2, label='Analytic Fit $\\tilde{x}_{64}$')
axes[0].set_ylabel('Amplitude', fontsize=16)
axes[0].legend(fontsize=14)
axes[0].grid(True, linestyle=':', alpha=0.7)
axes[0].set_title('CIE 1964 10° Standard Observer vs Analytic Approximation', fontsize=16)

# Y Channel
axes[1].plot(wl_data, y_data, 'g-', linewidth=4, alpha=0.4, label='CIE 1964 10° $\\bar{y}_{64}$ (Data Table)')
axes[1].plot(wl_fit, y_fit, 'k--', linewidth=2, label='Analytic Fit $\\tilde{y}_{64}$')
axes[1].set_ylabel('Amplitude', fontsize=16)
axes[1].legend(fontsize=14)
axes[1].grid(True, linestyle=':', alpha=0.7)

# Z Channel
axes[2].plot(wl_data, z_data, 'b-', linewidth=4, alpha=0.4, label='CIE 1964 10° $\\bar{z}_{64}$ (Data Table)')
axes[2].plot(wl_fit, z_fit, 'k--', linewidth=2, label='Analytic Fit $\\tilde{z}_{64}$')
axes[2].set_ylabel('Amplitude', fontsize=16)
axes[2].set_xlabel('Wavelength (nm)', fontsize=16)
axes[2].legend(fontsize=14)
axes[2].grid(True, linestyle=':', alpha=0.7)

plt.xlim(380, 780)
plt.show()