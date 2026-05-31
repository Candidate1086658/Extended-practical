import numpy as np
import matplotlib.pyplot as plt

# Ensure toolbar/programmatic saves use a tight bbox so axis labels and titles
# at the figure edges aren't clipped on export.
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.2
plt.rcParams['savefig.dpi'] = 150

# =====================================================================
# 1. Experimental Data
# =====================================================================
flake_r = np.array([182, 68, 190, 243, 233, 242, 22, 88, 214, 225, 245, 223, 81, 244, 45, 241, 67, 221, 175, 238, 214, 78, 219, 61, 242, 30, 156, 89, 212, 122, 237])
flake_g = np.array([238, 235, 107, 176, 239, 154, 181, 75, 236, 234, 217, 92, 211, 201, 232, 159, 209, 240, 125, 130, 240, 206, 91, 230, 203, 193, 131, 215, 238, 221, 139])
flake_b = np.array([46, 85, 202, 114, 39, 143, 195, 178, 112, 178, 46, 193, 190, 78, 101, 129, 198, 105, 204, 159, 132, 196, 192, 102, 74, 196, 196, 194, 137, 190, 186])

substrate_r = np.array([133, 133, 133, 130, 135, 134, 128, 128, 128, 128, 128, 134, 126, 129, 123, 131, 132, 125, 134, 134, 135, 134, 134, 133, 132, 136, 134, 136, 130, 134, 134])
substrate_g = np.array([58, 57, 56, 57, 57, 58, 59, 59, 59, 59, 59, 58, 60, 59, 59, 59, 59, 59, 58, 58, 59, 58, 58, 58, 57, 57, 59, 58, 61, 58, 58])
substrate_b = np.array([162, 162, 160, 162, 161, 162, 163, 163, 163, 163, 163, 162, 165, 165, 163, 165, 165, 163, 164, 160, 164, 162, 162, 162, 161, 162, 164, 163, 163, 162, 162])

thickness_exp = np.array([167.0, 140.0, 115.0, 91.0, 73.0, 75.0, 24.0, 3.6, 47.0, 51.0, 85.0, 108.0, 29.0, 93.0, 131.0, 96.0, 21.0, 57.0, 116.0, 88.0, 46.0, 31.0, 98.0, 278.0, 167.0, 28.0, 123.0, 33.0, 50.0, 34.0, 219.0])


# wavelengths from 380 nm to 780 nm in 5 nm increments (81 points total)
wavelength_nm = [
    380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 
    430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 
    480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 
    530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 
    580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 
    630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 
    680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 
    730, 735, 740, 745, 750, 755, 760, 765, 770, 775, 
    780
]

# refractive index of Si
si_n = [
    4.0812, 4.1404, 4.1903, 4.231, 4.2717, 4.3124, 4.3531, 4.3852, 4.4004, 4.4157,
    4.4309, 4.4462, 4.4615, 4.4713, 4.4742, 4.4772, 4.4801, 4.483, 4.486, 4.4889,
    4.4868, 4.4815, 4.4763, 4.471, 4.4681, 4.4656, 4.4632, 4.4608, 4.4528, 4.4421,
    4.4315, 4.4209, 4.4102, 4.3996, 4.3889, 4.3783, 4.3677, 4.3568, 4.3452, 4.3337,
    4.3221, 4.3106, 4.299, 4.2875, 4.276, 4.2644, 4.2529, 4.2413, 4.2298, 4.2206,
    4.2115, 4.2023, 4.1931, 4.1839, 4.1748, 4.1647, 4.1536, 4.1426, 4.1315, 4.1205,
    4.1094, 4.0984, 4.0876, 4.0778, 4.0679, 4.058, 4.0481, 4.0382, 4.0284, 4.0185,
    4.0088, 4.0, 3.9912, 3.9825, 3.9737, 3.9649, 3.9561, 3.9474, 3.9386, 3.9299,
    3.9231
]

wavelength_nm_arr = np.array(wavelength_nm)
si_n_arr = np.array(si_n)


# =====================================================================
# 2. Convert to Linear Intensity & Calculate Experimental Contrast
# =====================================================================
def srgb_to_linear(rgb_array):
    """Reverses the gamma compression to yield linear light intensity."""
    return (rgb_array / 255.0) ** 2.2

def experimental_contrast(flake_val, sub_val):
    """Calculates optical contrast C = (I_flake - I_substrate) / I_substrate"""
    I_flake = srgb_to_linear(flake_val)
    I_sub = srgb_to_linear(sub_val)
    return (I_flake - I_sub) / I_sub

Rc = experimental_contrast(flake_r, substrate_r)
Gc = experimental_contrast(flake_g, substrate_g)
Bc = experimental_contrast(flake_b, substrate_b)

# =====================================================================
# 3. Theoretical Model (TMM + CIE 1964 -> Linear Contrast)
# =====================================================================
def extract_theoretical_contrast(hbn_thicknesses, d_sio2=285.0):
    wl_nm = np.arange(380, 781, 5)
    wl_m = wl_nm * 1e-9

    # 5700K Blackbody Illuminant
    h = 6.626e-34; c = 3e8; k_B = 1.381e-23; T = 5700
    radiance = (2 * h * c**2 / (wl_m**5)) * (1 / (np.exp(h * c / (wl_m * k_B * T)) - 1))
    L5700 = radiance / np.max(radiance)

    # Approximate CIE 1964 10-deg Color Matching Functions
    x_bar = 0.398 * np.exp(-1250 * (np.log((wl_nm + 570.1) / 1014))**2) + 1.132 * np.exp(-234 * (np.log((1338 - wl_nm) / 743.5))**2)
    y_bar = 1.011 * np.exp(-0.5 * ((wl_nm - 556.1) / 46.14)**2)
    z_bar = 2.060 * np.exp(-32 * (np.log((wl_nm - 265.8) / 180.4))**2)

    X_white = np.sum(L5700 * x_bar)
    Y_white = np.sum(L5700 * y_bar)
    Z_white = np.sum(L5700 * z_bar)

    n0 = 1.0; n1 = 2.2; n2 = 1.46  # n3 (Si) is wavelength-dependent — looked up per λ

    # Helper function to get Linear RGB intensity for a specific hBN thickness
    def get_linear_rgb(d_hbn):
        reflectance = []
        for wl in wl_nm:
            n3 = np.interp(wl, wavelength_nm_arr, si_n_arr)
            delta1 = 2 * np.pi * n1 * d_hbn / wl
            delta2 = 2 * np.pi * n2 * d_sio2 / wl
            
            M1 = np.array([[np.cos(delta1), -1j * (1/n1) * np.sin(delta1)],
                           [-1j * n1 * np.sin(delta1), np.cos(delta1)]])
            M2 = np.array([[np.cos(delta2), -1j * (1/n2) * np.sin(delta2)],
                           [-1j * n2 * np.sin(delta2), np.cos(delta2)]])
            
            M = np.matmul(M1, M2)
            m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]
            
            r = ((m11 + m12 * n3) * n0 - (m21 + m22 * n3)) / ((m11 + m12 * n3) * n0 + (m21 + m22 * n3))
            reflectance.append(np.abs(r)**2)
            
        reflected_spectrum = np.array(reflectance) * L5700
        
        # Integrate into XYZ
        X = np.sum(reflected_spectrum * x_bar) / X_white
        Y = np.sum(reflected_spectrum * y_bar) / Y_white
        Z = np.sum(reflected_spectrum * z_bar) / Z_white
        
        # Convert to Linear RGB
        R_lin =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        B_lin =  0.0557 * X - 0.2040 * Y + 1.0570 * Z
        
        return R_lin, G_lin, B_lin

    # 1. Calculate the base intensity of the bare substrate (d_hbn = 0)
    R_base, G_base, B_base = get_linear_rgb(0.0)
    
    # 2. Sweep thicknesses and calculate contrast + display-encoded RGB
    C_R_list, C_G_list, C_B_list = [], [], []
    R255_list, G255_list, B255_list = [], [], []
    for d_hbn in hbn_thicknesses:
        R_lin, G_lin, B_lin = get_linear_rgb(d_hbn)

        C_R_list.append((R_lin - R_base) / R_base)
        C_G_list.append((G_lin - G_base) / G_base)
        C_B_list.append((B_lin - B_base) / B_base)

        # Gamma-encode (clip to gamut, apply 1/2.2, scale to 0-255)
        R255_list.append(round(np.clip(R_lin, 0, 1) ** (1/2.2) * 255))
        G255_list.append(round(np.clip(G_lin, 0, 1) ** (1/2.2) * 255))
        B255_list.append(round(np.clip(B_lin, 0, 1) ** (1/2.2) * 255))

    return (np.array(C_R_list), np.array(C_G_list), np.array(C_B_list),
            np.array(R255_list), np.array(G255_list), np.array(B255_list))

# =====================================================================
# 4. Execute and Plot
# =====================================================================
# Generate theoretical thickness array
hbn_array = np.arange(0, 300, 1)
sio2_thickness = 285 # Ensure this matches your experimental wafer

# Get theoretical RGB Contrasts and display-encoded 0-255 RGB
(contrast_R_theo, contrast_G_theo, contrast_B_theo,
 R255_theo, G255_theo, B255_theo) = extract_theoretical_contrast(hbn_array, d_sio2=sio2_thickness)

# --- Gradient of display RGB between 0 and 6 nm, scaled to per-atomic-layer (0.4 nm) ---
layer_thickness = 0.4  # nm per atomic layer of hBN
t_lo, t_hi = 0.0, 4.0  # nm
i_lo = int(np.argmin(np.abs(hbn_array - t_lo)))
i_hi = int(np.argmin(np.abs(hbn_array - t_hi)))
delta_t = hbn_array[i_hi] - hbn_array[i_lo]
dR_per_layer = (R255_theo[i_hi] - R255_theo[i_lo]) * layer_thickness / delta_t
dG_per_layer = (G255_theo[i_hi] - G255_theo[i_lo]) * layer_thickness / delta_t
dB_per_layer = (B255_theo[i_hi] - B255_theo[i_lo]) * layer_thickness / delta_t

print(f"White-light display RGB gradient (slope between {hbn_array[i_lo]:.0f} and {hbn_array[i_hi]:.0f} nm) "
      f"per atomic layer ({layer_thickness} nm/layer):")
print(f"  dR = {dR_per_layer:+.4f} per layer")
print(f"  dG = {dG_per_layer:+.4f} per layer")
print(f"  dB = {dB_per_layer:+.4f} per layer")

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                         constrained_layout=True)

# --- Red Channel Plot ---
axes[0].plot(hbn_array, contrast_R_theo, 'r--', linewidth=2, label='Theoretical Red Contrast')
axes[0].scatter(thickness_exp, Rc, color='red', marker='o', s=45, alpha=0.8, edgecolors='black', label='Experimental Red Contrast')
axes[0].set_title(f'Intensity Contrast vs. hBN Thickness (SiO$_2$ = {sio2_thickness} nm)', fontsize=14)
axes[0].set_ylabel('Contrast $(I_{hBN} - I_{sub}) / I_{sub}$', fontsize=12)
axes[0].axhline(0, color='black', linestyle='-', linewidth=1)
axes[0].grid(True, linestyle=':', alpha=0.7)
axes[0].legend(loc='upper right', framealpha=0.9)

# --- Green Channel Plot ---
axes[1].plot(hbn_array, contrast_G_theo, 'g--', linewidth=2, label='Theoretical Green Contrast')
axes[1].scatter(thickness_exp, Gc, color='green', marker='^', s=45, alpha=0.8, edgecolors='black', label='Experimental Green Contrast')
axes[1].set_title(f'Intensity Contrast vs. hBN Thickness (SiO$_2$ = {sio2_thickness} nm)', fontsize=14)
axes[1].set_ylabel('Contrast $(I_{hBN} - I_{sub}) / I_{sub}$', fontsize=12)
axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1].grid(True, linestyle=':', alpha=0.7)
axes[1].legend(loc='upper right', framealpha=0.9)

# --- Blue Channel Plot ---
axes[2].plot(hbn_array, contrast_B_theo, 'b--', linewidth=2, label='Theoretical Blue Contrast')
axes[2].scatter(thickness_exp, Bc, color='blue', marker='s', s=45, alpha=0.8, edgecolors='black', label='Experimental Blue Contrast')
axes[2].set_title(f'Intensity Contrast vs. hBN Thickness (SiO$_2$ = {sio2_thickness} nm)', fontsize=14)
axes[2].set_xlabel('hBN Thickness (nm)', fontsize=12)
axes[2].set_ylabel('Contrast $(I_{hBN} - I_{sub}) / I_{sub}$', fontsize=12)
axes[2].axhline(0, color='black', linestyle='-', linewidth=1)
axes[2].grid(True, linestyle=':', alpha=0.7)
axes[2].legend(loc='upper right', framealpha=0.9)
axes[2].set_xlim(0, 300)

# --- Theoretical 0-255 RGB Plot ---
fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax2.plot(hbn_array, R255_theo, 'r-', linewidth=2, label='Red')
ax2.plot(hbn_array, G255_theo, 'g-', linewidth=2, label='Green')
ax2.plot(hbn_array, B255_theo, 'b-', linewidth=2, label='Blue')
ax2.set_title(f'Theoretical Display RGB (0-255) vs. hBN Thickness (SiO$_2$ = {sio2_thickness} nm)', fontsize=14)
ax2.set_xlabel('hBN Thickness (nm)', fontsize=12)
ax2.set_ylabel('RGB Value (0-255)', fontsize=12)
ax2.set_xlim(0, 300)
ax2.set_ylim(0, 255)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper right', framealpha=0.9)

plt.show()