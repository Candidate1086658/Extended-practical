import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def reflected_color_5700(d_hbn_init=50.0, d_sio2_init=290.0, intensity_init=1.0):
    """
    Interactive plot showing the reflected spectrum and perceived color 
    of an hBN/SiO2/Si stack under 5700K illumination, including light intensity.
    """
    # 1. Setup Wavelengths for Visible Light (380nm to 780nm)
    wl_nm = np.arange(380, 781, 5)
    wl_m = wl_nm * 1e-9

    # 2. Generate 5700K Blackbody Illuminant
    h = 6.626e-34; c = 3e8; k_B = 1.381e-23; T = 5700
    radiance = (2 * h * c**2 / (wl_m**5)) * (1 / (np.exp(h * c / (wl_m * k_B * T)) - 1))
    L5700 = radiance / np.max(radiance)

    # 3. Approximate Human Eye Color Matching Functions (CIE 1964)
    x_bar = 0.398 * np.exp(-1250 * (np.log((wl_nm + 570.1) / 1014))**2) + 1.132 * np.exp(-234 * (np.log((1338 - wl_nm) / 743.5))**2)
    y_bar = 1.011 * np.exp(-0.5 * ((wl_nm - 556.1) / 46.14)**2)
    z_bar = 2.060 * np.exp(-32 * (np.log((wl_nm - 265.8) / 180.4))**2)

    # Calculate the XYZ values for the pure white light at baseline intensity (used for normalization)
    X_white = np.sum(L5700 * x_bar)
    Y_white = np.sum(L5700 * y_bar)
    Z_white = np.sum(L5700 * z_bar)

    # 4. TMM Calculation Function
    def calc_reflected_spectrum(d_hbn, d_sio2):
        n0 = 1.0; n1 = 2.1; n2 = 1.46; n3 = 3.96
        reflectance = []
        for wl in wl_nm:
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
            
        return np.array(reflectance) * L5700 # Return Baseline Reflected Spectrum

    # 5. Convert Spectrum to RGB Function
    '''Assume the formulas for relative XYZ, linear RGB conversion and 0-255 RGB' values are true'''
    def spectrum_to_rgb(spectrum):
        # Integrate spectrum with human eye receptors
        X = np.sum(spectrum * x_bar) / X_white
        Y = np.sum(spectrum * y_bar) / Y_white
        Z = np.sum(spectrum * z_bar) / Z_white
        
        # Matrix transform from XYZ to linear RGB
        R_lin =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        B_lin =  0.0557 * X - 0.2040 * Y + 1.0570 * Z
        
        # Clamp to [0, 1] to avoid out-of-gamut errors / simulate camera sensor saturation
        R_lin = np.clip(R_lin, 0, 1)
        G_lin = np.clip(G_lin, 0, 1)
        B_lin = np.clip(B_lin, 0, 1)
        
        # Gamma correction for monitors (gamma ~ 2.2)
        return (R_lin**(1/2.2), G_lin**(1/2.2), B_lin**(1/2.2))

    # Helper function to convert 0.0-1.0 float to 0-255 integer
    def format_rgb_text(color_tuple):
        r = int(round(color_tuple[0] * 255))
        g = int(round(color_tuple[1] * 255))
        b = int(round(color_tuple[2] * 255))
        return r, g, b

    # 6. Monochromatic helpers (TMM at single wavelength + raw XYZ -> RGB, no white-balance)
    def calc_reflectance_mono(d_hbn, d_sio2, wl):
        n0 = 1.0; n1 = 2.1; n2 = 1.46; n3 = 3.96
        delta1 = 2 * np.pi * n1 * d_hbn / wl
        delta2 = 2 * np.pi * n2 * d_sio2 / wl

        M1 = np.array([[np.cos(delta1), -1j * (1/n1) * np.sin(delta1)],
                       [-1j * n1 * np.sin(delta1), np.cos(delta1)]])
        M2 = np.array([[np.cos(delta2), -1j * (1/n2) * np.sin(delta2)],
                       [-1j * n2 * np.sin(delta2), np.cos(delta2)]])

        M = np.matmul(M1, M2)
        m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]

        r = ((m11 + m12 * n3) * n0 - (m21 + m22 * n3)) / ((m11 + m12 * n3) * n0 + (m21 + m22 * n3))
        return float(np.abs(r)**2)

    def mono_to_rgb(R_refl, wl, intensity):
        # Evaluate CMFs at single wavelength
        x = (0.398 * np.exp(-1250 * (np.log((wl + 570.1) / 1014))**2)
             + 1.132 * np.exp(-234 * (np.log((1338 - wl) / 743.5))**2))
        y = 1.011 * np.exp(-0.5 * ((wl - 556.1) / 46.14)**2)
        z = 2.060 * np.exp(-32 * (np.log((wl - 265.8) / 180.4))**2)

        # Raw XYZ (no white-balance normalization — CCD is not white-balanced)
        I = R_refl * intensity
        X = I * x
        Y = I * y
        Z = I * z

        R_lin =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        B_lin =  0.0557 * X - 0.2040 * Y + 1.0570 * Z

        R_lin = np.clip(R_lin, 0, 1)
        G_lin = np.clip(G_lin, 0, 1)
        B_lin = np.clip(B_lin, 0, 1)

        return (R_lin**(1/2.2), G_lin**(1/2.2), B_lin**(1/2.2))

    # --- Setup the Matplotlib UI ---
    fig = plt.figure(figsize=(10, 8))  # Extra height for the illumination-mode buttons

    # Create axes: [left, bottom, width, height]
    ax_plot = plt.axes([0.1, 0.40, 0.55, 0.42])
    ax_color = plt.axes([0.75, 0.40, 0.2, 0.42])

    # Illumination-mode button axes (top row)
    ax_btn_5700K = plt.axes([0.1, 0.90, 0.18, 0.06])
    ax_btn_470 = plt.axes([0.32, 0.90, 0.18, 0.06])
    ax_btn_590 = plt.axes([0.54, 0.90, 0.18, 0.06])

    # 3 Slider Axes
    ax_intensity = plt.axes([0.1, 0.22, 0.8, 0.04])
    ax_hbn = plt.axes([0.1, 0.13, 0.8, 0.04])
    ax_sio2 = plt.axes([0.1, 0.04, 0.8, 0.04])

    # Initial Calculation
    '''key change: multiply the reflected spectrum by the intensity slider value to simulate overexposure'''
    initial_spectrum = calc_reflected_spectrum(d_hbn_init, d_sio2_init) * intensity_init
    initial_color = spectrum_to_rgb(initial_spectrum)
    r_val, g_val, b_val = format_rgb_text(initial_color)

    # Substrate (bare SiO2/Si) color at d_hbn = 0
    initial_sub_spectrum = calc_reflected_spectrum(0.0, d_sio2_init) * intensity_init
    initial_sub_color = spectrum_to_rgb(initial_sub_spectrum)
    rs_val, gs_val, bs_val = format_rgb_text(initial_sub_color)

    # Plot initial spectrum
    line, = ax_plot.plot(wl_nm, initial_spectrum, color='black', lw=2)
    ax_plot.set_title("Reflected Spectrum under 5700K")
    ax_plot.set_xlabel("Wavelength (nm)")
    ax_plot.set_ylabel("Light Intensity")
    ax_plot.set_xlim(380, 780)
    ax_plot.set_ylim(0, max(1.05, intensity_init * 1.05))
    ax_plot.grid(True, linestyle=':')

    # Plot initial color swatch (outer = substrate, inner = hBN) and RGB text
    ax_color.set_xticks([])
    ax_color.set_yticks([])
    ax_color.set_title(
        f"RGB_hBN (inside): ({r_val}, {g_val}, {b_val})\n"
        f"RGB_substrate (outside): ({rs_val}, {gs_val}, {bs_val})"
    )
    rect_outer = plt.Rectangle((0, 0), 1, 1, facecolor=initial_sub_color)
    rect_inner = plt.Rectangle((0.25, 0.25), 0.5, 0.5, facecolor=initial_color)
    ax_color.add_patch(rect_outer)
    ax_color.add_patch(rect_inner)

    # Create Sliders
    slider_intensity = Slider(ax_intensity, 'Light Intensity', 0.0, 10.0, valinit=intensity_init, valstep=0.1)
    slider_hbn = Slider(ax_hbn, 'hBN (nm)', 0, 300, valinit=d_hbn_init, valstep=0.4)
    slider_hbn.valtext.set_text(
        f"{d_hbn_init:.1f} nm ({int(round(d_hbn_init / 0.4))} layers)"
    )
    slider_sio2 = Slider(ax_sio2, 'SiO$_2$ (nm)', 200, 400, valinit=d_sio2_init, valstep=1)

    # Illumination-mode buttons + state ('5700K' = broadband, '470' / '590' = monochromatic in nm)
    btn_5700K = Button(ax_btn_5700K, '5700K')
    btn_470 = Button(ax_btn_470, '470 nm')
    btn_590 = Button(ax_btn_590, '590 nm')
    current_mode = ['5700K']

    # Update function when sliders move or illumination mode changes
    def update(val):
        d_hbn = slider_hbn.val
        d_sio2 = slider_sio2.val
        intensity = slider_intensity.val

        # Update hBN slider label with thickness and layer count
        slider_hbn.valtext.set_text(
            f"{d_hbn:.1f} nm ({int(round(d_hbn / 0.4))} layers)"
        )

        mode = current_mode[0]

        if mode == '5700K':
            # Broadband path (white-balanced under 5700K)
            new_spectrum = calc_reflected_spectrum(d_hbn, d_sio2) * intensity
            new_color = spectrum_to_rgb(new_spectrum)

            new_sub_spectrum = calc_reflected_spectrum(0.0, d_sio2) * intensity
            new_sub_color = spectrum_to_rgb(new_sub_spectrum)

            line.set_ydata(new_spectrum)
            ax_plot.set_title("Reflected Spectrum under 5700K")
        else:
            # Monochromatic path (raw XYZ, NOT white-balanced)
            wl_target = float(mode)
            R_hbn = calc_reflectance_mono(d_hbn, d_sio2, wl_target)
            R_sub = calc_reflectance_mono(0.0, d_sio2, wl_target)
            new_color = mono_to_rgb(R_hbn, wl_target, intensity)
            new_sub_color = mono_to_rgb(R_sub, wl_target, intensity)

            # Build a spike spectrum for visualization
            spike = np.zeros_like(wl_nm, dtype=float)
            idx = int(np.argmin(np.abs(wl_nm - wl_target)))
            spike[idx] = R_hbn * intensity
            line.set_ydata(spike)
            ax_plot.set_title(f"Reflected Intensity under {wl_target:.0f} nm")

        r_val, g_val, b_val = format_rgb_text(new_color)
        rs_val, gs_val, bs_val = format_rgb_text(new_sub_color)

        rect_inner.set_facecolor(new_color)
        rect_outer.set_facecolor(new_sub_color)
        ax_color.set_title(
            f"RGB_hBN (inside): ({r_val}, {g_val}, {b_val})\n"
            f"RGB_substrate (outside): ({rs_val}, {gs_val}, {bs_val})"
        )

        # Dynamically scale Y-axis to handle overexposure visualization
        ax_plot.set_ylim(0, max(1.05, intensity * 1.05))

        fig.canvas.draw_idle()

    def set_mode_5700K(event):
        current_mode[0] = '5700K'
        update(None)

    def set_mode_470(event):
        current_mode[0] = '470'
        update(None)

    def set_mode_590(event):
        current_mode[0] = '590'
        update(None)

    btn_5700K.on_clicked(set_mode_5700K)
    btn_470.on_clicked(set_mode_470)
    btn_590.on_clicked(set_mode_590)

    slider_intensity.on_changed(update)
    slider_hbn.on_changed(update)
    slider_sio2.on_changed(update)

    plt.show()

# --- Run the application ---
reflected_color_5700(d_hbn_init=50, d_sio2_init=285, intensity_init=1.0)