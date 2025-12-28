import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace
import random
import io

# Configurazione Pagina
st.set_page_config(page_title="Aero-NDT Exam Suite", layout="wide")

# --- MOTORE DI GENERAZIONE ---
def generate_scan(kv, ma, time, material, thickness, iqi_type):
    size = 800
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    mu = mu_map[material] * (120/kv)**1.5
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    defects = ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"]
    chosen_defect = random.choice(defects)
    
    # Coordinate per il contorno (x, y, raggio/dimensione)
    defect_coords = {"x": 400, "y": 400, "w": 50, "h": 50} 
    
    cx, cy = 400, 400
    if chosen_defect == "Cricca":
        y_range = range(300, 500)
        for y in y_range:
            m_sp[y, int(cx)] -= 0.6; cx += random.uniform(-0.6, 0.6)
        defect_coords = {"x": 400, "y": 400, "w": 40, "h": 220}
    elif chosen_defect == "Porosità Singola":
        y, x = np.ogrid[:size, :size]
        m_sp[(x-400)**2 + (y-400)**2 <= 5**2] -= 2.0
        defect_coords = {"x": 400, "y": 400, "w": 30, "h": 30}
    elif chosen_defect == "Inclusione Tungsteno":
        y, x = np.ogrid[:size, :size]
        m_sp[(x-400)**2 + (y-400)**2 <= 4**2] += 12.0
        defect_coords = {"x": 400, "y": 400, "w": 30, "h": 30}
    elif chosen_defect == "Incisione Marginale":
        m_sp[200:600, 430:433] -= 1.2
        defect_coords = {"x": 431, "y": 400, "w": 20, "h": 420}
    elif chosen_defect == "Mancata Fusione":
        m_sp[200:600, 398:401] -= 1.8
        defect_coords = {"x": 399, "y": 400, "w": 20, "h": 420}
    elif chosen_defect == "Cluster Porosità":
        for _ in range(8):
            rx, ry = random.randint(370, 430), random.randint(370, 430)
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 3**2] -= 1.5
        defect_coords = {"x": 400, "y": 400, "w": 100, "h": 100}

    # Calcolo immagine
    dose = (kv**2) * ma * time * 0.05
    signal = dose * np.exp(-mu * m_sp)
    signal = gaussian_filter(signal, sigma=1.1)
    noise = np.random.normal(0, np.sqrt(signal + 1) * 2.2, (size, size))
    raw = np.clip(signal + noise, 0, 65535).astype(np.uint16)
    
    return raw, chosen_defect, defect_coords

# --- INTERFACCIA STREAMLIT ---
st.title("☢️ Aero-NDT Exam Suite v7.6")

if 'show_boundary' not in st.session_state:
    st.session_state['show_boundary'] = False

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Parametri")
    kv = st.slider("kV", 40, 250, 110)
    ma = st.slider("mA", 0.5, 15.0, 5.0)
    time = st.slider("Secondi", 1, 120, 25)
    mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718"])
    thick = st.number_input("Spessore (mm)", 1, 25, 10)
    
    if st.button("ACQUISICI NUOVA IMMAGINE"):
        raw, defect, coords = generate_scan(kv, ma, time, mat, thick, "ISO")
        st.session_state['raw_data'] = raw
        st.session_state['true_defect'] = defect
        st.session_state['coords'] = coords
        st.session_state['show_boundary'] = False # Reset contorno

with col2:
    if 'raw_data' in st.session_state:
        # Controlli visualizzazione
        c1, c2 = st.columns(2)
        l_val = c1.slider("Level", 0, 65535, 32768)
        w_val = c2.slider("Width", 100, 65535, 65535)
        
        # Rendering
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
        vmin, vmax = max(0, l_val - w_val//2), min(65535, l_val + w_val//2)
        ax.imshow(st.session_state['raw_data'], cmap='gray_r', vmin=vmin, vmax=vmax)
        
        # DISEGNA IL CONTORNO SE RICHIESTO
        if st.session_state['show_boundary']:
            c = st.session_state['coords']
            # Disegna un rettangolo rosso tratteggiato intorno al difetto
            rect = plt.Rectangle((c['x'] - c['w']//2, c['y'] - c['h']//2), 
                                 c['w'], c['h'], linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(c['x'] - c['w']//2, c['y'] - c['h']//2 - 10, "DIFETTO RILEVATO", color='red', fontsize=10, fontweight='bold')
        
        ax.axis('off')
        st.pyplot(fig)
        
        # MODULO VALUTAZIONE
        st.divider()
        scelta = st.selectbox("Cosa vedi?", ["Seleziona...", "Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Incisione Marginale", "Mancata Fusione"])
        
        if st.button("VALUTA ESAME"):
            st.session_state['show_boundary'] = True # Attiva il contorno
            if scelta == st.session_state['true_defect']:
                st.success(f"CORRETTO! Il difetto è un/una {st.session_state['true_defect']}. Vedi il riquadro rosso.")
            else:
                st.error(f"ERRATO. Il difetto era: {st.session_state['true_defect']}. Controlla la posizione evidenziata.")
            st.rerun() # Ricarica per mostrare il contorno sul grafico