# app.py — Robust Exoplanet Detector Dashboard
# Requirements (install once): pip install streamlit numpy pandas scikit-learn xgboost joblib imbalanced-learn plotly reportlab matplotlib

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import hashlib
import math
import io
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="Exoplanet Detector — Professional", layout="wide")
st.title("🔭 Exoplanet Detector — Professional Dashboard")

# -------------------------
# User-editable constant
# -------------------------
MODEL_PATH = "exoplanet_detector_perfect.pkl"  # put the .pkl here

# -------------------------
# Load model bundle (safe)
# -------------------------
try:
    bundle = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model bundle '{MODEL_PATH}': {e}")
    st.stop()

# Show bundle keys for debugging (collapsible)
with st.expander("🔑 Model bundle contents (click to expand)"):
    try:
        st.write({k: type(v).__name__ for k, v in bundle.items()})
    except Exception:
        st.write(list(bundle.keys()))

# -------------------------
# Find components robustly
# -------------------------
# Classifier(s): any object with callable predict_proba
classifiers = {}
for k, v in bundle.items():
    if hasattr(v, "predict_proba") and callable(getattr(v, "predict_proba")):
        classifiers[k] = v

# Also check commonly used keys
fallback_names = ["calibrated_classifier", "final_model", "model", "clf", "classifier"]
for name in fallback_names:
    if name in bundle and name not in classifiers:
        obj = bundle[name]
        if hasattr(obj, "predict_proba") and callable(getattr(obj, "predict_proba")):
            classifiers[name] = obj

# Imputer & scaler
imputer = bundle.get("imputer") or bundle.get("imp") or bundle.get("simple_imputer")
scaler = bundle.get("scaler") or bundle.get("std_scaler")

# Reference nearest-neighbor model & table
nn_model = bundle.get("nn_ref_model") or bundle.get("nn_model") or None
# Reference nearest-neighbor model & table
ref_table = None
if "ref_table" in bundle:
    ref_table = bundle["ref_table"]
elif "reference_table" in bundle:
    ref_table = bundle["reference_table"]

nn_model = bundle.get("nn_ref_model") or bundle.get("nn_model") or None


# Feature columns & OOD threshold (optional)
feature_cols = bundle.get("feature_columns") or bundle.get("feature_cols")
ood_threshold = bundle.get("ood_threshold")

# Meta
meta = bundle.get("meta", {})

# Validate we have at least one classifier and imputer+scaler
if len(classifiers) == 0:
    st.error("❌ No classifier with predict_proba found in the bundle. Keys found: " + ", ".join(list(bundle.keys())))
    st.stop()

if imputer is None or scaler is None:
    st.error("❌ Imputer or scaler missing from the bundle. Required for preprocessing.")
    st.stop()

# If feature_cols missing, try to infer from ref_table
STANDARD_FEATURE_SET = [
    "orbital_period_days","planet_radius_re","planet_mass_me","transit_depth","transit_duration",
    "model_snr","impact","eq_temperature_k","stellar_teff_k","stellar_radius_solar",
    "stellar_mass_solar","stellar_logg","insolation_flux","semi_major_au"
]
if feature_cols is None:
    if isinstance(ref_table, pd.DataFrame):
        feature_cols = [c for c in STANDARD_FEATURE_SET if c in ref_table.columns]
    else:
        feature_cols = STANDARD_FEATURE_SET.copy()

if len(feature_cols) == 0:
    st.error("❌ Couldn't determine feature columns. Ensure 'feature_columns' present in the .pkl or ref_table contains known columns.")
    st.stop()

# Prepare ref_scaled and NN (if needed)
ref_scaled = None
if isinstance(ref_table, pd.DataFrame) and imputer is not None and scaler is not None:
    try:
        # ensure feature cols exist in ref_table
        ref_table_for_feats = ref_table.copy()
        for f in feature_cols:
            if f not in ref_table_for_feats.columns:
                ref_table_for_feats[f] = np.nan
        ref_imp = imputer.transform(ref_table_for_feats[feature_cols])
        ref_scaled = scaler.transform(ref_imp)
    except Exception as e:
        st.warning(f"Could not prepare ref_scaled from ref_table: {e}")
        ref_scaled = None

# If no nn_model provided, and we have ref_scaled, train a NN locally (fast)
if nn_model is None and ref_scaled is not None:
    from sklearn.neighbors import NearestNeighbors
    try:
        nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(ref_scaled)
    except Exception as e:
        st.warning(f"Could not build nn_model: {e}")
        nn_model = None

# If ood_threshold missing, compute 95th percentile of self-nearest distances (if ref_scaled available)
if ood_threshold is None and ref_scaled is not None:
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1).fit(ref_scaled)
        dists, idxs = nbrs.kneighbors(ref_scaled)
        # self-nearest is column 1 (0 is zero distance to itself)
        self_nearest = dists[:, 1]
        ood_threshold = float(np.percentile(self_nearest, 95))
    except Exception as e:
        st.warning(f"Could not compute ood_threshold: {e}")
        ood_threshold = None

# --------------------------------
# Sidebar input defaults from ref_table medians (if available)
# --------------------------------
st.sidebar.header("🛰 Candidate Parameters")
inputs = {}
for f in feature_cols:
    default = 1.0
    try:
        if isinstance(ref_table, pd.DataFrame) and f in ref_table.columns:
            med = pd.to_numeric(ref_table[f], errors="coerce").median()
            if not np.isnan(med):
                default = float(med)
    except Exception:
        default = 1.0
    # Use appropriate formatting
    inputs[f] = st.sidebar.number_input(f, value=default, format="%.6f")

# Extra controls
show_debug = st.sidebar.checkbox("Show debug info (keys + chosen models)", value=False)

# Debug display of chosen classifiers
if show_debug:
    st.sidebar.write("Detected classifiers:", list(classifiers.keys()))
    st.sidebar.write("Imputer:", type(imputer).__name__)
    st.sidebar.write("Scaler:", type(scaler).__name__)
    st.sidebar.write("NN model:", type(nn_model).__name__ if nn_model is not None else None)
    st.sidebar.write("Feature cols:", feature_cols)
    st.sidebar.write("OOD threshold:", ood_threshold)
    st.sidebar.write("Ref_table rows:", len(ref_table) if isinstance(ref_table, pd.DataFrame) else None)

# -------------------------
# Small physics helpers
# -------------------------
G = 6.67430e-11
M_sun = 1.98847e30
R_sun = 6.957e8
AU_m = 1.495978707e11
R_earth_m = 6.371e6
M_earth_kg = 5.972e24

def period_to_a_au(P_days, M_star_solar):
    try:
        P = float(P_days) * 24 * 3600.0
        M_star = float(M_star_solar) * M_sun
        a_m = (G * M_star * P**2 / (4 * math.pi**2))**(1/3)
        return a_m / AU_m
    except Exception:
        return np.nan

def equilibrium_temperature(Teff_star, R_star_solar, a_au, albedo=0.3, f=1.0):
    try:
        Teff_star = float(Teff_star)
        R_star_m = float(R_star_solar) * R_sun
        a_m = float(a_au) * AU_m
        Teq = Teff_star * math.sqrt(R_star_m / (2 * a_m)) * (1 - albedo)**0.25 * f**(-0.25)
        return float(Teq)
    except Exception:
        return np.nan

# molecules guess by temperature
def guess_molecules(eq_temp_k):
    if np.isnan(eq_temp_k):
        return ["Unknown"]
    t = float(eq_temp_k)
    if t < 200:
        return ["CH4 (methane)", "NH3 (ammonia)"]
    if t < 500:
        return ["H2O (water vapor)", "CO2 (carbon dioxide)"]
    if t < 1500:
        return ["H2O (water vapor)", "CO2", "SO2"]
    return ["Na (sodium)", "K (potassium)", "H (atomic hydrogen)"]

# life probability heuristic
def life_probability_score(water_possible, hab_zone, atmosphere_type, snr=None):
    score = 0
    if water_possible: score += 40
    if hab_zone == "Habitable Zone": score += 30
    if "Earth" in atmosphere_type or "Nitrogen" in atmosphere_type: score += 20
    if snr is not None:
        # a small boost for higher SNR (more reliable detection)
        score += min(10, max(0, (snr - 5) / 5))
    return int(min(100, score))

# similarity percent from nearest distance
def similarity_pct(dist, ood_thresh):
    if ood_thresh is None or np.isnan(ood_thresh) or ood_thresh == 0:
        return None
    pct = max(0.0, 100.0 * (1.0 - (dist / (ood_thresh * 2.0))))  # scaled; if dist==0 -> ~100, dist==2*ood -> 0
    pct = max(0.0, min(100.0, pct))
    return pct

# -------------------------
# Analyze button
# -------------------------
if st.sidebar.button("🔎 Analyze candidate"):
    # Prepare DataFrame of inputs in same columns as feature_cols
    row = pd.DataFrame([{c: inputs.get(c, np.nan) for c in feature_cols}])
    # Preprocess
    try:
        row_imp = imputer.transform(row)
        row_scaled = scaler.transform(row_imp)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # --- Predictions from all available classifiers ---
    probs = {}
    for name, clf in classifiers.items():
        try:
            p = clf.predict_proba(row_scaled)[0, 1]
            probs[name] = float(p)
        except Exception as e:
            # if predict_proba fails, skip
            st.warning(f"predict_proba failed for model '{name}': {e}")

    if len(probs) == 0:
        st.error("❌ No classifier produced probabilities. Check bundle classifiers.")
        st.stop()

    # Ensemble average
    ensemble_prob = float(np.mean(list(probs.values())))
    # show individual model probs in an expander
    with st.expander("Model probabilities"):
        st.write(probs)

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ensemble_prob * 100,
        title={"text": "Exoplanet Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "lightcoral"},
                {"range": [50, 75], "color": "khaki"},
                {"range": [75, 100], "color": "lightgreen"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Suggested name & category
    radius_val = row.iloc[0].get("planet_radius_re", np.nan)
    try:
        radius_val_float = float(radius_val)
    except Exception:
        radius_val_float = np.nan
    if np.isnan(radius_val_float):
        cat_code, cat_label = "unknown", "Unknown"
    else:
        if radius_val_float < 1.5:
            cat_code, cat_label = "earth-like", "Earth-like"
        elif radius_val_float < 3.0:
            cat_code, cat_label = "super-earth", "Super-Earth"
        elif radius_val_float < 6.0:
            cat_code, cat_label = "mini-neptune", "Mini-Neptune"
        else:
            cat_code, cat_label = "gas-giant", "Gas Giant"
    name = deterministic_name = None
    try:
        name = deterministic_name(cat_code, row.iloc[0].to_dict()) if radius_val_float == radius_val_float else "Exo-Unknown"
    except Exception:
        # fallback deterministic_name implementation
        pool = {"earth-like": ["Terra","Gaia","Astra"], "super-earth": ["Kepleria","Novus"], "mini-neptune": ["Neptara","Azure"], "gas-giant": ["Jovia","Zephyr"], "unknown": ["Exo"]}
        suffix = list("abcdefghij")
        h = int(hashlib.sha256(str(row.iloc[0].to_dict()).encode()).hexdigest()[:8], 16)
        name = f"{pool.get(cat_code, ['Exo'])[h % len(pool.get(cat_code, ['Exo']))]}-{(h % 9999) + 1}{suffix[(h // len(pool.get(cat_code, ['Exo'])) ) % len(suffix)]}"

    st.subheader(f"🪐 Suggested name: **{name}** — {cat_label}")

    # Orbit visualization (use semi_major_au if provided, else try period->a if stellar_mass available)
    a = row.iloc[0].get("semi_major_au", np.nan)
    if np.isnan(a):
        # try compute from period & stellar mass
        P = row.iloc[0].get("orbital_period_days", np.nan)
        Mstar = row.iloc[0].get("stellar_mass_solar", np.nan)
        a = period_to_a_au(P, Mstar) if not (np.isnan(P) or np.isnan(Mstar)) else 1.0
    if np.isnan(a) or a <= 0:
        a = 1.0

    theta = np.linspace(0, 2 * np.pi, 200)
    fig_orbit = go.Figure()
    fig_orbit.add_trace(go.Scatter(x=a * np.cos(theta), y=a * np.sin(theta), mode="lines", name="Orbit"))
    fig_orbit.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=20, color="yellow"), name="Star"))
    fig_orbit.add_trace(go.Scatter(x=[a], y=[0], mode="markers", marker=dict(size=12, color="blue"), name="Candidate"))
    fig_orbit.update_layout(title="Orbit visualization (AU)", xaxis_title="X (AU)", yaxis_title="Y (AU)", width=600, height=450)
    st.plotly_chart(fig_orbit, use_container_width=False)

    # Closest known exoplanet + similarity and OOD
    nearest_info = None
    nearest_dist = None
    similarity = None
    is_ood = False
    if nn_model is not None:
        try:
            dist, idx = nn_model.kneighbors(row_scaled, n_neighbors=1)
            nearest_dist = float(dist[0][0])
            idx0 = int(idx[0][0])
            if isinstance(ref_table, pd.DataFrame) and idx0 < len(ref_table):
                nearest_info = ref_table.iloc[idx0].to_dict()
            # similarity percent relative to ood_threshold (if present)
            similarity = similarity_pct(nearest_dist, ood_threshold)
            is_ood = (ood_threshold is not None) and (nearest_dist > ood_threshold)
        except Exception as e:
            st.warning(f"Nearest neighbor lookup failed: {e}")

    # Display nearest planet
    st.markdown("### 🌍 Closest known confirmed exoplanet (from dataset)")
    if nearest_info:
        # try to show a friendly name if present
        planet_name = None
        for candidate_name_field in ["pl_name", "kepoi_name", "pl_hostname", "planet_name", "pl_letter"]:
            if candidate_name_field in nearest_info and pd.notna(nearest_info[candidate_name_field]):
                planet_name = str(nearest_info[candidate_name_field])
                break
        if planet_name is None:
            # choose any column that looks like a name (string, non-numeric)
            for k, v in nearest_info.items():
                if isinstance(v, str) and len(v) > 0 and not v.isnumeric():
                    planet_name = v
                    break
        if planet_name is None:
            planet_name = f"Index {idx0}"
        st.markdown(f"**Closest match:** {planet_name}")
        if similarity is not None:
            st.metric("Similarity to candidate", f"{similarity:.1f}%")
        if nearest_dist is not None:
            st.caption(f"Nearest-neighbor distance: {nearest_dist:.4f}")
        # show a compact table of key params
        compact_keys = [k for k in feature_cols if k in nearest_info][:8]
        compact = {k: nearest_info.get(k, None) for k in compact_keys}
        st.write(compact)
    else:
        st.write("No nearest confirmed planet available in the bundle.")

    if is_ood:
        st.warning("⚠️ This candidate appears to be out-of-distribution compared to known confirmed planets. The model may be unreliable for this input.")

    # -------------------
    # Derived Characteristics & Habitability heuristics
    # -------------------
    st.subheader("🧪 Predicted planetary characteristics (heuristic estimates)")

    eq_temp_input = row.iloc[0].get("eq_temperature_k", np.nan)
    # If eq_temp missing, try to estimate
    if np.isnan(eq_temp_input):
        try:
            st_teff = row.iloc[0].get("stellar_teff_k", np.nan)
            st_rad = row.iloc[0].get("stellar_radius_solar", np.nan)
            semimaj = a
            est_teq = equilibrium_temperature(st_teff, st_rad, semimaj)
            eq_temp_input = est_teq
        except Exception:
            eq_temp_input = np.nan

    # density & gravity if mass+radius provided
    mass_me = row.iloc[0].get("planet_mass_me", np.nan)
    den_text = "Unknown"; grav_text = "Unknown"
    if not np.isnan(mass_me) and not np.isnan(radius_val_float) and radius_val_float > 0:
        try:
            mass_kg = float(mass_me) * M_earth_kg
            r_m = float(radius_val_float) * R_earth_m
            volume = 4.0 / 3.0 * math.pi * (r_m ** 3)
            density = mass_kg / volume  # kg/m3
            density_gcc = density / 1000.0  # g/cm3
            den_text = f"{density_gcc:.2f} g/cm³"
            # gravity
            g = G * mass_kg / (r_m ** 2)
            grav_text = f"{g:.2f} m/s²"
        except Exception:
            pass

    # Habitability zone & water
    water_possible = False
    hab_zone_text = "Unknown"
    if not np.isnan(eq_temp_input):
        T = float(eq_temp_input)
        if 240 <= T <= 320:
            hab_zone_text = "Habitable Zone"
            water_possible = True
        elif T > 320:
            hab_zone_text = "Hot Zone"
            water_possible = False
        else:
            hab_zone_text = "Frozen Zone"
            water_possible = False

    # Atmosphere guess
    atmosphere_guess = "Unknown"
    if not np.isnan(radius_val_float):
        if radius_val_float < 1.5:
            atmosphere_guess = "Nitrogen-Oxygen (likely rocky)"
        elif radius_val_float < 3.0:
            atmosphere_guess = "CO₂-rich / Dense (Super-Earth / mini-Neptune)"
        else:
            atmosphere_guess = "Hydrogen-Helium dominated (gas giant)"

    # Molecule guesses
    molecules = guess_molecules(eq_temp_input)

    # Life probability heuristic
    snr_val = row.iloc[0].get("model_snr", None)
    life_score = life_probability_score(water_possible, hab_zone_text, atmosphere_guess, snr=snr_val)

    # Exploration priority derived (simple rule)
    exploration_priority = "Low"
    if life_score >= 70:
        exploration_priority = "High"
    elif life_score >= 40:
        exploration_priority = "Medium"

    # Display nicely
    colA, colB = st.columns(2)
    with colA:
        st.metric("🌡 Equilibrium Temperature (K)", f"{eq_temp_input:.1f}" if not np.isnan(eq_temp_input) else "Unknown")
        st.metric("⚖️ Density", den_text)
        st.metric("🌀 Surface gravity", grav_text)
    with colB:
        st.metric("🌊 Liquid water possible", "Yes" if water_possible else "No")
        st.metric("☁️ Atmosphere guess", atmosphere_guess)
        st.metric("🧬 Life probability (heuristic)", f"{life_score}%")

    st.markdown(f"**Likely molecules in atmosphere:** {', '.join(molecules)}")
    st.markdown(f"**Habitability zone:** {hab_zone_text} — **Exploration priority:** **{exploration_priority}**")

    # Radar / Comparison chart (candidate vs nearest) for core features (if nearest exists)
    try:
        if nearest_info is not None:
            radar_feats = ["planet_radius_re", "planet_mass_me", "eq_temperature_k", "insolation_flux"]
            cand_vals = [float(row.iloc[0].get(f, np.nan) if not pd.isna(row.iloc[0].get(f, np.nan)) else 0.0) for f in radar_feats]
            nearest_vals = [float(nearest_info.get(f, np.nan) if not pd.isna(nearest_info.get(f, np.nan)) else 0.0) for f in radar_feats]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=cand_vals, theta=radar_feats, fill="toself", name="Candidate"))
            fig_r.add_trace(go.Scatterpolar(r=nearest_vals, theta=radar_feats, fill="toself", name="Nearest"))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
            st.plotly_chart(fig_r, use_container_width=True)
    except Exception:
        pass

    # -------------------------
    # PDF Report generation
    # -------------------------
    try:
        if st.button("📄 Generate PDF Report"):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []
            elements.append(Paragraph("Exoplanet Detection Report", styles["Title"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Suggested name: {name}", styles["Normal"]))
            elements.append(Paragraph(f"Exoplanet probability: {ensemble_prob*100:.2f}%", styles["Normal"]))
            elements.append(Paragraph(f"Category: {cat_label}", styles["Normal"]))
            elements.append(Paragraph(f"Habitability zone: {hab_zone_text}", styles["Normal"]))
            elements.append(Paragraph(f"Liquid water possible: {'Yes' if water_possible else 'No'}", styles["Normal"]))
            elements.append(Paragraph(f"Atmosphere guess: {atmosphere_guess}", styles["Normal"]))
            elements.append(Paragraph(f"Likely molecules: {', '.join(molecules)}", styles["Normal"]))
            elements.append(Paragraph(f"Life probability (heuristic): {life_score}%", styles["Normal"]))
            elements.append(Spacer(1, 12))
            # Input table
            data = [["Parameter", "Value"]]
            for k, v in row.iloc[0].to_dict().items():
                data.append([k, str(v)])
            t = Table(data)
            elements.append(t)
            doc.build(elements)
            st.download_button("⬇️ Download PDF report", buffer.getvalue(), file_name="exoplanet_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF generation error: {e}")

# Footer
st.markdown("---")
st.caption(f"Model bundle meta: {meta}")
