import streamlit as st
import joblib
import pandas as pd
import pandas_ta as ta
import os
from datetime import datetime

# --- CONFIG ---
MODEL_PATH = "DeepQuant_XAUUSD_Model.pkl"
DATA_PATH = "XAUUSD_MT5_4H.csv" 

st.set_page_config(page_title="DeepQuant Hybrid", page_icon="📈")

# 1. LOAD ASSETS
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        return None, None
    model_package = joblib.load(MODEL_PATH)
    master_df = pd.read_csv(DATA_PATH)
    master_df.columns = master_df.columns.str.replace('[<>]', '', regex=True).str.lower().str.strip()
    return model_package, master_df

package, master_df = load_assets()

# --- UI ---
st.title("💰 DeepQuant Hybrid Terminal")

if package is None:
    st.error("File Model (.pkl) atau Data (.csv) tidak ditemukan di GitHub!")
else:
    # 2. INPUT MANUAL
    st.subheader("📝 Input Candle 4H Terbaru")
    with st.form("input_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1: n_open = st.number_input("Open", format="%.2f")
        with col2: n_high = st.number_input("High", format="%.2f")
        with col3: n_low = st.number_input("Low", format="%.2f")
        with col4: n_close = st.number_input("Close", format="%.2f")
        submitted = st.form_submit_button("Update & Predict")

    if submitted:
        # 3. LOGIKA HITUNG (DNA RISET 416%)
        new_row = pd.DataFrame([{'open': n_open, 'high': n_high, 'low': n_low, 'close': n_close}])
        # Ambil 100 baris terakhir untuk kalkulasi indikator
        combined = pd.concat([master_df.tail(100), new_row], ignore_index=True)
        
        # Indikator
        combined['atr'] = ta.atr(combined['high'], combined['low'], combined['close'], length=14)
        combined['atr_ma'] = ta.sma(combined['atr'], length=50)
        combined['atr_rel'] = combined['atr'] / combined['close']
        adx_df = ta.adx(combined['high'], combined['low'], combined['close'], length=14)
        combined['adx_real'] = adx_df.iloc[:, 0]
        combined['adx'] = adx_df.iloc[:, 0] / 100.0
        combined['hour'] = datetime.now().hour
        combined['ret_1'] = combined['close'].pct_change(1)
        combined['ret_3'] = combined['close'].pct_change(3)
        
        range_l = combined['high'] - combined['low']
        combined['upper_wick'] = (combined['high'] - combined[['close', 'open']].max(axis=1)) / (range_l + 1e-6)
        combined['lower_wick'] = (combined[['close', 'open']].min(axis=1) - combined['low']) / (range_l + 1e-6)
        
        last_row = combined.iloc[-1:]
        
        # 4. PREDIKSI
        prob_buy = package['model'].predict_proba(last_row[package['features']].values)[:, 1][0]
        
        # FILTER SAFETY
        is_volatile = last_row['atr'].values[0] > last_row['atr_ma'].values[0]
        is_trending = last_row['adx_real'].values[0] > 20

        st.divider()
        if prob_buy > 0.53 and is_volatile and is_trending:
            st.success(f"### 🚀 SIGNAL BUY VALID ({prob_buy*100:.1f}%)")
            st.balloons()
        else:
            st.error(f"### 😴 NO TRADE ({prob_buy*100:.1f}%)")
            if not is_volatile: st.warning("Market sedang Low Volatility (ATR < ATR MA)")
            if not is_trending: st.warning("Market No Trend (ADX < 20)")
