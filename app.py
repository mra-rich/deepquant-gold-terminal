import streamlit as st
import joblib
import pandas as pd
import pandas_ta as ta
import os
from datetime import datetime

# --- CONFIG ---
MODEL_PATH = "DeepQuant_XAUUSD_Model.pkl"
DATA_PATH = "XAUUSD_MT5_4H.csv" 
LOG_FILE = "session_log.csv" # File untuk menyimpan history permanen

st.set_page_config(page_title="DeepQuant Terminal", page_icon="📈", layout="wide")

# 1. LOAD ASSETS (OTAK AI & DATABASE)
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        return None, None
    model_package = joblib.load(MODEL_PATH)
    master_df = pd.read_csv(DATA_PATH)
    master_df.columns = master_df.columns.str.replace('[<>]', '', regex=True).str.lower().str.strip()
    return model_package, master_df

package, master_df = load_assets()

# --- FUNGSI PENYIMPANAN PERMANEN ---
def save_to_log(data_entry):
    if not os.path.exists(LOG_FILE):
        df_log = pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close', 'Confidence', 'Signal'])
    else:
        df_log = pd.read_csv(LOG_FILE)
    
    new_df = pd.DataFrame([data_entry])
    df_log = pd.concat([new_df, df_log], ignore_index=True)
    df_log.to_csv(LOG_FILE, index=False)

def get_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close', 'Confidence', 'Signal'])

# --- UI ---
st.title("💰 DeepQuant Institutional Command Center")

if package is None:
    st.error("File Model (.pkl) atau Data (.csv) tidak ditemukan di GitHub!")
else:
    col_input, col_stats = st.columns([1, 2])
    
    with col_input:
        st.subheader("📝 Input Candle 4H")
        with st.form("input_form", clear_on_submit=True):
            n_open = st.number_input("Open Price", format="%.2f")
            n_high = st.number_input("High Price", format="%.2f")
            n_low = st.number_input("Low Price", format="%.2f")
            n_close = st.number_input("Close Price", format="%.2f")
            submitted = st.form_submit_button("ANALISA SEKARANG")

    if submitted:
        # 2. LOGIKA HITUNG (SELARAS 100% DENGAN RISET)
        new_row = pd.DataFrame([{'open': n_open, 'high': n_high, 'low': n_low, 'close': n_close}])
        combined = pd.concat([master_df.tail(100), new_row], ignore_index=True)
        
        # Indikator (DNA Riset)
        combined['atr'] = ta.atr(combined['high'], combined['low'], combined['close'], length=14)
        combined['atr_ma'] = ta.sma(combined['atr'], length=50)
        adx_df = ta.adx(combined['high'], combined['low'], combined['close'], length=14)
        combined['adx_real'] = adx_df.iloc[:, 0]
        combined['adx'] = adx_df.iloc[:, 0] / 100.0
        combined['atr_rel'] = combined['atr'] / combined['close']
        combined['hour'] = datetime.now().hour
        combined['ret_1'] = combined['close'].pct_change(1)
        combined['ret_3'] = combined['close'].pct_change(3)
        range_l = combined['high'] - combined['low']
        combined['upper_wick'] = (combined['high'] - combined[['close', 'open']].max(axis=1)) / (range_l + 1e-6)
        combined['lower_wick'] = (combined[['close', 'open']].min(axis=1) - combined['low']) / (range_l + 1e-6)
        
        last_row = combined.iloc[-1:]
        
        # 3. PREDIKSI MODEL PKL
        prob_buy = package['model'].predict_proba(last_row[package['features']].values)[:, 1][0]
        
        # FILTERS KEAMANAN
        is_volatile = last_row['atr'].values[0] > last_row['atr_ma'].values[0]
        is_trending = last_row['adx_real'].values[0] > 20
        
        final_signal = "WAIT"
        if prob_buy > 0.53 and is_volatile and is_trending:
            final_signal = "BUY"

        # 4. SIMPAN KE LOG PERMANEN
        entry = {
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'Open': n_open, 'High': n_high, 'Low': n_low, 'Close': n_close,
            'Confidence': f"{prob_buy*100:.1f}%",
            'Signal': final_signal
        }
        save_to_log(entry)
        st.success(f"Sinyal Terkalkulasi: {final_signal}")

    # 5. TAMPILKAN HISTORY DARI FILE (ANTI-REFRESH)
    with col_stats:
        st.subheader("📜 History Sinyal Terakhir")
        history_df = get_logs()
        if not history_df.empty:
            def color_signal(val):
                color = '#2ecc71' if val == 'BUY' else '#95a5a6'
                return f'color: {color}; font-weight: bold'
            st.table(history_df.style.applymap(color_signal, subset=['Signal']))
            
            if st.button("Hapus Semua History"):
                if os.path.exists(LOG_FILE):
                    os.remove(LOG_FILE)
                    st.rerun()
        else:
            st.info("Menunggu input data pertama...")
