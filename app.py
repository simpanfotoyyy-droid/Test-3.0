from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================================
# KONFIGURASI PARAMETER (30% UPSIDE CONSOLIDATION SQUEEZE)
# ==========================================================
PERIOD = "1y"
DONCHIAN_LONG_PERIOD = 50     # Menentukan resistance besar (Upper Channel)
ATR_PERIOD = 14
MIN_UPSIDE_PCT = 30.0         # Minimal ruang kenaikan ke Upper Channel 30%
MAX_VOLATILITY_PCT = 5.0      # Konsolidasi ketat: Rentang rata-rata candle 5 hari terakhir sempit
MIN_VALUE = 100_000_000       # Minimal nilai transaksi harian Rp100 juta
MAX_WORKERS = 12
OUTPUT_FILE = "hasil_scanner_konsolidasi_30pct_upside.csv"

# Daftar lengkap seluruh emiten IDX (900+ saham)
RAW_TICKERS = [
    "ROTI", "MGLV", "TRUE", "LABA", "ARCI", "IPAC", "MASB", "BMHS", "FLMC", "NICL", "UVCR", "BUKA", "HAIS", "OILS", "GPSO", "MCOL", "RSGK", "RUNS", "SBMA", "CMNT", "GTSI", "IDEA", "KUAS", "BOBA", "MTEL", "DEPO", "BINO", "CMRY", "WGSH", "TAYS", "WMPP", "RMKE", "OBMD", "AVIA", "IPPE", "NASI", "BSML", "DRMA", "ADMR", "SEMA", "ASLC", "NETV", "BAUT", "ENAK", "NTBK", "SMKM", "STAA", "NANO", "BIKE", "WIRG", "SICO", "GOTO", "TLDN", "MTMH", "WINR", "IBOS", "OLIV", "ASHA", "SWID", "TRGU", "ARKO", "CHEM", "DEWI", "AXIO", "KRYA", "HATM", "RCCC", "GULA", "JARR", "AMMS", "RAFI", "KKES", "ELPI", "EURO", "KLIN", "TOOL", "BUAH", "CRAB", "MEDS", "COAL", "PRAY", "CBUT", "BELI", "MKTR", "OMED", "BSBK", "PDPP", "KDTN", "ZATA", "NINE", "MMIX", "PADA", "ISAP", "VTNY", "SOUL", "ELIT", "BEER", "CBPE", "SUNI", "CBRE", "WINE", "BMBL", "PEVE", "LAJU", "FWCT", "NAYZ", "IRSX", "PACK", "VAST", "CHIP", "HALO", "KING", "PGEO", "FUTR", "HILL", "BDKR", "PTMP", "SAGE", "TRON", "CUAN", "NSSS", "GTRA", "HAJJ", "PIPA", "NCKL", "MENN", "AWAN", "MBMA", "RAAM", "DOOH", "JATI", "TYRE", "MPXL", "SMIL", "KLAS", "MAXI", "VKTR", "RELF", "AMMN", "CRSN", "GRPM", "WIDI", "TGUK", "INET", "MAHA", "RMKO", "CNMA", "FOLK", "HBAT", "GRIA", "PPRI", "ERAL", "CYBR", "MUTU", "LMAX", "HUMI", "MSIE", "RSCH", "BABY", "AEGS", "IOTF", "KOCI", "PTPS", "BREN", "STRK", "KOKA", "LOPI", "UDNG", "RGAS", "MSTI", "IKPM", "AYAM", "SURI", "ASLI", "CGAS", "NICE", "MSJA", "SMLE", "ACRO", "MANG", "GRPH", "SMGA", "UNTD", "TOSK", "MPIX", "ALII", "MKAP", "MEJA", "LIVE", "HYGN", "BAIK", "VISI", "AREA", "MHKI", "ATLA", "DATA", "SOLA", "BATR", "SPRE", "PART", "GOLF", "ISEA", "BLES", "GUNA", "LABS", "DOSS", "NEST", "PTMR", "VERN", "DAAZ", "BOAT", "NAIK", "AADI", "MDIY", "KSIX", "RATU", "YOII", "HGII", "BRRC", "DGWG", "CBDK", "OBAT", "MINE", "KAQI", "YUPI", "FORE", "MDLA", "DKHH", "ASPR", "PSAT", "COIN", "CDIA", "BLOG", "MERI", "CHEK", "PMUI", "EMAS", "PJHB", "RLCO", "SUPA", "WBSA", "JECX", "JELI", "EMMI", "BACH", "PRDL", "RANS", "AALI", "ABBA", "ABDA", "ABMM", "ACES", "ACST", "ADES", "ADHI", "ADMF", "ADMG", "ADRO", "AGII", "AGRO", "AGRS", "AHAP", "CASH", "TECH", "BSSR", "BSWD", "BTEK", "BTEL", "BTON", "BTPN", "BUDI", "AIMS", "AISA", "AKKU", "AKPI", "AKRA", "AKSI", "ALDO", "ALKA", "ALMI", "ALTO", "AMAG", "AMFG", "AMIN", "AMRT", "ANJT", "ANTM", "APEX", "APIC", "APII", "APLI", "APLN", "ARGO", "ARII", "ARNA", "ARTA", "ARTI", "ARTO", "ASBI", "ASDM", "ASGR", "ASII", "ASJT", "ASMI", "ASRI", "ASRM", "ASSA", "ATIC", "AUTO", "BABP", "BACA", "BAJA", "BALI", "BAPA", "BATA", "BAYU", "BBCA", "BBHI", "BBKP", "BBLD", "BBMD", "BBNI", "BBRI", "BBRM", "BBTN", "BBYB", "BCAP", "BCIC", "BCIP", "BDMN", "BEKS", "BEST", "BFIN", "BGTG", "BHIT", "BIKA", "BIMA", "BINA", "BIPI", "BIPP", "BIRD", "BISI", "BJBR", "BJTM", "BKDP", "BKSL", "BKSW", "BLTA", "BLTZ", "BMAS", "BMRI", "BMSR", "BMTR", "BNBA", "BNBR", "BNGA", "BNII", "BNLI", "BOLT", "BPFI", "BPII", "BRAM", "BRMS", "BRNA", "BRPT", "BSDE", "BSIM", "BUKK", "BULL", "BUMI", "BUVA", "BVIC", "BWPT", "BYAN", "CANI", "CASS", "CEKA", "CENT", "CFIN", "CINT", "CITA", "CLPI", "CMNP", "CMPP", "CNKO", "COWL", "CPIN", "CPRO", "CSAP", "CTBN", "CTRA", "CTTH", "DART", "DEFI", "DEWA", "DGIK", "DILD", "DKFT", "DLTA", "DMAS", "DNAR", "DNET", "DOID", "DPNS", "DSFI", "DSNG", "DSSA", "DUTI", "DVLA", "DYAN", "ECII", "EKAD", "ELSA", "ELTY", "EMDE", "EMTK", "ENRG", "EPMT", "ERAA", "ERTX", "ESSA", "ESTI", "ETWA", "EXCL", "FAST", "FASW", "FISH", "FMII", "FORU", "FPNI", "GAMA", "GDST", "GDYR", "GEMA", "GEMS", "GGRM", "GIAA", "GJTL", "GLOB", "GMTD", "GOLD", "GOLL", "GPRA", "GSMF", "GTBO", "GWSA", "GZCO", "HADE", "HDFA", "HERO", "HEXA", "HITS", "HMSP", "HOME", "HOTL", "HRUM", "IATA", "IBFN", "IBST", "ICBP", "ICON", "IGAR", "IIKP", "IKAI", "IKBI", "IMAS", "IMJS", "IMPC", "INAF", "INAI", "INCI", "INCO", "INDF", "INDR", "INDS", "INDX", "INDY", "INKP", "INPC", "INPP", "INRU", "INTA", "INTD", "INTP", "IPOL", "ISAT", "ISSP", "ITMA", "ITMG", "JAWA", "JECC", "JIHD", "JKON", "JPFA", "JRPT", "JSMR", "JSPT", "JTPE", "KAEF", "KARW", "KBLI", "KBLM", "KBLV", "KBRI", "KDSI", "KIAS", "KICI", "KIJA", "KKGI", "KLBF", "KOBX", "KOIN", "KONI", "KOPI", "KPIG", "KRAS", "KREN", "LAPD", "LCGP", "LEAD", "LINK", "LION", "LMAS", "LMPI", "LMSH", "LPCK", "LPGI", "LPIN", "LPKR", "LPLI", "LPPF", "LPPS", "LRNA", "LSIP", "LTLS", "MAGP", "MAIN", "MAPI", "MAYA", "MBAP", "MBSS", "MBTO", "MCOR", "MDIA", "MDKA", "MDLN", "MDRN", "MEDC", "MEGA", "MERK", "META", "MFMI", "MGNA", "MICE", "MIDI", "MIKA", "MIRA", "MITI", "MKPI", "MLBI", "MLIA", "MLPL", "MLPT", "MMLP", "MNCN", "MPMX", "MPPA", "MRAT", "MREI", "MSKY", "MTDL", "MTFN", "MTLA", "MTSM", "MYOH", "MYOR", "MYTX", "NELY", "NIKL", "NIRO", "NISP", "NOBU", "NRCA", "OCAP", "OKAS", "OMRE", "PADI", "PALM", "PANR", "PANS", "PBRX", "PDES", "PEGE", "PGAS", "PGLI", "PICO", "PJAA", "PKPK", "PLAS", "PLIN", "PNBN", "PNBS", "PNIN", "PNLF", "PNSE", "POLY", "POOL", "PPRO", "PSAB", "PSDN", "PSKT", "PTBA", "PTIS", "PTPP", "PTRO", "PTSN", "PTSP", "PUDP", "PWON", "PYFA", "RAJA", "RALS", "RANC", "RBMS", "RDTX", "RELI", "RICY", "RIGS", "RIMO", "RODA", "ROTI", "RUIS", "SAFE", "SAME", "SCCO", "SCMA", "SCPI", "SDMU", "SDPC", "SDRA", "SGRO", "SHID", "SIDO", "SILO", "SIMA", "SIMP", "SIPD", "SKBM", "SKLT", "SKYB", "SMAR", "SMBR", "SMCB", "SMDM", "SMDR", "SMGR", "SMMA", "SMMT", "SMRA", "SMRU", "SMSM", "SOCI", "SONA", "SPMA", "SQMI", "SRAJ", "SRIL", "SRSN", "SRTG", "SSIA", "SSMS", "SSTM", "STAR", "STTP", "SUGI", "SULI", "SUPR", "TALF", "TARA", "TAXI", "TBIG", "TBLA", "TBMS", "TCID", "TELE", "TFCO", "TGKA", "TIFA", "TINS", "TIRA", "TIRT", "TKIM", "TLKM", "TMAS", "TMPO", "TOBA", "TOTL", "TOTO", "TOWR", "TPIA", "TPMA", "TRAM", "TRIL", "TRIM", "TRIO", "TRIS", "TRST", "TRUS", "TSPC", "ULTJ", "UNIC", "UNIT", "UNSP", "UNTR", "UNVR", "VICO", "VINS", "VIVA", "VOKS", "VRNA", "WAPO", "WEHA", "WICO", "WIIM", "WIKA", "WINS", "WOMF", "WSKT", "WTON", "YPAS", "YULE", "ZBRA", "SHIP", "CASA", "DAYA", "DPUM", "IDPR", "JGLE", "KINO", "MARI", "MKNT", "MTRA", "OASA", "POWR", "INCF", "WSBP", "PBSA", "PRDA", "BOGA", "BRIS", "PORT", "CARS", "MINA", "CLEO", "TAMU", "CSIS", "TGRA", "FIRE", "TOPS", "KMTR", "ARMY", "MAPB", "WOOD", "HRTA", "MABA", "HOKI", "MPOW", "MARK", "NASA", "MDKI", "BELL", "KIOS", "GMFI", "MTWI", "ZINC", "MCAS", "PPRE", "WEGE", "PSSI", "MORA", "DWGL", "PBID", "JMAS", "CAMP", "IPCM", "PCAR", "LCKM", "BOSS", "HELI", "JSKY", "INPS", "GHON", "TDPM", "DFAM", "NICK", "BTPS", "SPTO", "PRIM", "HEAL", "TRUK", "PZZA", "TUGU", "MSIN", "SWAT", "TNCA", "MAPA", "TCPI", "IPCC", "RISE", "BPTR", "POLL", "NFCX", "MGRO", "NUSA", "FILM", "ANDI", "LAND", "MOLI", "PANI", "DIGI", "CITY", "SAPX", "SURE", "HKMU", "MPRO", "DUCK", "GOOD", "SKRN", "YELO", "CAKK", "SATU", "SOSS", "DEAL", "POLA", "DIVA", "LUCK", "URBN", "SOTS", "ZONE", "PEHA", "FOOD", "BEEF", "POLI", "CLAY", "NATO", "JAYA", "COCO", "MTPS", "CPRI", "HRME", "POSA", "JAST", "FITT", "BOLA", "CCSI", "SFAN", "POLU", "KJEN", "KAYU", "ITIC", "PAMG", "IPTV", "BLUE", "ENVY", "EAST", "LIFE", "FUJI", "KOTA", "INOV", "ARKA", "SMKL", "HDIT", "KEEN", "BAPI", "TFAS", "GGRP", "OPMS", "NZIA", "SLIS", "PURE", "IRRA", "DMMX", "SINI", "WOWS", "ESIP", "TEBE", "KEJU", "PSGO", "AGAR", "IFSH", "REAL", "IFII", "PMJS", "UCID", "GLVA", "PGJO", "AMAR", "CSRA", "INDO", "AMOR", "TRIN", "DMND", "PURA", "PTPW", "TAMA", "IKAN", "AYLS", "DADA", "ASPI", "ESTA", "BESS", "AMAN", "CARE", "SAMF", "SBAT", "KBAG", "CBMF", "RONY", "CSMI", "BBSS", "BHAT", "EPAC", "UANG", "PGUN", "SOFA", "PPGL", "TOYS", "SGER", "TRJA", "PNGO", "SCNP", "BBSI", "KMDS", "PURI", "SOHO", "HOMI", "ROCK", "ENZO", "PLAN", "PTDU", "ATAP", "VICI", "PMMP", "WIFI", "FAPA", "DCII", "KETR", "DGNS", "UFOE", "BANK", "WMUU", "EDGE", "UNIQ", "BEBS", "SNLK", "ZYRX", "LFLO", "FIMP", "TAPG", "NPGF", "LUCY", "ADCP", "HOPE"
]

TICKERS = sorted(list(set([f"{t}.JK" if not t.endswith(".JK") else t for t in RAW_TICKERS])))

def scan_ticker(ticker):
    symbol = ticker.replace(".JK", "")
    try:
        df = yf.download(
            ticker,
            period=PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty or len(df) < DONCHIAN_LONG_PERIOD + 10:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        clean_df = pd.DataFrame(index=df.index)
        for col in required:
            if col in df.columns:
                s = df[col]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                clean_df[col] = pd.to_numeric(s, errors="coerce")
            else:
                return None

        clean_df = clean_df.dropna().sort_index()
        clean_df = clean_df[clean_df["Volume"] >= 0]
        clean_df = clean_df[~clean_df.index.duplicated(keep="last")]

        if len(clean_df) < DONCHIAN_LONG_PERIOD + 10:
            return None

        close = clean_df["Close"]
        high = clean_df["High"]
        low = clean_df["Low"]
        volume = clean_df["Volume"]

        # ==========================================
        # 1. HITUNG DONCHIAN UPPER 50 (Resistance Luas)
        # ==========================================
        dc_upper_50 = high.rolling(DONCHIAN_LONG_PERIOD).max().shift(1)
        
        # Hitung Jarak Upside Persentase dari Close ke Upper Donchian 50
        # Rumus: ((DC_Upper_50 - Close) / Close) * 100
        upside_room_pct = ((dc_upper_50 - close) / close) * 100

        # ==========================================
        # 2. HITUNG KONSOLIDASI SEMPIT (Candle Range / ATR)
        # ==========================================
        # Rentang harian (High - Low) / Close dalam persentase
        daily_range_pct = ((high - low) / close) * 100
        # Rata-rata rentang 5 hari terakhir (menunjukkan konsolidasi rapat/rendah volatilitas)
        avg_range_5d = daily_range_pct.rolling(5).mean()

        # ==========================================
        # 3. PROXY AKUMULASI BANDAR (Top Broker Hist / Money Flow)
        # ==========================================
        high_low_range = (high - low).replace(0, 1)
        clv = ((close - low) - (high - close)) / high_low_range
        top_broker_hist = clv * volume
        sum_tb_10 = top_broker_hist.rolling(10).sum()

        clean_df["DC_Upper_50"] = dc_upper_50
        clean_df["Upside_Room"] = upside_room_pct
        clean_df["Avg_Range_5d"] = avg_range_5d
        clean_df["Sum_TB_10"] = sum_tb_10

        last = clean_df.iloc[-1]

        last_close = float(last["Close"])
        upper_50 = float(last["DC_Upper_50"])
        upside = float(last["Upside_Room"])
        tightness = float(last["Avg_Range_5d"])
        acc_10 = float(last["Sum_TB_10"])
        trans_value = last_close * float(last["Volume"])

        # ==========================================
        # KETENTUAN PENYARINGAN (SCREENER RULES)
        # ==========================================
        # A. Ruang Upside ke Resistance Utama minimal 30%
        cond_upside = upside >= MIN_UPSIDE_PCT

        # B. Konsolidasi Ketat: Rata-rata rentang candle 5 hari terakhir sangat sempit (<= 5%)
        cond_squeeze = tightness <= MAX_VOLATILITY_PCT

        # C. Akumulasi Bandar 10 hari terakhir bernilai positif (tidak sedang distribusi)
        cond_accumulation = acc_10 > 0

        # D. Filter Likuiditas Minimal Transaksi Harian
        cond_liquidity = trans_value >= MIN_VALUE

        if not (cond_upside and cond_squeeze and cond_accumulation and cond_liquidity):
            return None

        timestamp = pd.Timestamp(clean_df.index[-1]).strftime("%Y-%m-%d")

        return {
            "symbol": symbol,
            "date": timestamp,
            "close": round(last_close, 2),
            "upper_resistance": round(upper_50, 2),
            "potensi_upside_%": round(upside, 2),
            "keketatan_candle_%": round(tightness, 2),
            "est_transaksi_rp": int(trans_value),
        }

    except Exception:
        return None

def run_scanner():
    print("=" * 70)
    print(" 🚀 BEI 30% UPSIDE CONSOLIDATION SQUEEZE SCANNER (DAILY) 🚀")
    print("=" * 70)
    print(f"Total Ticker Dimuat : {len(TICKERS)} Emiten")
    print(f"Memindai konsolidasi sempit dengan potensi naik >= {MIN_UPSIDE_PCT}%...\n")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scan_ticker, t): t for t in TICKERS}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    results.append(res)
                    print(f"🎯 [POTENSI 30%+] {res['symbol']} | Close={res['close']} | Upside={res['potensi_upside_%']}% | Squeeze={res['keketatan_candle_%']}%")
            except Exception:
                pass

    print("\n" + "=" * 55 + " 🏆 HASIL SCREENING KONSOLIDASI 30% UPSIDE 🏆 " + "=" * 55)
    if results:
        output = pd.DataFrame(results)
        # Urutkan berdasarkan potensi upside terbesar dan keketatan candle terrapat
        output = output.sort_values(["potensi_upside_%", "keketatan_candle_%"], ascending=[False, True]).reset_index(drop=True)
        
        ranks = [f"🥇 1 (Top Target)" if i == 0 else f"🥈 2" if i == 1 else f"🥉 3" if i == 2 else f"🔥 {i+1}" for i in range(len(output))]
        output.insert(0, "Ranking", ranks)
        
        print(output.to_string(index=False))
        output.to_csv(OUTPUT_FILE, index=False)
        print(f"\n📁 Hasil screening berhasil disimpan ke: {OUTPUT_FILE}")
    else:
        print("❌ Belum ada emiten yang memenuhi kriteria konsolidasi siap terbang 30% saat ini.")

    print(f"\nSelesai dalam {time.time() - start_time:.1f} detik.")

if __name__ == "__main__":
    run_scanner()
