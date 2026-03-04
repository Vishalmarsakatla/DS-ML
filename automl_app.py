import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, io, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, r2_score,
    mean_squared_error, classification_report, confusion_matrix,
    silhouette_score, davies_bouldin_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AutoML Studio", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{background:#060810;color:#d4d8f0;font-family:'Outfit',sans-serif}
[data-testid="stAppViewContainer"]{background:#060810;
  background-image:radial-gradient(ellipse 60% 40% at 20% 0%,rgba(0,200,150,.07) 0%,transparent 60%),
                   radial-gradient(ellipse 50% 30% at 80% 100%,rgba(0,120,255,.07) 0%,transparent 60%)}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none}
.block-container{padding:2rem 2.5rem 4rem;max-width:1200px;margin:auto}

/* hero */
.hero{padding:3rem 0 1.5rem;display:flex;align-items:center;gap:2rem;
      border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:2.5rem}
.hero-icon{font-size:3.5rem;background:linear-gradient(135deg,#00c896,#0078ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;flex-shrink:0}
.hero h1{font-family:'Outfit',sans-serif;font-size:2.4rem;font-weight:800;
  background:linear-gradient(135deg,#fff 0%,#00c896 60%,#0078ff 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1}
.hero p{font-size:.95rem;color:#6b7280;font-weight:300;margin-top:.4rem}

/* pipeline bar */
.pipeline{display:flex;margin-bottom:2.5rem;background:rgba(255,255,255,.02);
  border:1px solid rgba(255,255,255,.06);border-radius:16px;overflow:hidden}
.pipe-step{flex:1;padding:.8rem;text-align:center;border-right:1px solid rgba(255,255,255,.06)}
.pipe-step:last-child{border-right:none}
.pipe-step.active{background:rgba(0,200,150,.08)}
.pipe-step.done{background:rgba(0,200,150,.04)}
.pipe-num{font-family:'Space Mono',monospace;font-size:.6rem;color:#374151;margin-bottom:.2rem}
.pipe-step.active .pipe-num,.pipe-step.done .pipe-num{color:#00c896}
.pipe-label{font-size:.72rem;font-weight:600;color:#4b5563}
.pipe-step.active .pipe-label{color:#d4d8f0}
.pipe-step.done .pipe-label{color:#6b7280}

/* cards */
.card{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07);
      border-radius:16px;padding:1.5rem;margin-bottom:1.5rem}
.card-title{font-family:'Space Mono',monospace;font-size:.7rem;color:#00c896;
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:1rem}

/* section divider inside card */
.section-sep{height:1px;background:linear-gradient(90deg,transparent,rgba(0,200,150,.25),transparent);margin:1.5rem 0}
.section-head{font-family:'Space Mono',monospace;font-size:.65rem;color:#6b7280;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:.8rem}

/* metrics */
.metrics-row{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem}
.metric-box{flex:1;min-width:110px;background:rgba(0,200,150,.06);
  border:1px solid rgba(0,200,150,.2);border-radius:12px;padding:1rem;text-align:center}
.metric-val{font-family:'Space Mono',monospace;font-size:1.5rem;font-weight:700;color:#00c896}
.metric-key{font-size:.7rem;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-top:.2rem}

/* model rows */
.model-row{display:flex;align-items:center;gap:1rem;padding:.7rem .5rem;
  border-bottom:1px solid rgba(255,255,255,.04);border-radius:8px}
.model-row.best{background:rgba(0,200,150,.07);border:1px solid rgba(0,200,150,.2);
  border-radius:10px;margin-bottom:4px}
.model-name{flex:2;font-size:.88rem;font-weight:500;color:#d4d8f0}
.model-score{flex:1;font-family:'Space Mono',monospace;font-size:.82rem;color:#00c896;text-align:right}
.model-bar-wrap{flex:3;height:6px;background:rgba(255,255,255,.06);border-radius:10px;overflow:hidden}
.model-bar{height:100%;background:linear-gradient(90deg,#00c896,#0078ff);border-radius:10px}
.best-badge{background:linear-gradient(135deg,#00c896,#0078ff);color:#000;
  font-size:.6rem;font-weight:700;padding:.15rem .5rem;border-radius:50px}
.dl-badge{background:rgba(239,68,68,.2);border:1px solid rgba(239,68,68,.4);
  color:#f87171;font-size:.6rem;font-weight:700;padding:.15rem .5rem;border-radius:50px}
.cluster-badge{background:rgba(251,191,36,.2);border:1px solid rgba(251,191,36,.4);
  color:#fbbf24;font-size:.6rem;font-weight:700;padding:.15rem .5rem;border-radius:50px}
.dr-badge{background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.4);
  color:#818cf8;font-size:.6rem;font-weight:700;padding:.15rem .5rem;border-radius:50px}

/* chips */
.chip{display:inline-block;background:rgba(0,120,255,.1);border:1px solid rgba(0,120,255,.25);
  border-radius:50px;padding:.25rem .75rem;font-size:.78rem;color:#60a5fa;margin:.2rem}
.chip.green{background:rgba(0,200,150,.1);border-color:rgba(0,200,150,.25);color:#00c896}
.chip.red{background:rgba(239,68,68,.1);border-color:rgba(239,68,68,.25);color:#f87171}
.chip.purple{background:rgba(168,85,247,.1);border-color:rgba(168,85,247,.25);color:#c084fc}
.chip.yellow{background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.25);color:#fbbf24}
.chip.indigo{background:rgba(99,102,241,.1);border-color:rgba(99,102,241,.25);color:#818cf8}

/* nlp detect box */
.nlp-detect-box{background:rgba(168,85,247,.06);border:1px solid rgba(168,85,247,.25);
  border-radius:14px;padding:1.2rem 1.5rem;margin:1rem 0}
.nlp-detect-title{font-family:'Space Mono',monospace;font-size:.7rem;color:#c084fc;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem}

/* streamlit overrides */
div[data-testid="stFileUploader"]{background:rgba(0,200,150,.03);
  border:2px dashed rgba(0,200,150,.25);border-radius:16px;padding:1rem}
.stButton>button{background:linear-gradient(135deg,#00c896,#0078ff);color:#000;
  font-family:'Outfit',sans-serif;font-weight:700;font-size:.95rem;border:none;
  border-radius:12px;padding:.65rem 2rem;width:100%;transition:all .25s;
  box-shadow:0 4px 20px rgba(0,200,150,.25)}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(0,200,150,.4)}
.stProgress>div>div{background:linear-gradient(90deg,#00c896,#0078ff)!important}
[data-testid="stExpander"]{background:rgba(255,255,255,.02)!important;
  border:1px solid rgba(255,255,255,.07)!important;border-radius:12px!important}

@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.fadein{animation:fadeIn .5s ease}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">⚡</div>
  <div>
    <h1>AutoML Studio</h1>
    <p>Upload any CSV — EDA · NLP · ML · Deep Learning · Clustering · Dimensionality Reduction · Download</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session defaults ───────────────────────────────────────────────────────
defs = {'step':0,'df':None,'target':None,'results':None,'best_model':None,
        'problem_type':None,'nlp_mode':False,'text_col':None,
        'text_cols_detected':[],'df_proc':None,'scaler':None,
        'feat_cols':[],'X_test':None,'X_test_sc':None,'y_test':None,
        'le_target':None,'tfidf':None,'best_name':'','best_scaled':False,
        'best_is_dl':False,'metric_name':'Accuracy','sort_col':'Accuracy',
        'dl_model':None}
for k,v in defs.items():
    if k not in st.session_state:
        st.session_state[k] = v

step = st.session_state.step

# ── Pipeline indicator ─────────────────────────────────────────────────────
pipe_labels = ["Upload","EDA","Preprocess","Train","Results","Download"]
ph = '<div class="pipeline">'
for i,lbl in enumerate(pipe_labels):
    cls  = "active" if i==step else ("done" if i<step else "")
    icon = "✓" if i<step else str(i+1)
    ph  += f'<div class="pipe-step {cls}"><div class="pipe-num">{icon}</div><div class="pipe-label">{lbl}</div></div>'
ph += '</div>'
st.markdown(ph, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def nav(target_step):
    st.session_state.step = target_step
    st.rerun()

def dark_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w,h), facecolor='#060810')
    ax.set_facecolor('#0d1117')
    return fig, ax

def style_ax(ax):
    ax.tick_params(colors='#6b7280', labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#1f2937')

# ══════════════════════════════════════════════════════════════════════════
# STEP 0 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════
if step == 0:
    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 1 — Upload Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.markdown(f"""
        <div class="metrics-row">
          <div class="metric-box"><div class="metric-val">{df.shape[0]:,}</div><div class="metric-key">Rows</div></div>
          <div class="metric-box"><div class="metric-val">{df.shape[1]}</div><div class="metric-key">Columns</div></div>
          <div class="metric-box"><div class="metric-val">{df.isnull().sum().sum()}</div><div class="metric-key">Missing</div></div>
          <div class="metric-box"><div class="metric-val">{df.dtypes[df.dtypes=='object'].count()}</div><div class="metric-key">Categorical</div></div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        target = st.selectbox("🎯 Select Target Column (or leave for Clustering)", ["-- No target (Clustering only) --"] + df.columns.tolist())
        st.session_state.target = None if target.startswith("--") else target
        if st.button("Continue to EDA →"): nav(1)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — EDA
# ══════════════════════════════════════════════════════════════════════════
elif step == 1:
    df     = st.session_state.df
    target = st.session_state.target
    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 2 — Exploratory Data Analysis</div>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📊 Numeric Summary**")
        if num_cols: st.dataframe(df[num_cols].describe().round(2), use_container_width=True)
        else: st.info("No numeric columns.")
    with c2:
        st.markdown("**❓ Missing Values**")
        miss = df.isnull().sum(); miss = miss[miss>0]
        if len(miss):
            st.dataframe(pd.DataFrame({'Col':miss.index,'#':miss.values,'%':(miss.values/len(df)*100).round(1)}), use_container_width=True)
        else: st.success("✅ No missing values!")

    # detect text cols
    txt_detected = [c for c in cat_cols if c != target and df[c].dropna().apply(lambda x: len(str(x).split())).mean() > 5]
    st.session_state.text_cols_detected = txt_detected
    if txt_detected:
        st.markdown(f'<div class="nlp-detect-box"><div class="nlp-detect-title">🔍 NLP Text Columns Detected</div>'
                    f'<div style="font-size:.85rem;color:#9ca3af;">Found <strong style="color:#c084fc">'
                    f'{len(txt_detected)}</strong> long-text column(s) suitable for TF-IDF.</div></div>', unsafe_allow_html=True)

    if target:
        st.markdown("**🎯 Target Distribution**")
        fig, ax = dark_fig(8,3)
        if df[target].dtype=='object' or df[target].nunique()<=15:
            counts = df[target].value_counts()
            ax.bar(counts.index.astype(str), counts.values,
                   color=['#00c896','#0078ff','#a855f7','#f59e0b','#ef4444'][:len(counts)])
        else:
            ax.hist(df[target].dropna(), bins=30, color='#00c896', edgecolor='#060810', linewidth=0.5)
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    if len(num_cols)>1:
        st.markdown("**🔥 Correlation Heatmap**")
        fig2, ax2 = dark_fig(10,4)
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(150,220,as_cmap=True),
                    annot=len(num_cols)<=10, fmt='.2f', linewidths=0.5, ax=ax2,
                    annot_kws={'size':7}, cbar_kws={'shrink':.8})
        ax2.tick_params(colors='#6b7280', labelsize=7)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    plot_cols = [c for c in num_cols if c!=target][:6]
    if plot_cols:
        st.markdown("**📈 Feature Distributions**")
        fig3, axes = plt.subplots(1, len(plot_cols), figsize=(min(14,len(plot_cols)*2.5),3), facecolor='#060810')
        if len(plot_cols)==1: axes=[axes]
        for ax3,col in zip(axes,plot_cols):
            ax3.set_facecolor('#0d1117')
            ax3.hist(df[col].dropna(), bins=20, color='#0078ff', edgecolor='#060810', linewidth=0.3, alpha=0.8)
            ax3.set_title(col[:12], color='#9ca3af', fontsize=7)
            ax3.tick_params(colors='#4b5563', labelsize=6)
            for sp in ax3.spines.values(): sp.set_color('#1f2937')
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    ca, cb = st.columns(2)
    with ca:
        if st.button("← Back"): nav(0)
    with cb:
        if st.button("Continue to Preprocessing →"): nav(2)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════
elif step == 2:
    df     = st.session_state.df
    target = st.session_state.target
    txt_detected = st.session_state.text_cols_detected

    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 3 — Preprocessing</div>', unsafe_allow_html=True)

    # NLP toggle
    nlp_mode = False; text_col = None
    if txt_detected:
        st.markdown(f'<div class="nlp-detect-box"><div class="nlp-detect-title">🧠 NLP Mode Available</div>'
                    f'<div style="font-size:.85rem;color:#9ca3af;">Enable TF-IDF vectorization on a text column.</div></div>',
                    unsafe_allow_html=True)
        nlp_mode = st.toggle("Enable NLP Mode (TF-IDF)", value=True)
        if nlp_mode:
            text_col = st.selectbox("Confirm text column", txt_detected)
    st.session_state.nlp_mode  = nlp_mode
    st.session_state.text_col  = text_col

    with st.spinner("Auto-preprocessing..."):
        df_proc = df.copy()
        high_miss = [c for c in df_proc.columns if df_proc[c].isnull().mean()>0.5 and c!=target]
        if high_miss: df_proc.drop(columns=high_miss, inplace=True)

        # problem type
        if target:
            if df_proc[target].dtype=='object' or df_proc[target].nunique()<=20:
                problem_type='classification'
                le_t = LabelEncoder()
                df_proc[target] = le_t.fit_transform(df_proc[target].astype(str))
            else:
                problem_type='regression'; le_t=None
        else:
            problem_type='clustering'; le_t=None
        st.session_state.problem_type=problem_type; st.session_state.le_target=le_t

        # encode cats
        cat_cols = df_proc.select_dtypes(include='object').columns.tolist()
        non_txt  = [c for c in cat_cols if c!=target and c!=text_col]
        for col in non_txt:
            le=LabelEncoder(); df_proc[col]=le.fit_transform(df_proc[col].astype(str))

        # impute
        num_cols=[c for c in df_proc.select_dtypes(include=np.number).columns if c!=target]
        if df_proc[num_cols].isnull().sum().sum()>0:
            imp=SimpleImputer(strategy='median'); df_proc[num_cols]=imp.fit_transform(df_proc[num_cols])

        # TF-IDF
        tfidf=None; tfidf_cols=[]
        if nlp_mode and text_col and text_col in df_proc.columns:
            tfidf=TfidfVectorizer(max_features=300, stop_words='english')
            tm=tfidf.fit_transform(df_proc[text_col].fillna('').astype(str))
            tdf=pd.DataFrame(tm.toarray(), columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()], index=df_proc.index)
            df_proc=pd.concat([df_proc.drop(columns=[text_col]),tdf],axis=1)
            tfidf_cols=list(tdf.columns)
        st.session_state.tfidf=tfidf; st.session_state.df_proc=df_proc

    chips=[f'<span class="chip green">✓ Problem: {problem_type.title()}</span>',
           f'<span class="chip green">✓ {len(non_txt)} categorical cols encoded</span>',
           f'<span class="chip green">✓ Missing imputed (median)</span>']
    if high_miss: chips.append(f'<span class="chip red">✗ {len(high_miss)} high-missing cols dropped</span>')
    if nlp_mode and text_col: chips.append(f'<span class="chip purple">🧠 TF-IDF: {len(tfidf_cols)} features</span>')

    st.markdown("**Applied Steps:**")
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df_proc.head(), use_container_width=True)

    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(1)
    with cb:
        if st.button("Continue to Training →"): nav(3)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN
# ══════════════════════════════════════════════════════════════════════════
elif step == 3:
    df_proc      = st.session_state.df_proc
    target       = st.session_state.target
    problem_type = st.session_state.problem_type

    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 4 — Training All Models</div>', unsafe_allow_html=True)

    feat_cols=[c for c in df_proc.columns if c!=target]
    X=df_proc[feat_cols].values
    y=df_proc[target].values if target else None

    scaler=StandardScaler()
    X_sc=scaler.fit_transform(X)
    st.session_state.scaler=scaler; st.session_state.feat_cols=feat_cols

    if problem_type!='clustering':
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        Xtr_sc=scaler.transform(X_train); Xte_sc=scaler.transform(X_test)
        st.session_state.X_test=X_test; st.session_state.X_test_sc=Xte_sc; st.session_state.y_test=y_test

    results=[]

    # ── Supervised models ────────────────────────────────────────────────
    if problem_type=='classification':
        ml_models={
            "Random Forest":       (RandomForestClassifier(n_estimators=100,random_state=42),False),
            "Gradient Boosting":   (GradientBoostingClassifier(random_state=42),False),
            "Logistic Regression": (LogisticRegression(max_iter=1000),True),
            "SVM":                 (SVC(),True),
            "KNN":                 (KNeighborsClassifier(),True),
            "Decision Tree":       (DecisionTreeClassifier(random_state=42),False),
        }
        metric_name="Accuracy"; sort_col="Accuracy"
    elif problem_type=='regression':
        ml_models={
            "Random Forest":     (RandomForestRegressor(n_estimators=100,random_state=42),False),
            "Gradient Boosting": (GradientBoostingRegressor(random_state=42),False),
            "Linear Regression": (LinearRegression(),True),
            "Ridge Regression":  (Ridge(),True),
            "SVR":               (SVR(),True),
            "KNN":               (KNeighborsRegressor(),True),
            "Decision Tree":     (DecisionTreeRegressor(random_state=42),False),
        }
        metric_name="R² Score"; sort_col="R² Score"
    else:
        ml_models={}; metric_name="Silhouette"; sort_col="Silhouette"

    st.session_state.metric_name=metric_name; st.session_state.sort_col=sort_col

    n_cluster_models=4
    n_dr_models=3
    total=len(ml_models)+1+n_cluster_models+n_dr_models  # ML + DL + cluster + DR
    progress=st.progress(0); status=st.empty(); done=0

    for name,(model,use_sc) in ml_models.items():
        status.markdown(f'<div style="font-size:.85rem;color:#6b7280;">🤖 {name}...</div>',unsafe_allow_html=True)
        Xtr=Xtr_sc if use_sc else X_train; Xte=Xte_sc if use_sc else X_test
        model.fit(Xtr,y_train); yp=model.predict(Xte)
        if problem_type=='classification':
            results.append({'Model':name,'Category':'ML','Accuracy':round(accuracy_score(y_test,yp),4),
                'F1 Score':round(f1_score(y_test,yp,average='weighted',zero_division=0),4),
                '_model':model,'_scaled':use_sc,'_type':'ml'})
        else:
            results.append({'Model':name,'Category':'ML','R² Score':round(r2_score(y_test,yp),4),
                'RMSE':round(np.sqrt(mean_squared_error(y_test,yp)),4),
                '_model':model,'_scaled':use_sc,'_type':'ml'})
        done+=1; progress.progress(done/total)

    # ── Deep Learning ────────────────────────────────────────────────────
    if problem_type in ('classification','regression'):
        status.markdown('<div style="font-size:.85rem;color:#f87171;">🧠 Deep Learning (Keras)...</div>',unsafe_allow_html=True)
        try:
            import tensorflow as tf
            from tensorflow import keras
            tf.get_logger().setLevel('ERROR'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
            nf=Xtr_sc.shape[1]
            if problem_type=='classification':
                nc=len(np.unique(y_train))
                ou,oa,lo=(1,'sigmoid','binary_crossentropy') if nc==2 else (nc,'softmax','sparse_categorical_crossentropy')
                dl=keras.Sequential([keras.layers.Input(shape=(nf,)),
                    keras.layers.Dense(256,activation='relu'),keras.layers.BatchNormalization(),keras.layers.Dropout(.3),
                    keras.layers.Dense(128,activation='relu'),keras.layers.BatchNormalization(),keras.layers.Dropout(.2),
                    keras.layers.Dense(64,activation='relu'),keras.layers.Dense(ou,activation=oa)])
                dl.compile(optimizer='adam',loss=lo,metrics=['accuracy'])
                dl.fit(Xtr_sc,y_train,epochs=30,batch_size=32,validation_split=.1,verbose=0,
                       callbacks=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
                yp2=dl.predict(Xte_sc,verbose=0)
                yp2=(yp2>.5).astype(int).flatten() if nc==2 else np.argmax(yp2,axis=1)
                results.append({'Model':'Deep Learning','Category':'DL',
                    'Accuracy':round(accuracy_score(y_test,yp2),4),
                    'F1 Score':round(f1_score(y_test,yp2,average='weighted',zero_division=0),4),
                    '_model':dl,'_scaled':True,'_type':'dl'})
            else:
                dl=keras.Sequential([keras.layers.Input(shape=(nf,)),
                    keras.layers.Dense(256,activation='relu'),keras.layers.BatchNormalization(),keras.layers.Dropout(.3),
                    keras.layers.Dense(128,activation='relu'),keras.layers.BatchNormalization(),keras.layers.Dropout(.2),
                    keras.layers.Dense(64,activation='relu'),keras.layers.Dense(1)])
                dl.compile(optimizer='adam',loss='mse')
                dl.fit(Xtr_sc,y_train,epochs=30,batch_size=32,validation_split=.1,verbose=0,
                       callbacks=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
                yp2=dl.predict(Xte_sc,verbose=0).flatten()
                results.append({'Model':'Deep Learning','Category':'DL',
                    'R² Score':round(r2_score(y_test,yp2),4),
                    'RMSE':round(np.sqrt(mean_squared_error(y_test,yp2)),4),
                    '_model':dl,'_scaled':True,'_type':'dl'})
            st.session_state.dl_model=dl
        except Exception as e:
            st.warning(f"Deep Learning skipped: {e}")
    done+=1; progress.progress(done/total)

    # ── Clustering ───────────────────────────────────────────────────────
    status.markdown('<div style="font-size:.85rem;color:#fbbf24;">🔵 Clustering models...</div>',unsafe_allow_html=True)
    n_clusters=5 if len(X_sc)>200 else 3
    cluster_models={
        "K-Means":          KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "Agglomerative":    AgglomerativeClustering(n_clusters=n_clusters),
        "Gaussian Mixture": GaussianMixture(n_components=n_clusters, random_state=42),
        "DBSCAN":           DBSCAN(eps=0.5, min_samples=5),
    }
    cluster_results=[]
    for cname,cmodel in cluster_models.items():
        try:
            labels_c=cmodel.fit_predict(X_sc)
            n_unique=len(set(labels_c)-{-1})
            if n_unique>=2:
                sil=round(silhouette_score(X_sc,labels_c),4)
                db =round(davies_bouldin_score(X_sc,labels_c),4)
            else:
                sil=-1; db=999
            cluster_results.append({'Model':cname,'Category':'Clustering',
                'Silhouette':sil,'Davies-Bouldin':db,'Clusters':n_unique,
                '_model':cmodel,'_labels':labels_c,'_type':'cluster'})
        except Exception as e:
            cluster_results.append({'Model':cname,'Category':'Clustering',
                'Silhouette':-1,'Davies-Bouldin':999,'Clusters':0,
                '_model':None,'_labels':None,'_type':'cluster'})
        done+=1; progress.progress(done/total)

    st.session_state.cluster_results=cluster_results

    # ── Dimensionality Reduction ─────────────────────────────────────────
    status.markdown('<div style="font-size:.85rem;color:#818cf8;">🔷 Dimensionality Reduction...</div>',unsafe_allow_html=True)
    dr_results=[]
    n_comp=min(2, X_sc.shape[1])

    # PCA
    try:
        pca=PCA(n_components=min(X_sc.shape[1], X_sc.shape[0]))
        pca.fit(X_sc)
        var2=sum(pca.explained_variance_ratio_[:2])*100
        pca2=PCA(n_components=n_comp); X_pca=pca2.fit_transform(X_sc)
        dr_results.append({'Method':'PCA','Variance Explained (2D)':round(var2,2),
            'Components':n_comp,'_obj':pca2,'_X2d':X_pca,'_type':'dr'})
    except Exception as e:
        pass
    done+=1; progress.progress(done/total)

    # TruncatedSVD (LSA)
    try:
        svd=TruncatedSVD(n_components=n_comp, random_state=42)
        X_svd=svd.fit_transform(X_sc)
        var_svd=sum(svd.explained_variance_ratio_)*100
        dr_results.append({'Method':'Truncated SVD (LSA)','Variance Explained (2D)':round(var_svd,2),
            'Components':n_comp,'_obj':svd,'_X2d':X_svd,'_type':'dr'})
    except Exception as e:
        pass
    done+=1; progress.progress(done/total)

    # t-SNE
    try:
        sample_n=min(2000, len(X_sc))
        idx=np.random.choice(len(X_sc), sample_n, replace=False)
        tsne=TSNE(n_components=2, random_state=42, perplexity=min(30, sample_n-1), n_iter=300)
        X_tsne=tsne.fit_transform(X_sc[idx])
        dr_results.append({'Method':'t-SNE','Variance Explained (2D)':'N/A',
            'Components':2,'_obj':tsne,'_X2d':X_tsne,'_idx':idx,'_type':'dr'})
    except Exception as e:
        pass
    done+=1; progress.progress(done/total)

    st.session_state.dr_results=dr_results

    progress.progress(1.0); status.empty()

    # build master results for supervised
    if problem_type!='clustering' and results:
        rdf=pd.DataFrame(results).sort_values(sort_col,ascending=False).reset_index(drop=True)
        st.session_state.results=rdf
        best=rdf.iloc[0]
        st.session_state.best_model=best['_model']; st.session_state.best_name=best['Model']
        st.session_state.best_scaled=best['_scaled']; st.session_state.best_is_dl=best['_type']=='dl'

    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(2)
    with cb:
        if st.button("View Results →"): nav(4)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════
elif step == 4:
    problem_type = st.session_state.problem_type
    results_df   = st.session_state.results
    cluster_res  = st.session_state.cluster_results
    dr_res       = st.session_state.dr_results
    feat_cols    = st.session_state.feat_cols
    sort_col     = st.session_state.sort_col
    metric_name  = st.session_state.metric_name
    X_test       = st.session_state.X_test
    X_test_sc    = st.session_state.X_test_sc
    y_test       = st.session_state.y_test
    scaler       = st.session_state.scaler
    df_proc      = st.session_state.df_proc
    target       = st.session_state.target

    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 5 — Results</div>', unsafe_allow_html=True)

    # ── Summary metrics ──────────────────────────────────────────────────
    best_score = results_df.iloc[0][sort_col] if results_df is not None and len(results_df) else 0
    best_name  = st.session_state.best_name or "—"
    n_cluster_models = len(cluster_res) if cluster_res else 0
    n_dr_models      = len(dr_res) if dr_res else 0
    total_models = (len(results_df) if results_df is not None else 0) + n_cluster_models + n_dr_models

    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-box"><div class="metric-val">{total_models}</div><div class="metric-key">Total Models</div></div>
      <div class="metric-box"><div class="metric-val">{best_score:.2%}</div><div class="metric-key">Best {metric_name}</div></div>
      <div class="metric-box"><div class="metric-val">{n_cluster_models}</div><div class="metric-key">Cluster Models</div></div>
      <div class="metric-box"><div class="metric-val">{n_dr_models}</div><div class="metric-key">DR Methods</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Supervised model comparison ──────────────────────────────────────
    if results_df is not None and len(results_df):
        st.markdown('<div class="section-sep"></div><div class="section-head">🤖 ML + Deep Learning — Model Comparison</div>', unsafe_allow_html=True)
        max_s=results_df[sort_col].max(); min_s=results_df[sort_col].min(); rng=max_s-min_s if max_s!=min_s else 1
        for i,row in results_df.iterrows():
            is_best=i==0; bar_pct=int(((row[sort_col]-min_s)/rng)*100) if rng>0 else 100
            cls='best' if is_best else ''
            badge='<span class="best-badge">BEST</span>' if is_best else ''
            dl_b='<span class="dl-badge">DL</span>' if row['_type']=='dl' else ''
            sec=f"F1: {row['F1 Score']:.4f}" if problem_type=='classification' else f"RMSE: {row['RMSE']:.4f}"
            st.markdown(f"""
            <div class="model-row {cls}">
              <div class="model-name">{row['Model']} {badge} {dl_b}</div>
              <div class="model-bar-wrap"><div class="model-bar" style="width:{bar_pct}%"></div></div>
              <div class="model-score">{row[sort_col]:.4f}</div>
              <div style="font-size:.72rem;color:#4b5563;min-width:130px;text-align:right">{sec}</div>
            </div>""", unsafe_allow_html=True)

        # Best model details
        best_model  = st.session_state.best_model
        best_is_dl  = st.session_state.best_is_dl
        if best_is_dl:
            yp_b=best_model.predict(X_test_sc,verbose=0)
            nc=len(np.unique(y_test))
            yp_b=(yp_b>.5).astype(int).flatten() if nc==2 else np.argmax(yp_b,axis=1)
        else:
            Xte=X_test_sc if st.session_state.best_scaled else X_test
            yp_b=best_model.predict(Xte)

        if problem_type=='classification':
            with st.expander("📋 Classification Report — Best Model"):
                le_t=st.session_state.le_target
                tn=le_t.classes_.astype(str) if le_t else None
                rpt=classification_report(y_test,yp_b,target_names=tn,output_dict=True)
                st.dataframe(pd.DataFrame(rpt).transpose().round(3),use_container_width=True)
            with st.expander("🔲 Confusion Matrix"):
                cm=confusion_matrix(y_test,yp_b)
                fig,ax=dark_fig(6,4)
                sns.heatmap(cm,annot=True,fmt='d',cmap='Greens',ax=ax,linewidths=.5)
                ax.set_xlabel('Predicted',color='#6b7280',fontsize=9)
                ax.set_ylabel('Actual',color='#6b7280',fontsize=9)
                ax.tick_params(colors='#6b7280',labelsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            with st.expander("📈 Actual vs Predicted"):
                fig,ax=dark_fig(7,4)
                ax.scatter(y_test,yp_b,alpha=.5,color='#00c896',s=15)
                mn,mx=min(y_test.min(),yp_b.min()),max(y_test.max(),yp_b.max())
                ax.plot([mn,mx],[mn,mx],'--',color='#0078ff',linewidth=1.5)
                ax.set_xlabel('Actual',color='#6b7280',fontsize=9); ax.set_ylabel('Predicted',color='#6b7280',fontsize=9)
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        if not best_is_dl and hasattr(best_model,'feature_importances_'):
            with st.expander("🔍 Feature Importance (Top 15)"):
                fi=pd.Series(best_model.feature_importances_,index=feat_cols).sort_values(ascending=False)[:15]
                fig2,ax2=dark_fig(8,4)
                colors=['#00c896' if i==0 else '#0078ff' if i<3 else '#374151' for i in range(len(fi))]
                ax2.barh(fi.index[::-1],fi.values[::-1],color=colors[::-1])
                style_ax(ax2); plt.tight_layout(); st.pyplot(fig2); plt.close()

    # ── Clustering Results ───────────────────────────────────────────────
    if cluster_res:
        st.markdown('<div class="section-sep"></div><div class="section-head">🔵 Clustering Results</div>', unsafe_allow_html=True)
        valid_cr=[r for r in cluster_res if r['Silhouette']>-1]
        if valid_cr:
            best_cr=max(valid_cr, key=lambda x: x['Silhouette'])
            for r in cluster_res:
                is_b=r['Model']==best_cr['Model']
                cls='best' if is_b else ''
                s=r['Silhouette']; db=r['Davies-Bouldin']; nc=r['Clusters']
                badge='<span class="best-badge">BEST</span>' if is_b else ''
                cbadge='<span class="cluster-badge">CLUSTER</span>'
                bar_pct=int(max(0,s)*100) if s>0 else 0
                st.markdown(f"""
                <div class="model-row {cls}">
                  <div class="model-name">{r['Model']} {badge} {cbadge}</div>
                  <div class="model-bar-wrap"><div class="model-bar" style="width:{bar_pct}%"></div></div>
                  <div class="model-score">{s:.4f}</div>
                  <div style="font-size:.72rem;color:#4b5563;min-width:160px;text-align:right">
                    DB: {db:.4f} | Clusters: {nc}</div>
                </div>""", unsafe_allow_html=True)

            # Visualise best clustering
            with st.expander(f"🗺 Cluster Visualisation — {best_cr['Model']} (PCA 2D)"):
                feat_cols_all=[c for c in df_proc.columns if c!=target]
                Xall=scaler.transform(df_proc[feat_cols_all].values)
                pca_vis=PCA(n_components=2); X2d=pca_vis.fit_transform(Xall)
                lbl=best_cr['_labels']
                fig,ax=dark_fig(7,5)
                scatter_colors=['#00c896','#0078ff','#a855f7','#f59e0b','#ef4444','#06b6d4','#84cc16','#f43f5e']
                for ci in sorted(set(lbl)):
                    mask=lbl==ci
                    col=scatter_colors[ci%len(scatter_colors)] if ci>=0 else '#374151'
                    ax.scatter(X2d[mask,0],X2d[mask,1],c=col,s=15,alpha=.7,
                               label='Noise' if ci==-1 else f'Cluster {ci}')
                ax.legend(fontsize=7,labelcolor='#9ca3af',facecolor='#0d1117',edgecolor='#1f2937')
                ax.set_xlabel('PC1',color='#6b7280',fontsize=9); ax.set_ylabel('PC2',color='#6b7280',fontsize=9)
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

            # Elbow chart for K-Means
            with st.expander("📐 K-Means Elbow Chart"):
                inertias=[]; ks=range(2,min(11,len(X_sc)//5+2))
                Xall2=scaler.transform(df_proc[[c for c in df_proc.columns if c!=target]].values)
                for k in ks:
                    km=KMeans(n_clusters=k,random_state=42,n_init=10); km.fit(Xall2)
                    inertias.append(km.inertia_)
                fig,ax=dark_fig(7,3)
                ax.plot(list(ks),inertias,color='#00c896',marker='o',markersize=6,linewidth=2)
                ax.set_xlabel('Number of Clusters',color='#6b7280',fontsize=9)
                ax.set_ylabel('Inertia',color='#6b7280',fontsize=9)
                ax.set_title('Elbow Method',color='#9ca3af',fontsize=10)
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Dimensionality Reduction Results ─────────────────────────────────
    if dr_res:
        st.markdown('<div class="section-sep"></div><div class="section-head">🔷 Dimensionality Reduction</div>', unsafe_allow_html=True)
        for r in dr_res:
            drbadge='<span class="dr-badge">DR</span>'
            var=r['Variance Explained (2D)']
            bar_pct=int(var) if isinstance(var,float) else 50
            st.markdown(f"""
            <div class="model-row">
              <div class="model-name">{r['Method']} {drbadge}</div>
              <div class="model-bar-wrap"><div class="model-bar" style="width:{bar_pct}%"></div></div>
              <div class="model-score">{var}{'%' if isinstance(var,float) else ''}</div>
              <div style="font-size:.72rem;color:#4b5563;min-width:130px;text-align:right">
                Components: {r['Components']}</div>
            </div>""", unsafe_allow_html=True)

        # Scatter plots for each DR method
        lbl_col=None
        if cluster_res:
            valid_cr2=[r for r in cluster_res if r['_labels'] is not None]
            if valid_cr2: lbl_col=max(valid_cr2,key=lambda x:x['Silhouette'])['_labels']

        for r in dr_res:
            with st.expander(f"🗺 {r['Method']} — 2D Projection"):
                X2d=r['_X2d']; idx=r.get('_idx',None)
                colors_dr=lbl_col[idx] if lbl_col is not None and idx is not None else (
                          lbl_col[:len(X2d)] if lbl_col is not None else np.zeros(len(X2d)))
                fig,ax=dark_fig(7,5)
                scatter=ax.scatter(X2d[:,0],X2d[:,1],c=colors_dr,cmap='plasma',s=12,alpha=.7)
                ax.set_xlabel('Component 1',color='#6b7280',fontsize=9)
                ax.set_ylabel('Component 2',color='#6b7280',fontsize=9)
                ax.set_title(r['Method'],color='#9ca3af',fontsize=10)
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        # PCA variance explained bar chart
        pca_dr=[r for r in dr_res if r['Method']=='PCA']
        if pca_dr:
            with st.expander("📊 PCA — Variance Explained per Component"):
                feat_cols_all=[c for c in df_proc.columns if c!=target]
                Xall=scaler.transform(df_proc[feat_cols_all].values)
                n_comp_full=min(20,Xall.shape[1],Xall.shape[0])
                pca_full=PCA(n_components=n_comp_full); pca_full.fit(Xall)
                fig,ax=dark_fig(8,3)
                ax.bar(range(1,n_comp_full+1),pca_full.explained_variance_ratio_*100,color='#818cf8',alpha=.85)
                ax.plot(range(1,n_comp_full+1),np.cumsum(pca_full.explained_variance_ratio_*100),
                        color='#00c896',marker='o',markersize=4,linewidth=2,label='Cumulative')
                ax.set_xlabel('Component',color='#6b7280',fontsize=9)
                ax.set_ylabel('Variance Explained (%)',color='#6b7280',fontsize=9)
                ax.legend(fontsize=8,labelcolor='#9ca3af',facecolor='#0d1117',edgecolor='#1f2937')
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(3)
    with cb:
        if st.button("Download →"): nav(5)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════
elif step == 5:
    results_df   = st.session_state.results
    cluster_res  = st.session_state.cluster_results
    dr_res       = st.session_state.dr_results
    best_model   = st.session_state.best_model
    best_name    = st.session_state.best_name
    best_is_dl   = st.session_state.best_is_dl
    scaler       = st.session_state.scaler
    sort_col     = st.session_state.sort_col

    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 6 — Download</div>', unsafe_allow_html=True)
    best_score=results_df.iloc[0][sort_col] if results_df is not None and len(results_df) else 0
    st.markdown(f"""
    <div style="text-align:center;padding:2rem 0">
      <div style="font-size:3rem;margin-bottom:1rem">🎉</div>
      <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#00c896;margin-bottom:.5rem">All Models Trained!</div>
      <div style="font-size:.9rem;color:#6b7280">Best: <strong style="color:#d4d8f0">{best_name}</strong>
      &nbsp;|&nbsp; {sort_col}: <strong style="color:#00c896">{best_score:.4f}</strong></div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    with c1:
        if best_model:
            if best_is_dl:
                try:
                    import tempfile,zipfile
                    with tempfile.TemporaryDirectory() as td:
                        mp=os.path.join(td,'dl_model.keras'); best_model.save(mp)
                        zb=io.BytesIO()
                        with zipfile.ZipFile(zb,'w') as zf: zf.write(mp,'dl_model.keras')
                        zb.seek(0)
                    st.download_button("⬇ Download DL Model (.zip)",zb,"deep_learning_model.zip","application/zip",use_container_width=True)
                except Exception as e: st.warning(str(e))
            else:
                buf=io.BytesIO(); joblib.dump(best_model,buf); buf.seek(0)
                st.download_button("⬇ Download Best Model (.pkl)",buf,f"{best_name.replace(' ','_')}.pkl","application/octet-stream",use_container_width=True)
    with c2:
        buf2=io.BytesIO(); joblib.dump(scaler,buf2); buf2.seek(0)
        st.download_button("⬇ Download Scaler (.pkl)",buf2,"scaler.pkl","application/octet-stream",use_container_width=True)
    with c3:
        rows=[]
        if results_df is not None:
            for _,r in results_df.iterrows():
                rows.append({k:v for k,v in r.items() if not k.startswith('_')})
        if cluster_res:
            for r in cluster_res:
                rows.append({k:v for k,v in r.items() if not k.startswith('_')})
        if dr_res:
            for r in dr_res:
                rows.append({k:v for k,v in r.items() if not k.startswith('_')})
        if rows:
            csv=pd.DataFrame(rows).to_csv(index=False)
            st.download_button("⬇ Download All Results (.csv)",csv,"all_results.csv","text/csv",use_container_width=True)

    # tables
    if results_df is not None and len(results_df):
        st.markdown("<br>**🤖 ML + DL Comparison**")
        disp=[c for c in results_df.columns if not c.startswith('_')]
        st.dataframe(results_df[disp],use_container_width=True)

    if cluster_res:
        st.markdown("**🔵 Clustering Comparison**")
        cdf=pd.DataFrame([{k:v for k,v in r.items() if not k.startswith('_')} for r in cluster_res])
        st.dataframe(cdf,use_container_width=True)

    if dr_res:
        st.markdown("**🔷 Dimensionality Reduction**")
        ddf=pd.DataFrame([{k:v for k,v in r.items() if not k.startswith('_')} for r in dr_res])
        st.dataframe(ddf,use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("🔄 Start Over"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
  border-top:1px solid rgba(255,255,255,.04);
  font-family:'Space Mono',monospace;font-size:.7rem;color:#1f2937;letter-spacing:.08em">
  AUTOML STUDIO &nbsp;·&nbsp; ML · NLP · DEEP LEARNING · CLUSTERING · DIMENSIONALITY REDUCTION
</div>
""", unsafe_allow_html=True)
