import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

#Загрузка модели
MODEL_PATH = os.path.join('models', 'best_diabetes_model.pkl')

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error("Модель не найдена. Убедитесь, что файл `models/best_diabetes_model.pkl` существует.")
    st.stop()

try:
    df = load_data(os.path.join('data', 'diabetes.csv'))
except FileNotFoundError:
    df = None

#Настройки страницы
st.set_page_config(page_title='MedAI Diabetes', layout='wide', page_icon='🩺')

st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
        border-radius: 16px; padding: 3rem 2.5rem; margin-bottom: 2rem; color: white;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 0.5rem; }
    .hero p  { font-size: 1.1rem; opacity: 0.85; margin: 0; }
    .stat-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
    .stat-box {
        flex: 1; background: white; border-radius: 12px;
        padding: 1.4rem; border-top: 3px solid #1D9E75;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }
    .stat-box .num { font-size: 2rem; font-weight: 800; color: #064e3b; }
    .stat-box .txt { font-size: 0.82rem; color: #6b7280; margin-top: 2px; }
    .feature-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
    .feature-card {
        background: white; border-radius: 12px; padding: 1.4rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }
    .feature-card .icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
    .feature-card h4 { margin: 0 0 0.3rem; color: #1e293b; font-size: 1rem; }
    .feature-card p  { margin: 0; color: #6b7280; font-size: 0.85rem; line-height: 1.5; }
    .result-ok {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-radius: 12px; padding: 1.5rem 2rem;
        border-left: 5px solid #059669;
    }
    .result-bad {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-radius: 12px; padding: 1.5rem 2rem;
        border-left: 5px solid #dc2626;
    }
    .result-ok h3, .result-bad h3 { margin: 0 0 0.3rem; font-size: 1.3rem; }
    .result-ok p,  .result-bad p  { margin: 0; font-size: 0.95rem; opacity: 0.8; }
    .input-card {
        background: white; border-radius: 14px;
        padding: 1.8rem; box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }
    .norm-table {
        background: #f8fafc; border-radius: 10px; padding: 1.2rem;
        font-size: 0.85rem;
    }
    .norm-row { display: flex; justify-content: space-between;
                padding: 0.4rem 0; border-bottom: 1px solid #e2e8f0; }
    .norm-row:last-child { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

#Навигация
page = st.sidebar.radio('', ['Главная', 'Диагностика', 'Аналитика данных'])

#ГЛАВНАЯ
if page == 'Главная':
    st.markdown("""
    <div class="hero">
        <h1>MedAI — Диагностика диабета</h1>
        <p>Система раннего выявления сахарного диабета на основе машинного обучения.<br>
        Введите показатели пациента и получите мгновенный результат.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-box"><div class="num">768</div><div class="txt">пациентов в обучающей выборке</div></div>
        <div class="stat-box"><div class="num">8</div><div class="txt">медицинских признаков</div></div>
        <div class="stat-box"><div class="num">0.82</div><div class="txt">ROC-AUC лучшей модели</div></div>
        <div class="stat-box"><div class="num">3</div><div class="txt">алгоритма машинного обучения</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="icon">🩺</div>
            <h4>Быстрая диагностика</h4>
            <p>Введите 8 показателей пациента и получите вероятность диабета за секунду</p>
        </div>
        <div class="feature-card">
            <div class="icon">📊</div>
            <h4>Аналитика данных</h4>
            <p>Интерактивные графики распределений, корреляций и статистики датасета</p>
        </div>
        <div class="feature-card">
            <div class="icon">🤖</div>
            <h4>Три модели МО</h4>
            <p>Random Forest, Gradient Boosting и Logistic Regression в одном пайплайне</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.caption('Датасет: Pima Indians Diabetes Database (Kaggle / UCI). Приложение создано в учебных целях.')

#ДИАГНОСТИКА
elif page == 'Диагностика':
    st.title('Диагностика пациента')
    st.markdown('Заполните медицинские показатели и нажмите кнопку.')
    st.divider()

    col_form, col_side = st.columns([3, 2])

    with col_form:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input('Беременностей', 0, 20, 1)
            glucose     = st.number_input('Глюкоза (мг/дл)', 0, 300, 120)
            bp          = st.number_input('Давление (мм рт.ст.)', 0, 150, 70)
            skin        = st.number_input('Толщина кожи (мм)', 0, 100, 20)
        with c2:
            insulin = st.number_input('Инсулин (мкЕд/мл)', 0, 900, 80)
            bmi     = st.number_input('ИМТ', 0.0, 70.0, 25.0)
            dpf     = st.number_input('Наследственность', 0.0, 3.0, 0.5)
            age     = st.number_input('Возраст', 21, 100, 30)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        run = st.button('Поставить диагноз', type='primary', use_container_width=True)

        if run:
            x     = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            pred  = model.predict(x)[0]
            proba = model.predict_proba(x)[0][1]
            st.markdown('<br>', unsafe_allow_html=True)
            if pred == 1:
                st.markdown(f"""
                <div class="result-bad">
                    <h3>Диабет вероятен</h3>
                    <p>Вероятность по модели: <b>{proba:.1%}</b></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-ok">
                    <h3>Диабет маловероятен</h3>
                    <p>Вероятность по модели: <b>{proba:.1%}</b></p>
                </div>""", unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.progress(float(proba))
            st.caption('Результат не является медицинским заключением.')

    with col_side:
        st.markdown("""
        <div class="norm-table">
            <b style="font-size:0.9rem;color:#1e293b">Нормальные значения</b><br><br>
            <div class="norm-row"><span>Глюкоза</span><span style="color:#059669">70-99 мг/дл</span></div>
            <div class="norm-row"><span>Давление</span><span style="color:#059669">60-80 мм рт.ст.</span></div>
            <div class="norm-row"><span>ИМТ</span><span style="color:#059669">18.5-24.9</span></div>
            <div class="norm-row"><span>Инсулин</span><span style="color:#059669">16-166 мкЕд/мл</span></div>
            <div class="norm-row"><span>Толщина кожи</span><span style="color:#059669">10-40 мм</span></div>
        </div>
        """, unsafe_allow_html=True)

#АНАЛИТИКА
elif page == 'Аналитика данных':
    if df is None:
        st.warning('Файл `data/diabetes.csv` не найден. Загрузите датасет в папку `data/`.')
        st.stop()

    st.title('Аналитика датасета')
    st.markdown('Pima Indians Diabetes Database — 768 пациентов, 8 признаков.')
    st.divider()

    tab1, tab2, tab3 = st.tabs(['Распределения', 'Корреляции', 'Статистика'])

    with tab1:
        feat = st.selectbox('Признак', [
            'Glucose', 'BMI', 'Age', 'BloodPressure',
            'Insulin', 'SkinThickness', 'Pregnancies', 'DiabetesPedigreeFunction'
        ])
        fig, ax = plt.subplots(figsize=(9, 4))
        for outcome, color, label in [(0, '#5DCAA5', 'Нет диабета'), (1, '#E8593C', 'Есть диабет')]:
            data = df[df['Outcome'] == outcome][feat]
            data = data[data > 0]
            ax.hist(data, bins=30, alpha=0.65, color=color, label=label, edgecolor='none')
        ax.set_title(f'Распределение: {feat}', fontweight='bold', fontsize=12)
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, linewidths=0.5,
                    annot_kws={'size': 9}, ax=ax)
        ax.set_title('Корреляционная матрица', fontweight='bold', fontsize=12)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)

    with tab3:
        vc = df['Outcome'].value_counts()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric('Всего пациентов', len(df))
        with c2:
            st.metric('Нет диабета', f'{vc[0]} ({vc[0]/len(df)*100:.0f}%)')
        with c3:
            st.metric('Есть диабет', f'{vc[1]} ({vc[1]/len(df)*100:.0f}%)')

        st.markdown('<br>', unsafe_allow_html=True)
        st.dataframe(df.describe().round(2), use_container_width=True)
