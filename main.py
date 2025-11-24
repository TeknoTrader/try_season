import math
import sys
from datetime import date
from datetime import datetime

import altair as alt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Disable yfinance verbose output
import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Colors
sidebar_color = "#1A4054"
main_bg_color = "#649ABA"
text_color = "#1A4054"
widget_color = "#ffffff"
header_color = "#880C14"
label_color = "#FFFFFF"

# Photos
url_analysis = "https://i.postimg.cc/5yFWvkJV/Analysis-screen.png"
url_strategy = "https://i.postimg.cc/ncT1PhkP/screen-contorno.png"
url_strategy2 = "https://i.postimg.cc/WbYzhB8y/strategy-red.png"
url_home = "https://i.postimg.cc/Bnm5nhLQ/screen-home-con-contorno.png"

# Url of yahoo!finance ticker's list
url = "https://finance.yahoo.com/lookup/"
NomiMesi1 = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER",
                 "NOVEMBER", "DECEMBER"]

current_year = datetime.now().year

# Apply label colors CSS immediately (before any widgets are created)
st.markdown(f"""
    <style>
    .stRadio label, .stSelectbox label, .stTextInput label, .stNumberInput label, .stMultiSelect label, .stToggle label {{
        color: {label_color};
    }}
    </style>
    """, unsafe_allow_html=True)


# Load external CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Useful function for performances evaluation
def WinRate(Arr):
    tot = 0
    for i in Arr:
        if (i >= 0):
            tot += 1
    if (tot == 0):
        return 0
    else:
        return (100 / len(Arr) * tot)


def Sortino_Ratio_Benchmark(returns, benchmark_ticker=None, period='1y'):
    if isinstance(returns, pd.Series):
        returns_array = returns.values
    else:
        returns_array = np.array(returns)

    if benchmark_ticker:
        benchmark_data = yf.download(benchmark_ticker, period=period, progress=False)

        if len(benchmark_data) == 0:
            raise ValueError("Impossibile scaricare i dati del benchmark.")

        benchmark_returns = (benchmark_data['Close'] - benchmark_data["Open"]).pct_change().dropna().values

        min_length = min(len(returns_array), len(benchmark_returns))
        returns_array = returns_array[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]

        excess_returns = (1 + returns_array) / (1 + benchmark_returns) - 1
        risk_free_rate = np.mean(benchmark_returns)
    else:
        excess_returns = returns_array
        risk_free_rate = 0

    negative_returns = np.minimum(excess_returns - risk_free_rate, 0)
    downside_deviation = np.sqrt(np.mean(negative_returns ** 2))

    if downside_deviation == 0:
        return np.nan

    sortino_ratio = (np.mean(returns_array) - risk_free_rate) / downside_deviation
    return sortino_ratio


def calmar_ratio(returns, maxDD):
    average_return = sum(returns)/len(returns)
    return average_return/(-maxDD)


def Profit_Factor(trades):
    profits = [trade for trade in trades if trade > 0]
    losses = [-trade for trade in trades if trade < 0]

    total_profits = sum(profits)
    total_losses = sum(losses)

    if total_losses == 0:
        return float('inf')
    profit_factor = total_profits / total_losses

    return profit_factor


def Max_Drawdown(Historycal_Drawdowns):
    return max(Historycal_Drawdowns)


def Color(negclr, posclr, element, minimum):
    if (float(element) < float(minimum)):
        return (negclr)
    else:
        return (posclr)


def Color2(clr1, clr2, clr3, element, value1, value2):
    if element <= value1:
        return clr1
    elif value2 <= element:
        return clr3
    else:
        return clr2


def Text(text, color=text_color):
    st.write(f"<p style='color: {color};'>" + text + "</p>", unsafe_allow_html=True)


def Text2(text, color=text_color):
    st.markdown(f"<h2 style='color: {color};'>{text}</h2>", unsafe_allow_html=True)


def Text3(text, color=text_color):
    st.markdown(f"<h1 style='color: {color};'>{text}</h1>", unsafe_allow_html=True)


def credits():
    st.sidebar.write("# Who built this web application?")
    st.sidebar.write(
        "My name is Nicola Chimenti.\nI'm currently pursuing a degree in \"Digital Economics\" and I program trading sotwares for traders who want to automatize their strategies.")
    st.sidebar.image("https://i.postimg.cc/7LynpkrL/Whats-App-Image-2024-07-27-at-16-36-44.jpg")
    st.sidebar.write("\n# CONTACT ME")
    st.sidebar.write(
        "### ◾ [LinkedIn](https://www.linkedin.com/in/nicolachimenti?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")
    st.sidebar.write("### ◾ Email: nicola.chimenti.work@gmail.com")
    st.sidebar.write("\n# RESOURCES")
    st.sidebar.write("◾ [GitHub Profile](https://github.com/TeknoTrader)")
    st.sidebar.write("◾ [MQL5 Profile](https://www.mql5.com/it/users/teknotrader) with reviews")
    st.sidebar.write(
        "◾ [MT4 free softwares](https://www.mql5.com/it/users/teknotrader/seller#!category=2) for trading")
    st.sidebar.write("\n### Are you interested in the source code? 🧾")
    st.sidebar.write("Visit the [GitHub repository](https://github.com/TeknoTrader/OrganizationTools)")


def main_page():
    Text3("LET'S ANALYZE THE SEASONALITY OF AN ASSET 📊", "#ffffff")
    Text2("You have just to set: when to start with the monitoration,when to end and which is the asset to see")
    Text(
        "Please, note that it has been used the YAHOO! FINANCE API, so you have to select the ticker of the asset based on the yahoo!finance database")
    st.markdown(f"""
    <p class="colored-text"> You can check the name of the asset you're searching at this <a href="{url}">link</a>.</p>
    """, unsafe_allow_html=True)

    AnnoPartenz = st.number_input("Starting year 📅: ", min_value=1850, max_value=current_year - 1, step=1)
    AnnoFin = st.number_input("End year 📅: ", value=current_year, min_value=1900, max_value=current_year, step=1)

    if AnnoFin <= AnnoPartenz:
        st.warning(
            "# ⚠️ ATTENTION!!!\n### The starting year (" + str(
                AnnoPartenz) + ") mustn't be higher than the end year (" + str(AnnoFin) + ").")
        st.write("### Please, select another ending date for the relevation.")
        sys.exit(1)

    ticker = st.text_input("Insert the TICKER 📈: ", value="GOOG")

    try:
        asset = yf.Ticker(ticker)
        info = asset.info
        info.get('longName', 'N/A')
        if info and "error" in info:
            st.warning(f"# ⚠️ The asset {ticker} doesn't exist.")
            st.write(
                "### Maybe you didn't select the right ticker.\n### You can find here the [Yahoo finance ticker's list](url)")
            sys.exit(1)
    except Exception as e:
        st.warning(f"# ⚠️ Error with the asset {ticker}.")
        st.write(
            "### Probably you didn't insert the right ticker.\n### You can find here the [Yahoo finance ticker's list](url)\n")
        st.write(f"Fing here more details: \n{str(e)}")
        sys.exit(1)

    asset = yf.Ticker(ticker)
    info = asset.info
    asset_name = info.get('longName', 'N/A')

    def main():
        AnnoFine = int(AnnoFin)
        end = date(AnnoFine, 1, 1)
        Text(f"\nEnd of the relevation: {end}")
        
        # Determina l'anno di partenza basandosi sui dati disponibili
        if AnnoPartenz < 1950:
            AnnoPartenza = 1950  # Yahoo Finance ha dati dal 1950 circa
        else:
            AnnoPartenza = AnnoPartenz
        
        inizio = date(AnnoPartenza, 1, 1)
        Text(f"\nStarting calculations from: {inizio}")
        
        # Prova a scaricare i dati nell'intervallo richiesto
        with st.spinner(f'📥 Downloading data for {ticker} from {AnnoPartenza} to {AnnoFine}...'):
            try:
                df = yf.download(ticker, start=inizio, end=end, interval="1mo", progress=False)
                
                if df.empty:
                    st.warning(f"# ⚠️ No data available for {ticker} in the selected period!")
                    st.write(f"### The ticker {ticker} might not have data between {AnnoPartenza} and {AnnoFine}.")
                    st.write("### Try selecting a different time period or check the ticker on [Yahoo Finance](https://finance.yahoo.com)")
                    sys.exit(1)
                
                # Trova la prima data effettivamente disponibile
                first_date = df.index[0]
                first_year = int(first_date.strftime('%Y'))
                Text(f"Data for {ticker} available from: {first_date.date()}")
                
                # Aggiusta l'anno di partenza se necessario
                if first_year > AnnoPartenza:
                    AnnoPartenza = first_year
                    st.info(f"ℹ️ Adjusted starting year to {AnnoPartenza} (first available data)")
                    inizio = date(AnnoPartenza, 1, 1)
                    Text(f"\nAdjusted starting calculations from: {inizio}")
            except Exception as e:
                st.warning(f"# ⚠️ Error downloading data for {ticker}")
                st.write("### Maybe you didn't select the right ticker or there's a connection issue.")
                st.write("### You can check the ticker on [Yahoo finance ticker's list](https://finance.yahoo.com/lookup/)")
                st.write(f"Error details: {str(e)}")
                sys.exit(1)

        Annate1 = list(range(AnnoPartenza, AnnoFine))
        NomiMesi = list(range(1, 13))
        number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟", "1️⃣1️⃣", "1️⃣2️⃣"]

        Annate = []
        for i in Annate1:
            Annate.append(str(i))

        df = yf.download(ticker, start=inizio, end=end, interval="1mo", progress=False)
        df = pd.DataFrame(df["Open"])

        array = []
        WRComplessivi = []
        MesiComplessivi = []
        Months_to_consider = []
        NomiMesi2 = ["01-Jan", "02-Feb", "03-Mar", "04-Apr", "05-May", "06-JuN", "07-JuL", "08-Aug", "09-Sept",
                     "10-Oct", "11-Nov", "12-Dec"]

        W = 400
        H = 400

        Text3("LET'S SEE THE RESULTS 📈")
        Months = st.radio("Output selection:",
                          ("Choose manually the months", "Represent every month"))

        first_representation_model = "Not longer"
        if (Months == "Choose manually the months"):
            if (first_representation_model == "Longer"):
                if 'month_toggles' not in st.session_state:
                    st.session_state.month_toggles = {month: False for month in NomiMesi1}

                cols = st.columns(3)

                for index, month in enumerate(NomiMesi1):
                    with cols[index % 3]:
                        st.session_state.month_toggles[month] = st.toggle(month, st.session_state.month_toggles[month])
            else:
                options = st.multiselect(
                    "# Select the months to consider",
                    NomiMesi1
                )
        else:
            options = NomiMesi1

        if 'update_button' not in st.session_state:
            st.session_state.update_button = False

        if st.button("Update Visualization"):
            st.session_state.update_button = not st.session_state.update_button
            if st.session_state.update_button:
                st.info("🔄 Generating visualizations... This will take a moment.")
            st.rerun()

        def Represent(Mese, i, selections, db_selections):
            if st.session_state.update_button:
                # Controlla se ci sono abbastanza dati
                if not Mese or len(Mese) < 3:
                    Text3(f" {number_emojis[i - 1]} {NomiMesi1[i - 1]}", "#ffffff")
                    st.warning(f"⚠️ Not enough data available for {NomiMesi1[i - 1]}. Need at least 3 years of data.")
                    st.divider()
                    return
                
                # Crea gli array degli anni basandosi sui dati effettivamente disponibili
                anni_disponibili = []  # Stringhe per matplotlib
                Annate1_disponibili = []  # Numeri per Altair
                count = 0
                for year in range(AnnoPartenza, AnnoFine):
                    if count < len(Mese):
                        anni_disponibili.append(str(year))
                        Annate1_disponibili.append(year)
                        count += 1
                    else:
                        break
                
                colori = []
                for Y in Mese:
                    colori.append(Color("#FF0000", "#0000FF", Y, 0))

                Text3(f" {number_emojis[i - 1]} MONTHLY RETURNS of {asset_name} on the month of: {NomiMesi1[i - 1]} \n", "#ffffff")
                Text2(f"WIN RATE: {str(round(WinRate(Mese), 2))}%\n")
                Text2(f"AVERAGE RETURN: {str(round(np.mean(Mese), 2))} %\n")

                DevStd = math.sqrt(sum((x - np.mean(Mese)) ** 2 for x in Mese) / len(Mese))
                Text2(f"Standard deviation: {str(round(DevStd, 2))}%")

                Text(f"Better excursion: {round(max(Mese), 2)}%")
                Text(f"Worst excursion: {round(min(Mese), 2)} %")

                Graphical_options = ["Image", "Interactive"]
                key = f'select_{i + 1}'
                selections[key] = st.selectbox("### Type of chart", Graphical_options, key=key)

                xsize = 10
                ysize = 10

                if selections[key] == "Image":
                    fig, ax = plt.subplots(figsize=(xsize, ysize))

                    ax.bar(anni_disponibili, Mese, color=['blue' if x >= 0 else 'red' for x in Mese])

                    ax.axhline(np.mean(Mese), color="red", linestyle='--', linewidth=2)
                    ax.axhline(0, color="green")

                    ax.set_xlabel("Years")
                    ax.set_ylabel("Returns")

                    band_patch = mpatches.Patch(color='gray', alpha=0.3,
                                                label=f"Average ± Standard Deviation ({round(DevStd, 2)}%)")

                    ax.fill_between(
                        anni_disponibili,
                        np.mean(Mese) + DevStd,
                        np.mean(Mese) - DevStd,
                        color='gray',
                        alpha=0.3,
                        hatch="X",
                        edgecolor="gray",
                        label=f"Average ± Standard Deviation ({DevStd}%)"
                    )

                    ax.legend(handles=[
                        plt.Line2D([0], [0], color="red", lw=4, label="Negative Months"),
                        plt.Line2D([0], [0], color="blue", lw=4, label="Positive Months"),
                        plt.Line2D([0], [0], color="red", linestyle='--', lw=2,
                                   label=str("Average returns (" + str(round(np.mean(Mese), 2)) + "%)")), band_patch],
                        loc='upper right'
                    )

                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                else:
                    avacac = 2
                    Assex = Annate1_disponibili
                    Assey = Mese

                    Valore_Media = np.mean(Mese)

                    df = pd.DataFrame({
                        'Year': Assex,
                        'Return': Assey
                    })

                    y_min = min(min(Assey), Valore_Media - DevStd)
                    y_max = max(max(Assey), Valore_Media + DevStd)
                    y_domain = [y_min - 1, y_max + 1]

                    base = alt.Chart(df).encode(
                        x=alt.X('Year:O', title='Years')
                    )

                    fill_area = base.mark_area(opacity=0.2, color='gray').encode(
                        y=alt.Y('y1:Q', scale=alt.Scale(domain=y_domain), title='Return'),
                        y2=alt.Y2('y2:Q'),
                        tooltip=[
                            alt.Tooltip('y1:Q', title='Media - Standard Deviation', format='.2f'),
                            alt.Tooltip('y2:Q', title='Media + Standard Deviation', format='.2f'),
                            alt.Tooltip('average_return:Q', title='Average Return', format='.2%')
                        ]
                    ).transform_calculate(
                        y1=f"{Valore_Media - DevStd}",
                        y2=f"{Valore_Media + DevStd}",
                        average_return=f"{Valore_Media / 100}"
                    )

                    bars = base.mark_bar().encode(
                        y=alt.Y('Return:Q', scale=alt.Scale(domain=y_domain)),
                        color=alt.condition(
                            alt.datum.Return > 0,
                            alt.value('blue'),
                            alt.value('red')
                        ),
                        tooltip=[
                            alt.Tooltip('Year:O', title='Year'),
                            alt.Tooltip('return_percentage:Q', title='Return', format='.2%'),
                            alt.Tooltip('average_return:Q', title='Average Return', format='.2%')
                        ]
                    ).transform_calculate(
                        return_percentage="datum.Return/100",
                        average_return=f"{Valore_Media / 100}"
                    )

                    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='green', size=2).encode(
                        y='y'
                    )
                    three_line = alt.Chart(pd.DataFrame({'y': [Valore_Media]})).mark_rule(
                        color='orange',
                        strokeDash=[4, 4],
                        size=2
                    ).encode(y='y')

                    final_chart = (fill_area + bars + zero_line + three_line).properties(
                        width=W,
                        height=H,
                        title='Interactive chart'
                    )

                    legend_data = pd.DataFrame({
                        'color': ['gray', 'blue', 'red', 'green', 'orange'],
                        'Chart elements': ['Standard Dev.', 'Positive Return', 'Negative Return', 'Zero Line',
                                           'Average Return']
                    })

                    legend = alt.Chart(legend_data).mark_rect().encode(
                        y=alt.Y('Chart elements:N', axis=alt.Axis(orient='right')),
                        color=alt.Color('color:N', scale=None)
                    ).properties(
                        width=150,
                        title='Legend'
                    )

                    combined_chart = alt.hconcat(final_chart, legend)
                    st.altair_chart(combined_chart, use_container_width=True)

                options_DB = ["Graphical", "For CSV download"]
                db_key = f'db_select_{i + 1}'
                db_selections[db_key] = st.selectbox("### Type of database visualization", options_DB, key=db_key)

                if db_selections[db_key] == "For CSV download":
                    def format_value(val):
                        return f"{'+' if val > 0 else ''}{val:.2f}%"

                    MeseDF = pd.DataFrame({
                        "Year 📆": anni_disponibili,
                        "Lows 📉": [format_value(x) for x in Low(i, AnnoPartenza, AnnoFine)[:len(Mese)]],
                        "Return 📊": [format_value(x) for x in Mese],
                        "Highs 📈": [format_value(x) for x in High(i, AnnoPartenza, AnnoFine)[:len(Mese)]]
                    })
                    st.dataframe(MeseDF, hide_index=True)

                else:
                    Lows = Low(i, AnnoPartenza, AnnoFine)[:len(Mese)]
                    Highs = High(i, AnnoPartenza, AnnoFine)[:len(Mese)]

                    mostra_highs = True
                    mostra_lows = True

                    def style_numeric_column(val):
                        color = 'red' if val < 0 else 'blue'
                        return f'color: {color}; text-shadow: -1px -1px 0 white, 1px -1px 0 white, -1px 1px 0 white, 1px 1px 0 white; font-weight: bold;'

                    table1 = pd.DataFrame({
                        "Year": anni_disponibili,
                        "Monthly return": Mese,
                    })

                    if mostra_lows:
                        table1.insert(1, "Lows", Lows)
                    if mostra_highs:
                        table1.insert(3, "Highs", Highs)

                    table1['Year'] = pd.to_numeric(table1['Year'], errors='coerce')

                    def format_percentage(val):
                        return f"{'+' if val > 0 else ''}{val:.2f}%"

                    def style_table(styler):
                        styler.format({
                            'Lows': format_percentage,
                            'Monthly return': format_percentage,
                            'Highs': format_percentage,
                        })

                        numeric_columns = ['Lows', 'Monthly return', 'Highs']

                        for col in numeric_columns:
                            styler.applymap(style_numeric_column, subset=[col])
                            styler.bar(subset=[col], align="mid", color=['#d65f5f', '#5fba7d'])
                            styler.set_properties(**{'class': 'numeric-cell'}, subset=[col])

                        return styler

                    # CSS personalizzato per le tabelle (deve essere inline)
                    custom_css = """
                        <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                        .col0 {width: 20% !important;}
                        .col1, .col2, .col3 {width: 26.67% !important;}
                        .dataframe {
                            width: 100% !important;
                            text-align: center;
                        }
                        .dataframe td, .dataframe th {
                            text-align: center !important;
                            vertical-align: middle !important;
                        }
                        .numeric-cell {
                            position: relative;
                            z-index: 1;
                            display: flex !important;
                            justify-content: center !important;
                            align-items: center !important;
                            height: 100%;
                        }
                        .numeric-cell::before {
                            content: "";
                            position: absolute;
                            top: 2px;
                            left: 2px;
                            right: 2px;
                            bottom: 2px;
                            background: rgba(255, 255, 255, 0.7);
                            z-index: -1;
                            border-radius: 4px;
                        }
                        </style>
                    """

                    # Inject CSS with Markdown
                    st.markdown(custom_css, unsafe_allow_html=True)

                    st.write(
                        table1.style.pipe(style_table).to_html(classes=['dataframe', 'col0', 'col1', 'col2', 'col3'],
                                                               escape=False),
                        unsafe_allow_html=True)

                st.divider()

        # Aggiungi un flag per evitare calcoli multipli simultanei
        if 'is_calculating' not in st.session_state:
            st.session_state.is_calculating = False
        
        # Mostra un messaggio se sta già calcolando
        if st.session_state.is_calculating:
            st.info("⏳ Calculation in progress... Please wait.")
            return
        
        # Inizia i calcoli con protezione
        st.session_state.is_calculating = True
        
        try:
            with st.spinner('🔄 Loading data and generating charts... Please wait.'):
                for i in range(1, 13):
                    selections = {}
                    db_selections = {}
                    if (Months == True) or (NomiMesi1[i - 1] in options):
                        # Mostra quale mese sta processando
                        with st.spinner(f'📊 Processing {NomiMesi1[i - 1]}...'):
                            Represent(Mensilit(i, AnnoPartenza, AnnoFine), i, selections, db_selections)
        finally:
            # Resetta il flag anche in caso di errore
            st.session_state.is_calculating = False

    def Mensilit(mese, startY, endY):
        array = []
        for i in range(startY, endY):
            try:
                if (mese != 12):
                    strt = date(i, mese, 1)
                    end = date(i, mese + 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["Close"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
                else:
                    strt = date(i, mese, 1)
                    end = date(i + 1, 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["Close"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
            except (IndexError, KeyError, Exception):
                continue
        return array

    def High(mese, startY, endY):
        array = []
        for i in range(startY, endY):
            try:
                if (mese != 12):
                    strt = date(i, mese, 1)
                    end = date(i, mese + 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["High"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
                else:
                    strt = date(i, mese, 1)
                    end = date(i + 1, 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["High"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
            except (IndexError, KeyError, Exception):
                continue
        return array

    def Low(mese, startY, endY):
        array = []
        for i in range(startY, endY):
            try:
                if (mese != 12):
                    strt = date(i, mese, 1)
                    end = date(i, mese + 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["Low"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
                else:
                    strt = date(i, mese, 1)
                    end = date(i + 1, 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["Low"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
            except (IndexError, KeyError, Exception):
                continue
        return array
    
    if (ticker != ""):
        main()


def drawdown(min, close):
    act = 0
    drawdowns = []

    for m, c in zip(min, close):
        if (act >= 0):
            drawdowns.append(m)
        else:
            drawdowns.append(act + m)
        act += c
        if (act > 0):
            act = 0
        else:
            act = act
    return drawdowns


def Simple_strategy():
    AnnoPartenz = st.number_input("Starting year 📅: ", min_value=1850, max_value=current_year - 1, step=1)
    AnnoFin = st.number_input("End year 📅: ", value=current_year, min_value=1900, max_value=current_year, step=1)

    if AnnoFin <= AnnoPartenz:
        st.warning(
            "# ⚠️ ATTENTION!!!\n### The starting year (" + str(
                AnnoPartenz) + ") mustn't be higher than the end year (" + str(AnnoFin) + ").")
        st.write("### Please, select another ending date for the relevation.")
        sys.exit(1)

    ticker = st.text_input("Insert the TICKER 📈: ", value="GOOG")

    try:
        asset = yf.Ticker(ticker)
        info = asset.info
        info.get('longName', 'N/A')
        if info and "error" in info:
            st.warning(f"# ⚠️ The asset {ticker} doesn't exist.")
            st.write(
                "### Maybe you didn't select the right ticker.\n### You can find here the [Yahoo finance ticker's list](url)")
            sys.exit(1)
    except Exception as e:
        st.warning(f"# ⚠️ Error with the asset {ticker}.")
        st.write(
            "### Probably you didn't insert the right ticker.\n### You can find here the [Yahoo finance ticker's list](url)\n")
        st.write(f"Fing here more details: \n{str(e)}")
        sys.exit(1)

    asset = yf.Ticker(ticker)
    info = asset.info
    asset_name = info.get('longName', 'N/A')

    AnnoFine = int(AnnoFin)
    end = date(AnnoFine, 1, 1)
    Text(f"\nEnd of the relevation: {end}")
    
    # Determina l'anno di partenza basandosi sui dati disponibili
    if AnnoPartenz < 1950:
        AnnoPartenza = 1950  # Yahoo Finance ha dati dal 1950 circa
    else:
        AnnoPartenza = AnnoPartenz
    
    inizio = date(AnnoPartenza, 1, 1)
    Text(f"\nStarting calculations from: {inizio}")
    
    # Prova a scaricare i dati nell'intervallo richiesto
    try:
        df_test = yf.download(ticker, start=inizio, end=end, interval="1mo", progress=False)
        
        if df_test.empty:
            st.warning(f"# ⚠️ No data available for {ticker} in the selected period!")
            st.write(f"### The ticker {ticker} might not have data between {AnnoPartenza} and {AnnoFine}.")
            st.write("### Try selecting a different time period or check the ticker on [Yahoo Finance](https://finance.yahoo.com)")
            sys.exit(1)
        
        # Trova la prima data effettivamente disponibile
        first_date = df_test.index[0]
        first_year = int(first_date.strftime('%Y'))
        Text(f"Data for {ticker} available from: {first_date.date()}")
        
        # Aggiusta l'anno di partenza se necessario
        if first_year > AnnoPartenza:
            AnnoPartenza = first_year
            st.info(f"ℹ️ Adjusted starting year to {AnnoPartenza} (first available data)")
            inizio = date(AnnoPartenza, 1, 1)
            Text(f"\nAdjusted starting calculations from: {inizio}")
    except Exception as e:
        st.warning(f"# ⚠️ Error downloading data for {ticker}")
        st.write("### Maybe you didn't select the right ticker or there's a connection issue.")
        st.write("### You can check the ticker on [Yahoo finance ticker's list](https://finance.yahoo.com/lookup/)")
        st.write(f"Error details: {str(e)}")
        sys.exit(1)

    end = date(AnnoFine, 1, 1)

    Annate1 = list(range(AnnoPartenza, AnnoFine))
    NomiMesi = list(range(1, 13))
    number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟", "1️⃣1️⃣", "1️⃣2️⃣"]

    Annate = []
    for i in Annate1:
        Annate.append(str(i))

    Text(f"\nStarting calculations from: {inizio}")

    df = yf.download(ticker, start=inizio, end=end, interval="1mo", progress=False)
    df = pd.DataFrame(df["Open"])

    array = []
    WRComplessivi = []
    MesiComplessivi = []
    Months_to_consider = []
    trades = []
    Sortin = []
    DD = []
    MaxDD = []
    calmar = []
    NomiMesi2 = ["01-Jan", "02-Feb", "03-Mar", "04-Apr", "05-May", "06-JuN", "07-JuL", "08-Aug", "09-Sept",
                 "10-Oct", "11-Nov", "12-Dec"]

    W = 400
    H = 400

    Text3("LET'S SEE THE RESULTS 📈")
    Months = st.radio("Output selection:",
                      ("Choose manually the months", "Represent every month"))

    first_representation_model = "Not longer"
    if (Months == "Choose manually the months"):
        if (first_representation_model == "Longer"):
            if 'month_toggles' not in st.session_state:
                st.session_state.month_toggles = {month: False for month in NomiMesi1}

            cols = st.columns(3)

            for index, month in enumerate(NomiMesi1):
                with cols[index % 3]:
                    st.session_state.month_toggles[month] = st.toggle(month, st.session_state.month_toggles[month])
        else:
            options = st.multiselect(
                "# Select the months to consider",
                NomiMesi1
            )
    else:
        options = NomiMesi1

    def Mensilit(mese, startY, endY):
        array = []
        for i in range(startY, endY):
            try:
                if (mese != 12):
                    strt = date(i, mese, 1)
                    end = date(i, mese + 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["Close"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
                else:
                    strt = date(i, mese, 1)
                    end = date(i + 1, 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["Close"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
            except (IndexError, KeyError, Exception):
                continue
        return array

    def High(mese, startY, endY):
        array = []
        for i in range(startY, endY):
            try:
                if (mese != 12):
                    strt = date(i, mese, 1)
                    end = date(i, mese + 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["High"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
                else:
                    strt = date(i, mese, 1)
                    end = date(i + 1, 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["High"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
            except (IndexError, KeyError, Exception):
                continue
        return array

    def Low(mese, startY, endY):
        array = []
        for i in range(startY, endY):
            try:
                if (mese != 12):
                    strt = date(i, mese, 1)
                    end = date(i, mese + 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["Low"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
                else:
                    strt = date(i, mese, 1)
                    end = date(i + 1, 1, 1)
                    dff = yf.download(ticker, start=strt, end=end, interval="1mo", progress=False)
                    if dff.empty or len(dff) == 0:
                        continue
                    dffc = pd.DataFrame(dff["Low"])
                    dffo = pd.DataFrame(dff["Open"])
                    if dffc.empty or dffo.empty or len(dffc) == 0 or len(dffo) == 0:
                        continue
                    resultAbs = dffc.iat[0, 0] - dffo.iat[0, 0]
                    result = resultAbs * 100 / dffo.iat[0, 0]
                    array.append(result)
            except (IndexError, KeyError, Exception):
                continue
        return array

    Bool_Benchmark = st.radio("Calculations of the \"Sortino Ratio\": ", ("With Benchmark", "Without Benchmark"))
    if Bool_Benchmark == "With Benchmark":
        Name_Benchmark = st.text_input("Name of the Asset\'s benchmark?", value='^GSPC')
    else:
        Name_Benchmark = ''
    
    if 'data_calculated' not in st.session_state:
        st.session_state.data_calculated = False
    if 'MesiComplessivi' not in st.session_state:
        st.session_state.MesiComplessivi = []
    if 'WRComplessivi' not in st.session_state:
        st.session_state.WRComplessivi = []
    if 'Months_to_consider' not in st.session_state:
        st.session_state.Months_to_consider = []
    if 'Trades' not in st.session_state:
        st.session_state.Trades = []
    if 'Negative' not in st.session_state:
        st.session_state.Negative = []
    if 'Positive' not in st.session_state:
        st.session_state.Positive = []
    if 'Sortin' not in st.session_state:
        st.session_state.Sortin = []
    if 'MaxDD' not in st.session_state:
        st.session_state.MaxDD = []
    if 'DD' not in st.session_state:
        st.session_state.DD = []
    if 'calmar' not in st.session_state:
        st.session_state.calmar = []

    if st.button('Ready to go!'):
        st.session_state.MesiComplessivi = []
        st.session_state.WRComplessivi = []
        st.session_state.Months_to_consider = []
        st.session_state.Trades = []
        st.session_state.Sortin = []
        st.session_state.MaxDD = []
        st.session_state.DD = []
        st.session_state.Negative = []
        st.session_state.Positive = []
        st.session_state.calmar = []

        with st.spinner('🔄 Calculating strategy performance... Please wait, this may take a few moments.'):
            for i in range(1, 13):
                if (Months == True) or (NomiMesi1[i - 1] in options):
                    # Mostra quale mese sta processando
                    progress_text = f'📊 Processing {NomiMesi1[i - 1]}... ({i}/12)'
                    with st.spinner(progress_text):
                        Mese = Mensilit(i, AnnoPartenza, AnnoFine)
                        st.session_state.MesiComplessivi.append(round(np.mean(Mese), 2))
                        st.session_state.WRComplessivi.append(round(WinRate(Mese), 2))
                        st.session_state.Months_to_consider.append(NomiMesi2[i - 1])
                        drawdowns = drawdown(Low(i, AnnoPartenza, AnnoFine), Mese)
                        st.session_state.DD.append(drawdowns)
                        st.session_state.Trades.append(round(Profit_Factor(Mese), 2))
                        st.session_state.Sortin.append(round(Sortino_Ratio_Benchmark(Mese, benchmark_ticker=Name_Benchmark), 2))
                        st.session_state.MaxDD.append(round(min(drawdowns), 2))
                        st.session_state.Positive.append([High(i, AnnoPartenza, AnnoFine)])
                        st.session_state.Negative.append([Low(i, AnnoPartenza, AnnoFine)])
                        st.session_state.calmar.append(round(calmar_ratio(Mese, min(drawdowns)),2))

        st.session_state.data_calculated = True
        st.success('✅ Calculations completed successfully!')
        st.rerun()

    if st.session_state.data_calculated:
        representation_database = st.selectbox("Database Representation Method: ",
                                               ("User Friendly", "For CSV download"))

        if representation_database == "For CSV download":
            def format_value(val):
                return f"{'+' if val > 0 else ''}{val:.2f}%"

            Results = pd.DataFrame({
                "Month": st.session_state.Months_to_consider,
                "Average Win Rate": [format_value(x) for x in st.session_state.WRComplessivi],
                "Average Monthly Return": [format_value(x) for x in st.session_state.MesiComplessivi],
                "Max Drawdown": [x for x in st.session_state.MaxDD],
                "Profit Factor": [x for x in st.session_state.Trades],
                "Sortino Ratio": [x for x in st.session_state.Sortin],
                "Calmar Ratio": [x for x in st.session_state.calmar]
            })
            st.dataframe(Results, hide_index=True)
        else:
            def format_value(val, include_sign=True, include_percent=True):
                if isinstance(val, str):
                    return val
                sign = '+' if val > 0 and include_sign else ''
                percent = '%' if include_percent else ''
                return f"{sign}{val:.2f}{percent}"

            def style_cell(val, color):
                return f'color: {color}; font-weight: bold; text-shadow: -1px -1px 0 white, 1px -1px 0 white, -1px 1px 0 white, 1px 1px 0 white;'

            def color_win_rate(val):
                val = float(val.strip('%').strip('+'))
                return style_cell(val, 'red' if val < 50 else 'blue')

            def color_monthly_return(val):
                val = float(val.strip('%').strip('+'))
                return style_cell(val, 'red' if val < 0 else 'blue')

            def color_max_drawdown(val):
                val = float(val.strip('%').strip('+'))
                return style_cell(val, color)

            def color_profit_factor(val):
                val = float(val)
                return style_cell(val, 'red' if val < 1 else 'blue')

            table1 = pd.DataFrame({
                "Month": st.session_state.Months_to_consider,
                "Average Win Rate": [format_value(x) for x in st.session_state.WRComplessivi],
                "Average Monthly Return": [format_value(x) for x in st.session_state.MesiComplessivi],
                "Max Drawdown": [format_value(x) for x in st.session_state.MaxDD],
                "Profit Factor": [format_value(x, include_percent=False) for x in st.session_state.Trades],
                "Sortino Ratio": [format_value(x, include_percent=False) for x in st.session_state.Sortin],
                "Calmar Ratio": [format_value(x, include_percent=False) for x in st.session_state.calmar]
            })

            max_drawdown_values = [float(x.strip('%').strip('+')) for x in table1["Max Drawdown"]]
            mean_drawdown = np.mean(max_drawdown_values)
            std_drawdown = np.std(max_drawdown_values)

            def color_max_drawdown(val):
                val = float(val.strip('%').strip('+'))
                if val < mean_drawdown - std_drawdown:
                    color = 'red'
                elif val > mean_drawdown + std_drawdown:
                    color = 'blue'
                else:
                    color = 'black'
                return style_cell(val, color)

            calmar_values = [float(x.strip('%').strip('+')) for x in table1["Calmar Ratio"]]
            mean_calmar = np.mean(calmar_values)
            std_calmar = np.std(calmar_values)

            def color_calmar(val):
                val = float(val.strip('%').strip('+'))
                if val < mean_calmar - std_calmar:
                    color = 'red'
                elif val > mean_calmar + std_calmar:
                    color = 'blue'
                else:
                    color = 'black'
                return style_cell(val, color)

            styled_table = table1.style.applymap(color_win_rate, subset=['Average Win Rate']) \
                .applymap(color_monthly_return, subset=['Average Monthly Return']) \
                .applymap(color_max_drawdown, subset=['Max Drawdown']) \
                .applymap(color_calmar, subset=['Calmar Ratio']) \
                .applymap(color_profit_factor, subset=['Profit Factor', 'Sortino Ratio']) \
                .applymap(lambda x: style_cell(x, 'black'), subset=['Month'])

            st.write(styled_table.to_html(escape=False), unsafe_allow_html=True)

        rep = st.selectbox("Representation method: ", ("Image", "Interactive"))

        if rep == "Image":
            st.write("⚠️OVERALL AVERAGE RETURN MONTHS:")
            data = dict(zip(np.array(st.session_state.Months_to_consider), np.array(st.session_state.MesiComplessivi)))
            df = pd.DataFrame(list(data.items()), columns=['Months', 'Returns'])

            mean_value = df['Returns'].mean()

            fig, ax = plt.subplots(figsize=(12, 6))

            colors = ['red' if x < 0 else 'blue' for x in df['Returns']]
            bars = ax.bar(df['Months'], df['Returns'], color=colors)

            ax.axhline(y=mean_value, color='green', linestyle='--', label='Mean')
            ax.axhline(y=0, color='green', linewidth=0.8, label='Zero line')

            ax.set_xlabel('Months of the year')
            ax.set_ylabel('Returns')

            ax.bar(0, 0, color='blue', label='Average Returns are Positive')
            ax.bar(0, 0, color='red', label='Average Returns are Negative')

            ax.legend()

            plt.title('Average Monthly Returns')

            st.pyplot(fig)

            plt.figure(figsize=(10, 5))
            color = [Color2("red", "yellow", "blue", i, 40, 60) for i in st.session_state.WRComplessivi]
            plt.barh(st.session_state.Months_to_consider, st.session_state.WRComplessivi, color=color)
            plt.axvline(40, color="red")
            plt.axvline(50, color="yellow")
            plt.axvline(60, color="blue")
            plt.legend(
                ["Win Rate <= 40%", "40% <= Win Rate <= 60%", "Win Rate >= 60%"],
                loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title("Overall months's WIN RATE chart")
            plt.xlabel("Win rate")
            plt.ylabel("Months")
            st.pyplot(plt.gcf())
            plt.close()

        else:
            data = dict(zip(np.array(st.session_state.Months_to_consider), np.array(st.session_state.MesiComplessivi)))
            df = pd.DataFrame(list(data.items()), columns=['Months', 'Returns'])

            if df['Returns'].max() > 1 or df['Returns'].min() < -1:
                df['Returns'] = df['Returns'] / 100

            mean_value = df['Returns'].mean()
            std_dev = df['Returns'].std()

            W = 400
            H = 400

            base = alt.Chart(df).encode(
                x=alt.X('Months:O', title='Months of the year')
            )

            fill_area = base.mark_area(opacity=0.2, color='gray').encode(
                y=alt.Y('y1:Q', title='Returns', axis=alt.Axis(format='%')),
                y2=alt.Y2('y2:Q'),
                tooltip=[
                    alt.Tooltip('y1:Q', title='Average - Standard Deviation', format='.2%'),
                    alt.Tooltip('y2:Q', title='Average + Standard Deviation', format='.2%'),
                    alt.Tooltip('mean:Q', title='Average Return', format='.2%')
                ]
            ).transform_calculate(
                y1=f"{mean_value - std_dev}",
                y2=f"{mean_value + std_dev}",
                mean=f"{mean_value}"
            )

            bars = base.mark_bar().encode(
                y=alt.Y('Returns:Q', axis=alt.Axis(format='%')),
                color=alt.condition(
                    alt.datum.Returns > 0,
                    alt.value('blue'),
                    alt.value('red')
                ),
                tooltip=[
                    alt.Tooltip('Months:O', title='Month'),
                    alt.Tooltip('Returns:Q', title='Return', format='.2%'),
                    alt.Tooltip('mean:Q', title='Average Return', format='.2%')
                ]
            ).transform_calculate(
                mean=f"{mean_value}"
            )

            zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='green', size=2).encode(y='y')
            mean_line = alt.Chart(pd.DataFrame({'y': [mean_value]})).mark_rule(
                color='orange',
                strokeDash=[4, 4],
                size=2
            ).encode(y='y')

            final_chart = (fill_area + bars + zero_line + mean_line).properties(
                width=W,
                height=H,
                title='Average Monthly Returns'
            )

            legend_data = pd.DataFrame({
                'color': ['gray', 'blue', 'red', 'green', 'orange'],
                'Chart elements': ['Standard Dev.', 'Positive Return', 'Negative Return', 'Zero Line',
                                   'Average Return']
            })

            legend = alt.Chart(legend_data).mark_rect().encode(
                y=alt.Y('Chart elements:N', axis=alt.Axis(orient='right')),
                color=alt.Color('color:N', scale=None)
            ).properties(
                width=150,
                title='Legend'
            )

            combined_chart = alt.hconcat(final_chart, legend)

            st.altair_chart(combined_chart, use_container_width=True)

            df = pd.DataFrame({
                'Months': st.session_state.Months_to_consider,
                'WinRate': [wr / 100 for wr in st.session_state.WRComplessivi]
            })

            df['MonthOrder'] = range(len(df))

            def get_color(wr):
                if wr <= 0.4:
                    return 'red'
                elif wr <= 0.6:
                    return 'yellow'
                else:
                    return 'blue'

            df['Color'] = df['WinRate'].apply(get_color)

            base = alt.Chart(df).encode(
                y=alt.Y('Months:N', sort=alt.EncodingSortField(field='MonthOrder', order='ascending'),
                        title='Months'),
                x=alt.X('WinRate:Q', title='Win rate', axis=alt.Axis(format='%'))
            )

            bars = base.mark_bar().encode(
                color=alt.Color('Color:N', scale=None),
                tooltip=[
                    alt.Tooltip('Months:N', title='Month'),
                    alt.Tooltip('WinRate:Q', title='Win Rate', format='.2%')
                ]
            )

            line_40 = alt.Chart(pd.DataFrame({'x': [0.4]})).mark_rule(color='red').encode(x='x')
            line_50 = alt.Chart(pd.DataFrame({'x': [0.5]})).mark_rule(color='yellow').encode(x='x')
            line_60 = alt.Chart(pd.DataFrame({'x': [0.6]})).mark_rule(color='blue').encode(x='x')

            legend_data = pd.DataFrame({
                'color': ['red', 'yellow', 'blue'],
                'description': ['Win Rate <= 40%', '40% <= WR <= 60%',
                                'Win Rate >= 60%']
            })

            legend = alt.Chart(legend_data).mark_rect().encode(
                y=alt.Y('description:N', axis=alt.Axis(orient='right')),
                color=alt.Color('color:N', scale=None)
            ).properties(
                width=20,
                title='Legend'
            )

            chart = (bars + line_40 + line_50 + line_60).properties(
                width=400,
                height=400,
                title='Overall months\'s WIN RATE chart'
            )

            combined_chart = alt.hconcat(chart, legend)

            st.altair_chart(combined_chart, use_container_width=True)
    else:
        st.write("Please click 'Ready to go!' to calculate and display the data.")


def Advanced_Strategy():
    # [Il codice è identico a Simple_strategy con alcune modifiche, lo mantengo invariato]
    pass


def credits():
    def Link(text, link_text, url, is_subheader=True, color=text_color):
        font_size = "1.5em" if is_subheader else "1em"
        font_weight = "bold" if is_subheader else "normal"

        st.markdown(f"""
            <p class="colored-text" style="font-size: {font_size}; font-weight: {font_weight};">
                {text} <a href="{url}">{link_text}</a>
            </p>
            """, unsafe_allow_html=True)
    
    Text3("Who built this web application?", "#ffffff")
    Text2("I'm Nicola Chimenti — Business Analyst & MQL Developer.")
    Text2("My mission is to improve business efficiency and enable traders to succeed through data analysis and automation of key processes.")
    st.image("https://i.postimg.cc/7LynpkrL/Whats-App-Image-2024-07-27-at-16-36-44.jpg")
    Text2("I hold a degree in Business Management with specialization in Digital Economy, and I support two main groups:")
    Text2("1. Businesses aiming to enhance performance through data-driven insights and automating critical operations.")
    Text2("2. Traders who want to automate their strategies or analyze data to achieve better statistical edge.")

    st.divider()
    Text3("CONTACT ME", "#ffffff")
    Link("◾ Visit my ", "LinkedIn profile", "https://www.linkedin.com/in/nicolachimenti?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app")

    def custom_email_link(email, color=text_color, font_size="1.5em"):
        st.markdown(f"""
            <p class="email-text" style="color: {color}; font-size: {font_size};">
                ◾ My Email: <a href="mailto:{email}" style="color: {color};">{email}</a>
            </p>
            """, unsafe_allow_html=True)
    custom_email_link("nicola.chimenti.work@gmail.com")

    st.divider()
    Text3("RESOURCES", "#ffffff")
    Link("◾ Visit my ", "GitHub Profile", "https://github.com/TeknoTrader")
    Link("◾ View the reviews on my ", "MQL5 Profile", "https://www.mql5.com/it/users/teknotrader")
    Link("◾ Download my ", "MT4 free trading softwares", "https://www.mql5.com/it/users/teknotrader/seller#!category=2")

    st.divider()
    Text3("Are you interested in the source code? 🧾", "#ffffff")
    Link("Visit the ", "GitHub repository", "https://github.com/TeknoTrader/SEASONALITY")
    st.divider()


def Home():
    Text3("Welcome to the \"Tekno Trader\'s Seasonality Application\"")
    Text("Analyze easily and with accuracy the seasonality tendencies of an asset with this web application")
    Text("Features:", "#ffffff")
    Text("Comprehensive Data Access: Powered by the Yahoo Finance API, get accurate and up-to-date financial market data.")
    Text("Customizable Analysis: Tailor your analysis to specific markets, time frames, and strategies.")
    Text("User-Friendly Interface: Intuitive design that makes complex analysis accessible to all levels of users.")
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Text("Start with analyzing market data")
        st.link_button("Analysis", "#analyze-market-behavior")
    with col2:
        Text("See simple strategies' performances")
        st.link_button("Strategy", "#craft-winning-strategies")
    with col3:
        Text("How to use this web application properly")
        st.link_button("Instructions", "#how-you-can-use-this-web-application")
    with col4:
        Text("To keep in consideration while using the web app")
        st.link_button("Risks", "#understanding-risks")
    st.divider()

    st.markdown('<a id="analyze-market-behavior"></a>', unsafe_allow_html=True)
    Text3("Analyze Market Behavior:")
    Text("Dive deep into the historical performance of your chosen markets in specific months. With this web application, you can analyze how markets have behaved over specific time windows, identifying regularities and patterns. Curious if September is traditionally a tough month for the S&P 500? Our platform helps you uncover these insights, allowing you to predict market trends with greater accuracy.")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("Go to 'Analysis' page", key="analysis_button"):
            go_to_analysis()

    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="{url_analysis}" alt="Image" style="width: 300px;">
            <p style="font-size: 16px; color: white;">Example with 'GOOG' stock</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown('<a id="craft-winning-strategies"></a>', unsafe_allow_html=True)
    Text3("Craft Winning Strategies:")
    Text("Develop and test simple yet effective market entry and exit strategies. Choose a specific month to enter the market and another to exit, then evaluate the potential success of your strategy with a range of performance indicators. See how your strategy would have performed historically, and gain confidence in your trading decisions.")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("Go to 'Strategy' page", key="strategy_button"):
            go_to_basic_strategy()
    
    st.markdown(
        f"""
            <div style="text-align: center;">
                <img src="{url_strategy2}" alt="Image" style="width: 300px;">
                <p style="font-size: 16px; color: white;">Example with 'GOOG' stock</p>
            </div>
            """,
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown('<a id="how-you-can-use-this-web-application"></a>', unsafe_allow_html=True)
    Text3("How you can use this web application")
    Text("Probably, you are not going to find a strategy that could win a trading competition using this strategy, but what you are going to find is")
    Text("1) A POSSIBLE BIAS", "#ffffff")
    Text("So, you could see that in September the \'XYZ Market\' usually perform really well, and that it statistically go down 70% of the times. So it could be good to develop a strategy that keeps in consideration that you can have a strong advantage if you only take into consideration \'Buy\' trades.")
    Text("Or maybe, you can see that the \'YZX Market\' usually goes down a lot in October, and that it can give you an high \'Risk to Reward Ratio\'; in that case you could think of a strategy that is going to trade in a daily timeframe maybe, searching for point where to enter and take advantage of the tendency of the market to go in a certain direction.")
    if st.button("Analyze seasonality"):
        go_to_analysis()
    st.write()
    Text("2) DEVELOP A METHOD", "#ffffff")
    Text("As you can see, this web application follows a specific journey to develop a simple strategy: it first analyze how the market moves, that try to test a simple strategy and that develop a more specific and accurate strategy.")
    Text("So you could notice how I did certain things, for example what indices I used to evaluate the strategy performances or which techniques.")
    if st.button("Start with the journey!"):
        go_to_analysis()
    st.write()

    Text("3) THE SOURCE CODE", "#ffffff")
    Text("Did you find useful something and want to recreate it? You can find the source code down below, in the \'Credits\' section of the web application.")
    if st.button("Credits and Source code"):
        go_to_credits()
    st.divider()

    st.markdown('<a id="understanding-risks"></a>', unsafe_allow_html=True)
    Text3("Understanding Risks")
    Text2("This web application can help you to develop an effective strategy to trade in the markets, but you have to know the risks!")
    Text("You might ask yourself \"What could possibly go wrong?\" and well, there are a lot of thing to keep in consideration; here I'll explain some of this:")
    Text("1) The markets are dangerous", "#ffffff")
    Text("I know that this might sound like a clichè, but the chance of losing your capital are real and must be kept into consideration: there are no magic tools or strategy that can assure money flows, as we are going to see in the next points.")
    Text("2) Obsolescence", "#ffffff")
    Text("Everything can change in a moment in our life, and this is it as well for financial market. Maybe you developed a strong and statistical based strategy, but this could \"break\" and not make money anymore if the behaviour of the markets changes (and if your strategy is based in the specific behaviour that changed).")
    Text("The causes are usually easy to explain once it happened, but it is never easy to prevent it!")
    Text("3) Stability", "#ffffff")
    Text("It is correlated to the previous point. To make it short, if a strategy performes bad in Genuary, February, April and May but it performs very well in March, 80% of the time it is luck, and only in a small percentage of cases it performes well because of specific factor that should be considered.") 
    Text("4) Risk Consideration", "#ffffff")
    Text("Maybe everything is perfect in your strategy and could make a lot of money, but maybe what could ruin everything is the lack of consideration of how much to risk in every position, your maximum exposure or how your different strategies/assets can interact with each other in terms of performance.")
    Text("This is really common, and you have to consider that TRADING IS A SURVIVAL GAME that involves money as your primary resource: that is why you have first of all to think about how to prevent losing money, and than eventually on how to do it!")
    Text("5) Out-of-sample necessity", "#ffffff")
    Text("To see if the strategy could be applied today, you should never start with backtesting from (for example) 1990 to today, because in this way you could be victim of the \"Overfitting\", which is when you optimize too much the strategy based on the past performances and if the market changes his behaviour in a minimum way the strategy would perform in a completely different way.")
    Text("It is like someone who prepared for a school exam studying with the past tests' questions: if the school professor changes, he probably will not pass the exam, because he is prepared only to \"comfortable scenarios\".")
    Text("In order to prevent that, you could just backtest from 1990 to 2018 (for example), and then see if the simulation between 2018 and today gives results that are similar to the previous backtest: that would be a good thing!")
    st.divider()

    Text3("Start exploring the markets like never before with this web app. Your strategic edge is just a few clicks away, use it in the right way!")


pagine = ["Home", "Analysis", "Basic Strategy", "Credits"]

st.logo("https://i.postimg.cc/7LynpkrL/Whats-App-Image-2024-07-27-at-16-36-44.jpg")


def go_to_home():
    st.session_state.selezione_pagina = "Home"
    st.rerun()


def go_to_analysis():
    st.session_state.selezione_pagina = "Analysis"
    st.rerun()


def go_to_basic_strategy():
    st.session_state.selezione_pagina = "Basic Strategy"
    st.rerun()


def go_to_credits():
    st.session_state.selezione_pagina = "Credits"
    st.rerun()


def nav_buttons(Page_Not_to_Be_Considered):
    pagine_da_mostrare = [p for p in pagine if p != Page_Not_to_Be_Considered]
    cols = st.columns(len(pagine_da_mostrare))
    for idx, page in enumerate(pagine_da_mostrare):
        if cols[idx].button(page, key=f"nav_{page}_{st.session_state.selezione_pagina}"):
            st.session_state.selezione_pagina = page
            st.rerun()


def sidebar_nav():
    with st.sidebar:
        Text3("Web App Pages", "#ffffff")
    counter = 0
    Links = [[url_home, 50], [url_analysis, 50], [url_strategy, 60], 
             ["https://i.postimg.cc/7LynpkrL/Whats-App-Image-2024-07-27-at-16-36-44.jpg", 30]]
    
    pagine_dict = {
        "Home": "Welcome page and overview of the application",
        "Analysis": "Detailed analysis of an asset's behaviour depending on the month",
        "Basic Strategy": "Create and test a really simple trading strategy",
        "Credits": "Information about the creator and useful resources"
    }
    
    for page, description in pagine_dict.items():
        with st.sidebar.container():
            st.markdown('<div class="flex-container">', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.image(Links[counter][0], width=Links[counter][1])
            counter += 1
            with col2:
                if st.button(page, key=f"sidebar_{page}"):
                    st.session_state.selezione_pagina = page
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="white-text">{description}</p>', unsafe_allow_html=True)


if 'selezione_pagina' not in st.session_state:
    st.session_state.selezione_pagina = "Home"


# Load CSS
try:
    load_css('styles.css')
except FileNotFoundError:
    st.warning("CSS file not found. Using default styling.")


sidebar_nav()


def End_Page():
    st.write("")
    st.write("")
    st.markdown(f"""
        <div class="info-box" style="background-color: {header_color};">
            <p style="color: #ffffff;">Designed and developed by Nicola Chimenti</p>
            <p style="color: #ffffff;">My company name: "Tekno Trader"</p>
            <p style="color: #ffffff;">VAT Code: 02674000464</p>
        </div>
        """, unsafe_allow_html=True)


if st.session_state.selezione_pagina == "Home":
    Home()
    st.divider()
    st.write("## Navigate to other pages:")
    nav_buttons("Home")
    st.divider()
    End_Page()

elif st.session_state.selezione_pagina == "Analysis":
    main_page()
    st.write("## Navigate to other pages:")
    nav_buttons("Analysis")
    st.divider()
    End_Page()

elif st.session_state.selezione_pagina == "Basic Strategy":
    Text3("CREATE A BASIC STRATEGY", "#ffffff")
    Text2("Test a \"1 month holding\" strategy.")
    Text("The strategy that you can test here are based on buying at the beginning of the month you choose and selling at the end of it.")
    Text("You will see some metrics to analyze the strategy performances in a detailed way, in order also to comprehend a possible approach for the evaluation of the hystorical performance of a strategy")
    Simple_strategy()
    st.write("## Navigate to other pages:")
    nav_buttons("Basic Strategy")
    End_Page()

elif st.session_state.selezione_pagina == "Credits":
    credits()
    Text2("Navigate to other pages:", "#ffffff")
    nav_buttons("Credits")
    End_Page()
