import streamlit as st
import pandas as pd
import folium
import matplotlib.pyplot as plt
import requests
from streamlit.components.v1 import html
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score 

import streamlit as st



st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 12px;
    backdrop-filter: blur(8px);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="typing">🌾 AgriGIS AI System</div>', unsafe_allow_html=True)

st.markdown(
    """
    <style>

/* =========================
   1. PAGE FADE-IN
========================= */
.stApp {
    animation: fadeIn 1.2s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0px); }
}

/* =========================
   2. FLOATING LEAVES BACKGROUND
========================= */
body::before {
    content: "🌿 🌾 🌱 🌿 🌾 🌱 🌿 🌾";
    position: fixed;
    top: -50px;
    left: 0;
    width: 100%;
    font-size: 30px;
    opacity: 0.15;
    animation: floatLeaves 12s linear infinite;
    pointer-events: none;
}

@keyframes floatLeaves {
    0% { transform: translateY(-50px); }
    100% { transform: translateY(100vh); }
}

/* =========================
   3. GLOWING BUTTONS
========================= */
.stButton > button {
    border-radius: 10px;
    transition: 0.3s ease;
    box-shadow: 0 0 10px rgba(0,255,100,0.3);
}

.stButton > button:hover {
    transform: scale(1.08);
    background-color: #4CAF50;
    color: white;
    box-shadow: 0 0 20px rgba(0,255,100,0.8);
}

/* =========================
   4. HOVER CARDS
========================= */
div[data-testid="column"] {
    transition: 0.3s ease;
    padding: 10px;
    border-radius: 15px;
}

div[data-testid="column"]:hover {
    transform: scale(1.04);
    box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
}

/* =========================
   5. TYPING EFFECT (TITLE)
========================= */
.typing {
    width: 0;
    white-space: nowrap;
    overflow: hidden;
    border-right: 3px solid #4CAF50;
    animation: typing 3s steps(30) forwards, blink 0.7s infinite;
    font-size: 32px;
    font-weight: bold;
}

@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes blink {
    50% { border-color: transparent; }
}

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AgriGIS AI Portfolio",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# LANDING PAGE (HERO SECTION)
# -----------------------------

def landing_page():
    st.markdown("""
    <style>
    .hero {
        text-align: center;
        padding: 60px 20px;
        background: linear-gradient(135deg, #d4fc79, #96e6a1);
        border-radius: 20px;
        color: #1e3d2f;
    }
    .title {
        font-size: 50px;
        font-weight: bold;
    }
    .subtitle {
        font-size: 20px;
        margin-top: 10px;
    }
    </style>

    <div class='hero'>
        <div class='title'>🌾 AgriGIS AI System</div>
        <div class='subtitle'>Smart Agriculture • Crop Prediction • AI Insights for Farmers</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Crop Prediction")
        st.write("AI model to predict best crops based on soil and climate.")

    with col2:
        st.subheader("🌍 GIS Mapping")
        st.write("Visualize agricultural data using location-based insights.")

    with col3:
        st.subheader("📈 Yield Analysis")
        st.write("Analyze and improve crop yield using historical data.")

    st.write("---")



# =========================
# PAGE CONFIG
# =========================

import streamlit as st
st.set_page_config(page_title="Agri AI App", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #2e7d32;
        text-align: center;
        margin-top: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🌱 Agri AI Smart Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">  Smart Farming with AI</div>', unsafe_allow_html=True)
st.set_page_config(

    page_title="AI-GIS Agriculture Dashboard",

    layout="wide"

)

# =========================
# SIDEBAR NAVIGATION
# =========================

st.sidebar.title("🌾 Navigation")

page = st.sidebar.radio(

    "Go To",

    [

        "Dashboard",

        "Prediction",

        "Analytics",

        "GIS Map"

    ]

)

# =========================
# TITLE
# =========================

st.title("🌍 AI-GIS Agriculture Dashboard")

st.markdown(
    "### Smart Agriculture Monitoring using AI, GIS and Live Weather Data"
)

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("data/yield_df.csv")

# Paste your OpenWeather API Key here

API_KEY = "68e010a3b258a570af5eab419603af09"

# =========================
# DASHBOARD PAGE
# =========================

if page == "Dashboard":

    st.header("📊 Crop Dataset")

    st.dataframe(data)

    # =========================
    # TEMPERATURE GRAPH
    # =========================

    st.header("🌡 Average Temperature Analysis")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(

        range(20),

        data['avg_temp'].head(20)

    )

    ax.set_xlabel("Data Index")

    ax.set_ylabel("Average Temperature")

    ax.set_title("Average Temperature Analysis")

    st.pyplot(fig)

    # =========================
    # RAINFALL VS YIELD
    # =========================

    st.header("🌧 Rainfall vs Crop Yield")

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.plot(

        data['average_rain_fall_mm_per_year'].head(20),

        label="Rainfall"

    )

    ax2.plot(

        data['hg/ha_yield'].head(20),

        label="Crop Yield"

    )

    ax2.set_xlabel("Data Index")

    ax2.set_ylabel("Values")

    ax2.set_title("Rainfall vs Crop Yield")

    ax2.legend()

    st.pyplot(fig2)

# =========================
# PREDICTION PAGE
# =========================

elif page == "Prediction":

    st.header("🤖 Crop Yield Prediction")

    # =========================
    # CROP SELECTION
    # =========================

    crop_list = data['Item'].unique()

    selected_crop = st.selectbox(

        "🌾 Select Crop",

        crop_list

    )

    st.success(f"Selected Crop: {selected_crop}")

    # =========================
    # FILTER DATA
    # =========================

    crop_data = data[data['Item'] == selected_crop]

    # =========================
    # MACHINE LEARNING MODEL
    # =========================

    X = crop_data[
        ['average_rain_fall_mm_per_year', 'avg_temp']
    ]

    y = crop_data['hg/ha_yield']

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    # =========================
    # MODEL ACCURACY
    # =========================

    y_pred = model.predict(X_test)

    accuracy = r2_score(y_test, y_pred)

    st.header("📈 Model Accuracy")

    st.success(f"AI Model Accuracy: {accuracy:.2f}")

    # =========================
    # LIVE WEATHER API
    # =========================

    cities = [

        "Hyderabad",

        "Chennai",

        "Bangalore",

        "Mumbai",

        "Delhi"

    ]

    selected_location_prediction = st.selectbox(

        "📍 Select Location",

        cities

    )

    # Weather API URL

    url = f"https://api.openweathermap.org/data/2.5/weather?q={selected_location_prediction}&appid={API_KEY}&units=metric"

    # Fetch Weather Data

    response = requests.get(url)

    weather_data = response.json()

    # =========================
    # API RESPONSE
    # =========================

    if response.status_code == 200:

        temperature = weather_data['main']['temp']

        rainfall = weather_data.get(

            'rain',

            {}

        ).get(

            '1h',

            0

        )

        humidity = weather_data['main']['humidity']

        wind_speed = weather_data['wind']['speed']

        weather_condition = weather_data['weather'][0]['main']

        # =========================
        # WEATHER DASHBOARD
        # =========================

        st.header("🌦 Live Weather Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(

                "🌡 Temperature",

                f"{temperature} °C"

            )

        with col2:

            st.metric(

                "🌧 Rainfall",

                f"{rainfall} mm"

            )

        with col3:

            st.metric(

                "💧 Humidity",

                f"{humidity}%"

            )

        with col4:

            st.metric(

                "🌪 Wind Speed",

                f"{wind_speed} m/s"

            )

        st.info(
            f"Current Weather Condition: {weather_condition}"
        )

    else:

        st.error("Weather API Error")

        st.write(weather_data)

        temperature = 25

        rainfall = 0

    # =========================
    # CROP RECOMMENDATION
    # =========================

    st.header("🌾 Recommended Crop")

    recommended_crop = "No Recommendation"

    if temperature > 30 and rainfall > 5:

        recommended_crop = "Rice"

    elif temperature > 25 and rainfall < 3:

        recommended_crop = "Maize"

    elif temperature < 25 and rainfall > 2:

        recommended_crop = "Wheat"

    else:

        recommended_crop = "Potato"

    st.success(f"Recommended Crop: {recommended_crop}")

    # =========================
    # WEATHER RISK ALERTS
    # =========================

    st.header("⚠ Weather Risk Alerts")

    if temperature > 35:

        st.error("High Temperature Alert!")

    elif rainfall < 1:

        st.warning("Low Rainfall Alert!")

    else:

        st.success("Weather Conditions are Stable")

    # =========================
    # YIELD PREDICTION
    # =========================

    if st.button("🚀 Predict Yield"):

        prediction = model.predict(

            [[rainfall, temperature]]

        )

        # Convert hectogram/hectare to kg/hectare

        yield_kg = prediction[0] / 10

        st.success(

            f"Predicted Crop Yield: {yield_kg:.2f} kg/hectare"

        )

# =========================
# ANALYTICS PAGE
# =========================

elif page == "Analytics":

    st.header("📊 Agricultural Analytics Dashboard")

    crop_list = data['Item'].unique()

    selected_crop_analytics = st.selectbox(

        "🌾 Select Crop for Analytics",

        crop_list

    )

    analytics_data = data[
        data['Item'] == selected_crop_analytics
    ]

    st.success(
        f"Selected Crop: {selected_crop_analytics}"
    )

    # =========================
    # TEMPERATURE ANALYTICS
    # =========================

    st.subheader("🌡 Average Temperature Analysis")

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.plot(

        analytics_data['avg_temp'].head(30)

    )

    ax3.set_xlabel("Data Index")

    ax3.set_ylabel("Temperature")

    ax3.set_title("Temperature Analytics")

    st.pyplot(fig3)

    # =========================
    # RAINFALL ANALYTICS
    # =========================

    st.subheader("🌧 Rainfall Analysis")

    fig4, ax4 = plt.subplots(figsize=(10, 5))

    ax4.plot(

        analytics_data[
            'average_rain_fall_mm_per_year'
        ].head(30)

    )

    ax4.set_xlabel("Data Index")

    ax4.set_ylabel("Rainfall")

    ax4.set_title("Rainfall Analytics")

    st.pyplot(fig4)

    # =========================
    # YIELD ANALYTICS
    # =========================

    st.subheader("🌾 Crop Yield Analysis")

    fig5, ax5 = plt.subplots(figsize=(10, 5))

    ax5.plot(

        analytics_data['hg/ha_yield'].head(30)

    )

    ax5.set_xlabel("Data Index")

    ax5.set_ylabel("Crop Yield")

    ax5.set_title("Yield Analytics")

    st.pyplot(fig5)

# =========================
# GIS MAP PAGE
# =========================

elif page == "GIS Map":

    st.header("🗺 GIS Crop Monitoring Map")

    # =========================
    # LOCATION SELECTION
    # =========================

    locations = {

        "Hyderabad": [17.3850, 78.4867],

        "Chennai": [13.0827, 80.2707],

        "Bangalore": [12.9716, 77.5946],

        "Mumbai": [19.0760, 72.8777],

        "Delhi": [28.7041, 77.1025]

    }

    selected_location = st.selectbox(

        "📍 Select Location",

        list(locations.keys())

    )

    latitude = locations[selected_location][0]

    longitude = locations[selected_location][1]

    # =========================
    # LOCATION TABLE
    # =========================

    location_data = pd.DataFrame({

        'Location': [selected_location],

        'Latitude': [latitude],

        'Longitude': [longitude]

    })

    st.header("📌 Location Coordinates")

    st.table(location_data)

    # =========================
    # CREATE MAP
    # =========================

    m = folium.Map(

        location=[latitude, longitude],

        zoom_start=6

    )

    # =========================
    # ADD MARKER
    # =========================

    folium.Marker(

        [latitude, longitude],

        popup=selected_location

    ).add_to(m)

    # =========================
    # DISPLAY MAP
    # =========================

    map_html = m._repr_html_()

    html(map_html, height=500)