from __future__ import annotations

import importlib
from datetime import date

from src.models.predict import InferenceEngine


try:
    gr = importlib.import_module("gradio")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Gradio is not installed. Install dependencies with: pip install -r requirements.txt"
    ) from exc


engine = InferenceEngine()
provinces = engine.metadata.get("provinces", [])
metrics = engine.metadata.get("metrics", {})


def get_weather_status(rain: float) -> tuple[str, str]:
    if rain >= 5.0:
        return "Rainy", "🌧️"
    if rain >= 0.5:
        return "Cloudy", "⛅"
    return "Sunny", "☀️"


def render_weather_card(weather_label: str, weather_icon: str, rain: float) -> str:
    return (
        "<div style='width:100%;height:85px;padding:10px 12px;border-radius:8px;"
        "background:linear-gradient(135deg,#f8fbff,#eef6ff);border:1px solid #d9e8ff;"
        "box-sizing:border-box;display:flex;align-items:center;gap:10px;'>"
        f"<div style='width:40px;height:40px;flex:0 0 40px;display:flex;align-items:center;justify-content:center;"
        "font-size:1.8rem;'>"
        f"{weather_icon}</div>"
        "<div style='flex:1;'>"
        "<div style='font-size:0.7rem;color:#5f6b7a;'>Weather</div>"
        f"<div style='font-size:0.95rem;font-weight:700;color:#1f2a37;'>{weather_label}</div>"
        "</div></div>"
    )


def render_temp_card(temp_value: float) -> str:
    return (
        "<div style='width:100%;height:85px;padding:10px 12px;border-radius:8px;"
        "background:linear-gradient(135deg,#f0f4f8,#e8eff8);border:1px solid #c7dce8;"
        "box-sizing:border-box;display:flex;align-items:center;gap:10px;'>"
        "<div style='width:40px;height:40px;flex:0 0 40px;display:flex;align-items:center;justify-content:center;"
        "font-size:1.8rem;'>🌡️</div>"
        "<div style='flex:1;'>"
        "<div style='font-size:0.7rem;color:#5f6b7a;'>Temperature</div>"
        f"<div style='font-size:0.95rem;font-weight:700;color:#1f2a37;'>{temp_value}°C</div>"
        "</div></div>"
    )


def predict_temp_max(
    selected_date,
    province: str,
    temp_min: float,
    rain: float,
    wind_speed: float,
    lat: float,
    lon: float,
):
    selected = selected_date if isinstance(selected_date, date) else date.fromisoformat(str(selected_date))
    prediction = engine.predict(
        temp_min=temp_min,
        rain=rain,
        wind_speed=wind_speed,
        lat=lat,
        lon=lon,
        year=selected.year,
        month=selected.month,
        day=selected.day,
        dayofweek=selected.weekday(),
        province=province,
    )
    weather_label, weather_icon = get_weather_status(float(rain))
    return render_temp_card(round(prediction, 2)), render_weather_card(weather_label, weather_icon, rain)


with gr.Blocks(title="Cambodia Weather Forecast") as demo:
    gr.Markdown("# Cambodia Weather Prediction ☀️")
    gr.Markdown("Predict next expected max temperature using your trained model.")

    with gr.Row():
        input_date = gr.Textbox(label="Date", value=str(date.today()), placeholder="YYYY-MM-DD")
        province = gr.Dropdown(choices=provinces, value=provinces[0] if provinces else None, label="Province")

    with gr.Row():
        temp_min = gr.Number(label="Min Temperature (C)", value=24.0)
        rain = gr.Number(label="Rainfall (mm)", value=0.0)
        wind_speed = gr.Number(label="Wind Speed (km/h)", value=10.0)

    with gr.Row():
        lat = gr.Number(label="Latitude", value=11.55)
        lon = gr.Number(label="Longitude", value=104.91)

    with gr.Row():
        predict_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        output = gr.HTML(value=render_temp_card(0.0), scale=1)
        weather_icon = gr.HTML(value=render_weather_card(*get_weather_status(0.0), 0.0), scale=1)

    predict_btn.click(
        fn=predict_temp_max,
        inputs=[input_date, province, temp_min, rain, wind_speed, lat, lon],
        outputs=[output, weather_icon],
    )

    gr.Markdown(
        f"Training metrics: MAE={metrics.get('mae', 'N/A')}, RMSE={metrics.get('rmse', 'N/A')}, R2={metrics.get('r2', 'N/A')}"
    )


if __name__ == "__main__":
    demo.launch()
