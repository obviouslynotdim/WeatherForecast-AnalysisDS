import pandas as pd

from src.data.preprocess import add_time_features, build_features


def test_build_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "temp_max": [33.1, 34.2],
            "temp_min": [24.0, 25.0],
            "rain": [0.0, 1.0],
            "wind_speed": [10.0, 9.0],
            "province": ["Phnom Penh", "Siem Reap"],
            "lat": [11.55, 13.36],
            "lon": [104.91, 103.86],
        }
    )

    with_time = add_time_features(df)
    bundle = build_features(with_time)

    assert "month" in bundle.features.columns
    assert "month_sin" in bundle.features.columns
    assert "province_Phnom Penh" in bundle.features.columns
    assert len(bundle.features) == len(bundle.target)
