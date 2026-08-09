# AgriGuru Crop Recommendation

A Flask web app that recommends suitable crops for Tamil Nadu locations using
district soil data, weather information, crop suitability, and market data.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

## Notes

- Select a district and taluk; village is optional.
- The user interface supports English and Tamil.
- Local account/OTP functionality is a development demo only.
