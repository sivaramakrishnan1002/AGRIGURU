import os
import sys

# Avoid joblib's physical-core probe failing in restricted Windows shells.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# Windows consoles commonly default to cp1252, which cannot print the emoji
# and rupee characters used in the command-line report.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
from datetime import datetime, timedelta
import joblib
from fuzzywuzzy import fuzz
import warnings
from typing import Dict, List, Optional, Union

# Suppress warnings
warnings.filterwarnings('ignore')

# ======= CONFIGURATION ======= #
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILES = {
        'crop_weather': os.path.join(BASE_DIR, "cleaned_crop_data.csv"),
        'crop_soil': os.path.join(BASE_DIR, "crop_soil_requirements_with_tamil.csv"),
        'district_soil': os.path.join(BASE_DIR, "tn_district_soil_data.csv"),
        'model': os.path.join(BASE_DIR, "crop_model.pkl"),
        'encoders': os.path.join(BASE_DIR, "encoders.pkl"),
        'price_data': os.path.join(BASE_DIR, "tamilnadu_crop_prices_quintal.csv"),
        'growth_data': os.path.join(BASE_DIR, "crop_avg_growth_days_only.csv")
    }
    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    FALLBACK_WEATHER = {
        'temperature (°C)': 28.0,
        'humidity (%)': 65.0,
        'rainfall (mm)': 50.0
    }
    MODEL_PARAMS = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'random_state': 42
    }

# ======= DATA LOADER ======= #
class DataLoader:
    def __init__(self):
        self._load_data()
        self._clean_data()
        
    def _load_data(self):
        """Load all required datasets"""
        self.crop_weather = pd.read_csv(Config.DATA_FILES['crop_weather'])
        self.crop_soil = pd.read_csv(Config.DATA_FILES['crop_soil'])
        self.district_soil = pd.read_csv(Config.DATA_FILES['district_soil'])
        self.price_data = pd.read_csv(Config.DATA_FILES['price_data'])
        self.growth_data = pd.read_csv(Config.DATA_FILES['growth_data'])
        
    def _clean_data(self):
        """Clean and standardize data"""
        # Standardize district names
        self.district_soil['District'] = self.district_soil['District'].str.strip().str.title()
        
        # Handle missing values in crop weather data
        self.crop_weather['Humidity'] = self.crop_weather['Humidity'].fillna(
            self.crop_weather['Humidity'].median()
        )
        
        # Convert price data from wide to long format
        self.price_data = self.price_data.melt(
            id_vars=['Crop'],
            var_name='District',
            value_name='Price_Per_Quintal'
        )
        
        # Clean crop names
        self.price_data['Crop'] = self.price_data['Crop'].str.strip()
        self.growth_data['Crop'] = self.growth_data['Crop'].str.strip()
        
        # Create soil type mapping for consistency
        self._create_soil_mapping()
    
    def _create_soil_mapping(self):
        """Create mapping between similar soil types"""
        self.soil_mapping = {
            'Red Sandy': 'Red Sandy Loam',
            'Red Loam': 'Red Loamy Soil',
            'Laterite': 'Lateritic Soil',
            'Alluvial': 'Alluvial Soil',
            'Black Soil': 'Black Cotton Soil'
        }
        
        # Apply mapping to both datasets
        self.crop_soil['Soil_Type'] = self.crop_soil['Soil_Type'].replace(self.soil_mapping)
        self.district_soil['Soil_Type'] = self.district_soil['Soil_Type'].replace(self.soil_mapping)

# ======= WEATHER SERVICE ======= #
class WeatherService:
    def __init__(self):
        self.cache = {}
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    def get_weather_forecast(self, lat: float, lon: float, start_date: str) -> Dict:
        """Get combined weather forecast with fallback"""
        start_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_obj = start_obj + timedelta(days=90)
        
        cache_key = f"{lat}_{lon}_{start_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Try Open-Meteo first
            forecast = self._get_open_meteo_forecast(lat, lon, start_obj, end_obj)
            if forecast is None:
                # Fallback to NASA with historical average
                forecast = self._get_nasa_with_historical(lat, lon, start_obj, end_obj)
            
            if forecast is None:
                # Final fallback to default values
                forecast = Config.FALLBACK_WEATHER.copy()
                forecast['source'] = 'fallback'
            else:
                forecast['source'] = 'api'
                
            self.cache[cache_key] = forecast
            return forecast
            
        except Exception as e:
            print(f"⚠️ Weather forecast error: {str(e)}")
            return Config.FALLBACK_WEATHER.copy()
    
    def _get_open_meteo_forecast(self, lat: float, lon: float, start_obj: datetime, end_obj: datetime) -> Optional[Dict]:
        """Get forecast from Open-Meteo API"""
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=auto"
        )
        try:
            # A recommendation should remain responsive even when a free host
            # cannot reach the weather provider. The local fallback is used
            # after this short, bounded attempt.
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()['daily']
            
            df = pd.DataFrame({
                "date": pd.to_datetime(data["time"]),
                "temperature": [(tmin + tmax) / 2 for tmin, tmax in zip(data["temperature_2m_min"], data["temperature_2m_max"])],
                "humidity": data["relative_humidity_2m_mean"],
                "rainfall": data["precipitation_sum"]
            })
            
            # Filter for the requested date range
            df = df[(df["date"] >= start_obj) & (df["date"] <= end_obj)]
            
            if df.empty:
                return None
                
            return {
                "temperature (°C)": round(df["temperature"].mean(), 2),
                "humidity (%)": round(df["humidity"].mean(), 2),
                "rainfall (mm)": round(df["rainfall"].sum(), 2)
            }
        except Exception:
            return None
    
    def _get_nasa_with_historical(self, lat: float, lon: float, start_obj: datetime, end_obj: datetime) -> Optional[Dict]:
        """Get NASA data with historical average"""
        try:
            hist_avg = self._generate_historical_avg(lat, lon, start_obj, end_obj)
            if hist_avg is None:
                return None
                
            return {
                "temperature (°C)": round(hist_avg["temperature"].mean(), 2),
                "humidity (%)": round(hist_avg["humidity"].mean(), 2),
                "rainfall (mm)": round(hist_avg["rainfall"].sum(), 2)
            }
        except Exception:
            return None
    
    def _generate_historical_avg(self, lat: float, lon: float, start_obj: datetime, end_obj: datetime) -> Optional[pd.DataFrame]:
        """Generate 5-year historical average"""
        all_data = []
        for year_offset in range(1, 6):
            start = (start_obj - timedelta(days=365 * year_offset)).strftime("%Y%m%d")
            end = (end_obj - timedelta(days=365 * year_offset)).strftime("%Y%m%d")
            df = self._get_nasa_power_data(lat, lon, start, end)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            return None
        
        combined = pd.concat(all_data)
        monthly_avg = combined.groupby(combined.date.dt.month).mean(numeric_only=True)
        return monthly_avg.reset_index()
    
    def _get_nasa_power_data(self, lat: float, lon: float, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """NASA POWER API implementation"""
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,RH2M,PRECTOTCORR",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()['properties']['parameter']
            df = pd.DataFrame({
                "date": list(data["T2M"].keys()),
                "temperature": list(data["T2M"].values()),
                "humidity": list(data["RH2M"].values()),
                "rainfall": list(data["PRECTOTCORR"].values())
            })
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return None

# ======= SOIL SERVICE ======= #
class SoilService:
    def __init__(self, data_loader: DataLoader):
        self.district_soil = data_loader.district_soil
    
    def get_soil_data(self, district_name: str) -> Dict:
        """Get soil data for specified district with fuzzy matching"""
        district_name = district_name.strip().title()
        
        # Exact match first
        soil_data = self.district_soil[
            self.district_soil['District'] == district_name
        ]
        
        # Fuzzy match if no exact match found
        if soil_data.empty:
            best_match = None
            best_score = 0
            
            for district in self.district_soil['District'].unique():
                score = fuzz.ratio(district_name, district)
                if score > best_score and score > 70:  # Only consider good matches
                    best_score = score
                    best_match = district
            
            if best_match:
                soil_data = self.district_soil[
                    self.district_soil['District'] == best_match
                ]
        
        if soil_data.empty:
            raise ValueError(f"Soil data not found for district: {district_name}")
        
        return {
            'type': soil_data.iloc[0]['Soil_Type'],
            'pH': soil_data.iloc[0]['pH'],
            'N': soil_data.iloc[0]['N'],
            'P': soil_data.iloc[0]['P'],
            'K': soil_data.iloc[0]['K']
        }

# ======= ML MODEL ======= #
class CropModel:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model = None
        self.encoders = None
        self._prepare_model()
    
    def _prepare_model(self):
        """Prepare or load the ML model"""
        if os.path.exists(Config.DATA_FILES['model']):
            self._load_model()
        else:
            self._train_model()
            self._save_model()
    
    def _train_model(self):
        """Train the LightGBM model"""
        merged = pd.merge(
            self.data_loader.crop_weather, 
            self.data_loader.crop_soil, 
            on='Crop', 
            how='inner'
        )
        
        # Create and fit encoders
        self.encoders = {
            'Soil_Type': LabelEncoder().fit(
                pd.concat([
                    self.data_loader.district_soil['Soil_Type'], 
                    self.data_loader.crop_soil['Soil_Type']
                ])
            ),
            'N': LabelEncoder().fit(
                pd.concat([
                    self.data_loader.district_soil['N'], 
                    self.data_loader.crop_soil['N']
                ])
            ),
            'P': LabelEncoder().fit(
                pd.concat([
                    self.data_loader.district_soil['P'], 
                    self.data_loader.crop_soil['P']
                ])
            ),
            'K': LabelEncoder().fit(
                pd.concat([
                    self.data_loader.district_soil['K'], 
                    self.data_loader.crop_soil['K']
                ])
            )
        }
        
        # Apply encoding
        for col in ['Soil_Type', 'N', 'P', 'K']:
            merged[col] = self.encoders[col].transform(merged[col])
        
        # Train model
        features = ['Temperature', 'Humidity', 'Rainfall', 'Soil_Type', 
                   'pH_min', 'pH_max', 'N', 'P', 'K']
        X = merged[features]
        y = merged['Crop']
        
        self.model = lgb.LGBMClassifier(**Config.MODEL_PARAMS)
        self.model.fit(X, y)
    
    def _save_model(self):
        """Save model and encoders to disk"""
        joblib.dump(self.model, Config.DATA_FILES['model'])
        joblib.dump(self.encoders, Config.DATA_FILES['encoders'])
    
    def _load_model(self):
        """Load model and encoders from disk"""
        self.model = joblib.load(Config.DATA_FILES['model'])
        self.encoders = joblib.load(Config.DATA_FILES['encoders'])
    
    def predict(self, weather: Dict, soil: Dict) -> List[Dict]:
        """Predict suitable crops with probabilities"""
        predictions = []
        
        for _, crop in self.data_loader.crop_soil.iterrows():
            try:
                input_data = {
                    'Temperature': weather['temperature (°C)'],
                    'Humidity': weather['humidity (%)'],
                    'Rainfall': weather['rainfall (mm)'],
                    'Soil_Type': self.encoders['Soil_Type'].transform([soil['type']])[0],
                    'pH_min': crop['pH_min'],
                    'pH_max': crop['pH_max'],
                    'N': self.encoders['N'].transform([soil['N']])[0],
                    'P': self.encoders['P'].transform([soil['P']])[0],
                    'K': self.encoders['K'].transform([soil['K']])[0]
                }
                
                input_df = pd.DataFrame([input_data])
                proba = self.model.predict_proba(input_df)[0]
                crop_idx = np.where(self.model.classes_ == crop['Crop'])[0]
                
                if len(crop_idx) > 0:
                    predictions.append({
                        'crop': crop['Crop'],
                        'probability': float(proba[crop_idx[0]]),
                        'soil_match': self._check_soil_match(crop, soil),
                        'ph_match': self._check_ph_match(crop, soil)
                    })
            except Exception as e:
                print(f"Prediction error for {crop['Crop']}: {str(e)}")
                continue
        
        # Sort by probability and additional factors
        predictions.sort(
            key=lambda x: (
                x['probability'], 
                x['soil_match'],
                x['ph_match']
            ), 
            reverse=True
        )
        
        return predictions[:5]  # Return top 5 recommendations
    
    def _check_soil_match(self, crop: pd.Series, soil: Dict) -> float:
        """Check soil type compatibility (0-1 score)"""
        if crop['Soil_Type'] == soil['type']:
            return 1.0
        return 0.5  # Partial match if soil types are different but both in mapping
    
    def _check_ph_match(self, crop: pd.Series, soil: Dict) -> float:
        """Check pH compatibility (0-1 score)"""
        soil_ph = soil['pH']
        min_ph = crop['pH_min']
        max_ph = crop['pH_max']
        
        if min_ph <= soil_ph <= max_ph:
            return 1.0
        elif (min_ph - 0.5) <= soil_ph <= (max_ph + 0.5):
            return 0.7
        else:
            return 0.3

# ======= MARKET ANALYSIS ======= #
class MarketAnalysis:
    def __init__(self, data_loader: DataLoader):
        self.price_data = data_loader.price_data
        self.growth_data = data_loader.growth_data
    
    def analyze_profitability(self, crops: List[Dict], district: str) -> List[Dict]:
        """Analyze profitability of recommended crops"""
        district = district.strip().title()
        analyzed_crops = []
        
        for crop in crops:
            crop_name = crop['crop'].strip()
            try:
                # Get current market price
                price_row = self.price_data[
                    (self.price_data['District'].str.lower() == district.lower()) & 
                    (self.price_data['Crop'].str.lower() == crop_name.lower())
                ]
                
                if price_row.empty:
                    continue
                
                current_price = price_row.iloc[0]['Price_Per_Quintal']
                
                # Get growth period
                growth_row = self.growth_data[
                    self.growth_data['Crop'].str.lower() == crop_name.lower()
                ]
                
                if growth_row.empty:
                    continue
                
                growth_days = growth_row.iloc[0]['Avg Growth Period (Days)']
                
                # Calculate profit potential
                future_price = self._predict_future_price(crop_name, current_price, growth_days)
                profit = future_price - current_price
                roi = (profit / current_price) * 100 if current_price > 0 else 0
                
                analyzed_crops.append({
                    **crop,
                    'current_price': current_price,
                    'future_price': future_price,
                    'growth_days': growth_days,
                    'profit': profit,
                    'roi': roi,
                    'combined_score': self._calculate_combined_score(crop['probability'], roi)
                })
                
            except Exception as e:
                print(f"Market analysis error for {crop_name}: {str(e)}")
                continue
        
        # Sort by combined score and select top 3
        analyzed_crops.sort(key=lambda x: x['combined_score'], reverse=True)
        return analyzed_crops[:3]

    def _predict_future_price(self, crop_name: str, current_price: float, growth_days: int) -> float:
        """Predict price considering seasonal trends"""
        crop_type = self._get_crop_type(crop_name)
        factors = {'vegetables': 1.15, 'fruits': 1.10, 'staples': 1.08}
        factor = factors.get(crop_type, 1.05)
        
        # Adjust for growth period (longer growth = more price variation)
        months = growth_days / 30
        return current_price * (factor ** (months/3))

    def _get_crop_type(self, crop_name: str) -> str:
        """Categorize crop for seasonal adjustments"""
        crop_name = crop_name.lower()
        veg_keywords = ['tomato', 'onion', 'brinjal', 'chilli']
        fruit_keywords = ['mango', 'banana', 'guava']
        
        if any(x in crop_name for x in veg_keywords):
            return 'vegetables'
        elif any(x in crop_name for x in fruit_keywords):
            return 'fruits'
        elif any(x in crop_name for x in ['paddy', 'rice', 'wheat']):
            return 'staples'
        return 'other'

    def _calculate_combined_score(self, suitability: float, roi: float) -> float:
        """Combine suitability and ROI into single score"""
        return (0.6 * suitability) + (0.4 * (min(roi, 100)/100))  # Normalize ROI to 0-1

# ======= INTEGRATED RECOMMENDATION SYSTEM ======= #
class IntegratedRecommendationSystem:
    def __init__(self):
        self.data_loader = DataLoader()
        self.weather_service = WeatherService()
        self.soil_service = SoilService(self.data_loader)
        self.crop_model = CropModel(self.data_loader)
        self.market_analysis = MarketAnalysis(self.data_loader)
    
    def get_recommendations(self, lat: float, lon: float, district_name: str, start_date: str) -> Dict:
        """Get complete recommendations with market analysis"""
        try:
            # Get weather data
            weather = self.weather_service.get_weather_forecast(lat, lon, start_date)
            
            # Get soil data
            soil = self.soil_service.get_soil_data(district_name)
            
            # Get agronomic recommendations
            agronomic_recs = self.crop_model.predict(weather, soil)
            
            # Analyze market potential
            final_recommendations = self.market_analysis.analyze_profitability(agronomic_recs, district_name)
            
            return {
                'status': 'success',
                'coordinates': {'latitude': lat, 'longitude': lon},
                'district': district_name.title(),
                'weather': {
                    **weather,
                    'season': self._get_season(start_date)
                },
                'soil': soil,
                'recommendations': final_recommendations
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'available_districts': sorted(self.data_loader.district_soil['District'].unique())
            }
    
    def _get_season(self, date_str: str) -> str:
        """Get season based on date"""
        month = datetime.strptime(date_str, "%Y-%m-%d").month
        if 3 <= month <= 5:
            return "Summer"
        elif 6 <= month <= 9:
            return "Monsoon"
        elif 10 <= month <= 11:
            return "Autumn"
        else:
            return "Winter"

# ======= USER INTERFACE ======= #
class UserInterface:
    @staticmethod
    def get_user_input() -> Dict:
        """Get and validate user input"""
        print("\n=== Smart Crop Recommendation System ===")
        print("Enter location and planting details:")
        
        try:
            lat = float(input("Latitude (e.g. 13.0827): ").strip())
            lon = float(input("Longitude (e.g. 80.2707): ").strip())
            district = input("District Name (e.g. Chennai): ").strip()
            planting_date = input("Planting date (YYYY-MM-DD): ").strip()
            
            # Validate date format
            datetime.strptime(planting_date, "%Y-%m-%d")
            
            return {
                'latitude': lat,
                'longitude': lon,
                'district': district,
                'planting_date': planting_date
            }
        except ValueError as e:
            print(f"\n❌ Invalid input: {str(e)}")
            return None
    
    @staticmethod
    def display_results(results: Dict):
        """Display recommendation results."""
        print(format_recommendation_report(results))


def format_recommendation_report(results: Dict) -> str:
    """Return the report consumed by the web interface and CLI."""
    if results['status'] == 'error':
        available = ", ".join(results['available_districts'])
        return f"\nError: {results['message']}\n\nAvailable districts:\n{available}"

    lines = [
        "", "=== Final Recommendation Report ===",
        f"Location: {results['coordinates']['latitude']}, {results['coordinates']['longitude']}",
        f"District: {results['district']}",
        f"Season: {results['weather']['season']}",
        "", "Weather Conditions:",
        f"- Temperature: {next(value for key, value in results['weather'].items() if key.startswith('temperature'))}°C",
        f"- Humidity: {results['weather']['humidity (%)']}%",
        f"- Rainfall: {results['weather']['rainfall (mm)']}mm",
        f"- Source: {results['weather'].get('source', 'unknown')}",
        "", "Soil Conditions:",
        f"- Type: {results['soil']['type']}",
        f"- pH: {results['soil']['pH']}",
        f"- Nutrients (N-P-K): {results['soil']['N']}-{results['soil']['P']}-{results['soil']['K']}",
        "", "Top 3 Most Profitable Crops:", "=" * 120,
        f"{'Rank':<5}{'Crop':<25}{'Suitability':<12}{'Current Price':<15}{'Future Price':<15}{'Profit':<12}{'ROI%':<10}{'Growth Days':<12}",
        "-" * 120,
    ]
    for rank, crop in enumerate(results['recommendations'], 1):
        lines.append(
            f"{rank:<5}{crop['crop']:<25}{crop['probability']:<12.2f}₹{crop['current_price']:<14.2f}₹{crop['future_price']:<14.2f}₹{crop['profit']:<11.2f}{crop['roi']:<10.1f}%{crop['growth_days']:>12}d"
        )
    lines.extend(["=" * 120, "Note: Suitability (0-1) measures agronomic compatibility", "      Recommendations combine suitability (60%) and profit potential (40%)"])
    return "\n".join(lines)


class _UnusedLegacyUserInterface:
    @staticmethod
    def get_user_input() -> Dict:
        """Get and validate user input"""
        return None
        if results['status'] == 'error':
            print(f"\n❌ Error: {results['message']}")
            print("\nAvailable districts:")
            print(", ".join(results['available_districts']))
            return
        
        print("\n=== Final Recommendation Report ===")
        print(f"📍 Location: {results['coordinates']['latitude']}, {results['coordinates']['longitude']}")
        print(f"🗺️ District: {results['district']}")
        print(f"📅 Season: {results['weather']['season']}")
        
        print("\n🌦️ Weather Conditions:")
        print(f"- Temperature: {results['weather']['temperature (°C)']}°C")
        print(f"- Humidity: {results['weather']['humidity (%)']}%")
        print(f"- Rainfall: {results['weather']['rainfall (mm)']}mm")
        print(f"- Source: {results['weather'].get('source', 'unknown')}")
        
        print("\n🌱 Soil Conditions:")
        print(f"- Type: {results['soil']['type']}")
        print(f"- pH: {results['soil']['pH']}")
        print(f"- Nutrients (N-P-K): {results['soil']['N']}-{results['soil']['P']}-{results['soil']['K']}")
        
        print("\n🏆 Top 3 Most Profitable Crops:")
        print("="*120)
        print(f"{'Rank':<5}{'Crop':<25}{'Suitability':<12}{'Current Price':<15}{'Future Price':<15}{'Profit':<12}{'ROI%':<10}{'Growth Days':<12}")
        print("-"*120)
        
        for rank, crop in enumerate(results['recommendations'], 1):
            print(f"{rank:<5}{crop['crop']:<25}{crop['probability']:<12.2f}₹{crop['current_price']:<14.2f}₹{crop['future_price']:<14.2f}₹{crop['profit']:<11.2f}{crop['roi']:<10.1f}%{crop['growth_days']:>12}d")
        
        print("="*120)
        print("Note: Suitability (0-1) measures agronomic compatibility")
        print("      Recommendations combine suitability (60%) and profit potential (40%)")

def run_recommendation(lat, lon, district, date):
    system = IntegratedRecommendationSystem()
    results = system.get_recommendations(
        lat=lat,
        lon=lon,
        district_name=district,
        start_date=date
    )
    ui = UserInterface()
    ui.display_results(results)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        lat, lon, district, date = sys.argv[1:5]
        print(f"Input Details:\nLatitude: {lat}\nLongitude: {lon}\nDistrict: {district}\nDate: {date}\n")
        run_recommendation(lat, lon, district, date)
