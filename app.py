from pathlib import Path
from difflib import SequenceMatcher
import secrets

from flask import Flask, render_template, request, jsonify
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime

from market_integrated import IntegratedRecommendationSystem

BASE_DIR = Path(__file__).resolve().parent

# The project's HTML files are kept beside this module rather than in Flask's
# default ``templates`` directory.
app = Flask(__name__, template_folder=str(BASE_DIR), static_folder=str(BASE_DIR / 'static'))

# Development-only account store. Replace this with a database and a real SMS
# provider before deploying the application publicly.
users = {}
otp_codes = {}
DISTRICTS = [
    'Chennai', 'Kanchipuram', 'Tiruvallur', 'Vellore', 'Ranipet',
    'Tiruvannamalai', 'Villupuram', 'Cuddalore', 'Pondicherry', 'Thanjavur',
    'Tiruvarur', 'Nagapattinam', 'Trichy', 'Karur', 'Ariyalur', 'Perambalur',
    'Namakkal', 'Salem', 'Dharmapuri', 'Krishnagiri', 'Erode', 'Coimbatore',
    'Tiruppur', 'Dindigul', 'Madurai', 'Sivagangai', 'Ramanathapuram',
    'Virudhunagar', 'Thoothukudi', 'Tirunelveli', 'Nilgiris', 'Kanyakumari',
    'Mayiladuthurai',
]
TALUKS = {
    'Chennai': ['Alandur', 'Ambattur', 'Guindy', 'Mylapore', 'Velachery'],
    'Kanchipuram': ['Kanchipuram', 'Sriperumbudur', 'Uthiramerur', 'Walajabad'],
    'Tiruvallur': ['Avadi', 'Gummidipoondi', 'Ponneri', 'Poonamallee', 'Tiruvallur', 'Uthukkottai'],
    'Vellore': ['Vellore', 'Katpadi', 'Gudiyattam', 'Anaicut'], 'Ranipet': ['Ranipet', 'Arakkonam', 'Walajah'],
    'Tiruvannamalai': ['Tiruvannamalai', 'Arani', 'Cheyyar', 'Polur'], 'Villupuram': ['Villupuram', 'Tindivanam', 'Gingee', 'Kallakurichi'],
    'Cuddalore': ['Cuddalore', 'Chidambaram', 'Panruti', 'Virudhachalam'], 'Pondicherry': ['Ozhukarai', 'Villianur', 'Bahour'],
    'Thanjavur': ['Thanjavur', 'Kumbakonam', 'Papanasam', 'Pattukkottai'], 'Tiruvarur': ['Tiruvarur', 'Mannargudi', 'Nannilam'],
    'Nagapattinam': ['Nagapattinam', 'Vedaranyam', 'Kilvelur'], 'Trichy': ['Tiruchirappalli', 'Manapparai', 'Lalgudi'],
    'Karur': ['Karur', 'Aravakurichi', 'Krishnarayapuram'], 'Ariyalur': ['Ariyalur', 'Udayarpalayam', 'Sendurai'],
    'Perambalur': ['Perambalur', 'Kunnam', 'Veppanthattai'], 'Namakkal': ['Namakkal', 'Rasipuram', 'Tiruchengode'],
    'Salem': ['Salem', 'Attur', 'Mettur', 'Omalur'], 'Dharmapuri': ['Dharmapuri', 'Harur', 'Palacode'],
    'Krishnagiri': ['Krishnagiri', 'Hosur', 'Pochampalli'], 'Erode': ['Erode', 'Bhavani', 'Gobichettipalayam'],
    'Coimbatore': ['Coimbatore North', 'Coimbatore South', 'Pollachi', 'Mettupalayam'], 'Tiruppur': ['Tiruppur North', 'Tiruppur South', 'Dharapuram'],
    'Dindigul': ['Dindigul', 'Kodaikanal', 'Palani'], 'Madurai': ['Madurai North', 'Madurai South', 'Melur', 'Usilampatti'],
    'Sivagangai': ['Sivagangai', 'Karaikudi', 'Manamadurai'], 'Ramanathapuram': ['Ramanathapuram', 'Paramakudi', 'Rameswaram'],
    'Virudhunagar': ['Virudhunagar', 'Aruppukkottai', 'Rajapalayam'], 'Thoothukudi': ['Thoothukudi', 'Kovilpatti', 'Tiruchendur'],
    'Tirunelveli': ['Tirunelveli', 'Ambasamudram', 'Palayamkottai'], 'Nilgiris': ['Udhagamandalam', 'Coonoor', 'Gudalur'],
    'Kanyakumari': ['Nagercoil', 'Kalkulam', 'Vilavancode'], 'Mayiladuthurai': ['Mayiladuthurai', 'Sirkazhi', 'Kuthalam'],
}

# Keep the data and ML model in the Gunicorn worker instead of starting a new
# Python process for every request. This also lets the weather cache be reused.
recommendation_system = None
geocoder = Nominatim(user_agent='agriguru-crop-recommendation/1.0', timeout=5)
location_cache = {}
coordinate_cache = {}


def get_recommendation_system():
    global recommendation_system
    if recommendation_system is None:
        recommendation_system = IntegratedRecommendationSystem()
    return recommendation_system


def format_web_recommendation(results):
    """Create the stable text format consumed by chatbot.html."""
    weather = results['weather']
    temperature_key = next(key for key in weather if key.startswith('temperature'))
    lines = [
        '=== Final Recommendation Report ===',
        f"Location: {results['coordinates']['latitude']}, {results['coordinates']['longitude']}",
        f"District: {results['district']}",
        f"Season: {weather['season']}",
        '', 'Weather Conditions:',
        f"- Temperature: {weather[temperature_key]}°C",
        f"- Humidity: {weather['humidity (%)']}%",
        f"- Rainfall: {weather['rainfall (mm)']}mm",
        '', 'Soil Conditions:',
        f"- Type: {results['soil']['type']}",
        f"- pH: {results['soil']['pH']}",
        f"- Nutrients (N-P-K): {results['soil']['N']}-{results['soil']['P']}-{results['soil']['K']}",
        '', 'Top 3 Most Profitable Crops:',
    ]
    rupee = chr(0x20B9)
    for rank, crop in enumerate(results['recommendations'], 1):
        lines.append(
            f"{rank} {crop['crop']} {crop['probability']:.2f}"
            f" {rupee}{crop['current_price']:.2f} {rupee}{crop['future_price']:.2f}"
            f" {rupee}{crop['profit']:.2f}"
        )
    return '\n'.join(lines)


def normalize_place_name(value):
    """Normalize location names before comparing government-area labels."""
    return ''.join(char for char in (value or '').lower() if char.isalnum())


def best_match(value, choices):
    """Return a confident match from the app's supported location list."""
    candidate = normalize_place_name(value)
    if not candidate:
        return None
    for choice in choices:
        normalized = normalize_place_name(choice)
        if candidate == normalized or candidate.startswith(normalized) or normalized.startswith(candidate):
            return choice
    scored = [(SequenceMatcher(None, candidate, normalize_place_name(choice)).ratio(), choice) for choice in choices]
    score, choice = max(scored, default=(0, None))
    return choice if score >= 0.82 else None


def reverse_geocode(latitude, longitude):
    """Resolve a coordinate to supported district/taluk fields when possible."""
    cache_key = (round(latitude, 4), round(longitude, 4))
    if cache_key in location_cache:
        return location_cache[cache_key]

    location = geocoder.reverse((latitude, longitude), exactly_one=True, language='en')
    address = getattr(location, 'raw', {}).get('address', {}) if location else {}
    district = best_match(
        address.get('state_district') or address.get('district') or address.get('county'),
        DISTRICTS,
    )
    area_values = [
        address.get(key, '')
        for key in ('taluk', 'city_district', 'county', 'municipality', 'city', 'town', 'village', 'suburb')
    ]
    taluk = next(
        (match for value in area_values if (match := best_match(value, TALUKS.get(district, [])))),
        None,
    )
    village = next((address.get(key) for key in ('village', 'hamlet', 'suburb', 'neighbourhood') if address.get(key)), '')
    result = {'district': district or '', 'taluk': taluk or '', 'village': village or ''}
    location_cache[cache_key] = result
    return result


def geocode_place(district, taluk='', village=''):
    """Resolve user-entered Tamil Nadu place fields to coordinates."""
    cache_key = (district, taluk, village)
    if cache_key in coordinate_cache:
        return coordinate_cache[cache_key]

    parts = [part.strip() for part in (village, taluk, district, 'Tamil Nadu', 'India') if part and part.strip()]
    location = geocoder.geocode(', '.join(parts), exactly_one=True, language='en')
    if location is None:
        raise ValueError('Location not found. Check the district, taluk, and village names, or enter coordinates manually.')
    result = {'latitude': round(location.latitude, 6), 'longitude': round(location.longitude, 6)}
    coordinate_cache[cache_key] = result
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', districts=DISTRICTS, taluks=TALUKS)


@app.route('/resolve_location', methods=['POST'])
def resolve_location():
    data = request.get_json(silent=True) or {}
    try:
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Enter valid latitude and longitude values.'}), 400
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return jsonify({'error': 'Latitude or longitude is outside its valid range.'}), 400
    try:
        return jsonify(reverse_geocode(latitude, longitude))
    except (GeocoderTimedOut, GeocoderServiceError, OSError):
        # Geolocation itself remains useful even if the optional lookup service is unavailable.
        return jsonify({'district': '', 'taluk': '', 'village': '', 'warning': 'Location found, but place details are temporarily unavailable.'})


@app.route('/resolve_coordinates', methods=['POST'])
def resolve_coordinates():
    data = request.get_json(silent=True) or {}
    district = (data.get('district') or '').strip()
    taluk = (data.get('taluk') or '').strip()
    village = (data.get('village') or '').strip()
    if district not in DISTRICTS:
        return jsonify({'error': 'Select a district from the available list first.'}), 400
    if taluk and taluk not in TALUKS[district]:
        return jsonify({'error': 'Select a taluk from the available list.'}), 400
    if not village and not taluk:
        return jsonify({'error': 'Enter a village or select a taluk to find coordinates.'}), 400
    try:
        return jsonify(geocode_place(district, taluk, village))
    except ValueError as error:
        return jsonify({'error': str(error)}), 404
    except (GeocoderTimedOut, GeocoderServiceError, OSError):
        return jsonify({'error': 'Place lookup is temporarily unavailable. Enter coordinates manually and try again later.'}), 503

@app.route('/market')
def market():
     return render_template('market.html')

@app.route('/farmer')
def farmer():
     return render_template('farmer.html')

@app.route('/customer')
def customer():
     return render_template('customer.html')


@app.route('/send_otp', methods=['POST'])
def send_otp():
    phone = (request.get_json(silent=True) or {}).get('phone', '').strip()
    if not phone.isdigit() or len(phone) != 10:
        return jsonify({'success': False, 'error': 'Enter a valid 10-digit phone number.'}), 400

    code = f'{secrets.randbelow(1_000_000):06d}'
    otp_codes[phone] = code
    # No SMS gateway is configured yet, so expose the one-time code to the
    # local UI. Do not use this endpoint as-is in production.
    return jsonify({'success': True, 'development_otp': code})


@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.get_json(silent=True) or {}
    verified = otp_codes.get(data.get('phone', '')) == str(data.get('otp', '')).strip()
    return jsonify({'verified': verified})


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    phone = data.get('phone', '').strip()
    password = data.get('password', '')
    if not phone.isdigit() or len(phone) != 10 or not password:
        return jsonify({'success': False, 'error': 'Valid phone number and password are required.'}), 400
    if phone in users:
        return jsonify({'success': False, 'error': 'An account already exists for this phone number.'}), 409

    users[phone] = {
        'name': f"{data.get('first', '').strip()} {data.get('last', '').strip()}".strip(),
        'password': generate_password_hash(password),
    }
    otp_codes.pop(phone, None)
    return jsonify({'success': True})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    user = users.get(data.get('phone', '').strip())
    if user and check_password_hash(user['password'], data.get('password', '')):
        return jsonify({'success': True, 'name': user['name']})
    return jsonify({'success': False, 'error': 'Incorrect phone number or password.'}), 401


@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    print("Received request for recommendation")
    data = request.get_json(silent=True) or {}
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    district = (data.get('district') or '').strip()

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return jsonify({'error': 'Enter valid latitude and longitude values.'}), 400
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return jsonify({'error': 'Latitude or longitude is outside its valid range.'}), 400
    if district not in DISTRICTS:
        return jsonify({'error': 'Select a district from the available list.'}), 400

    current_date = datetime.now().strftime('%Y-%m-%d')

    try:
        results = get_recommendation_system().get_recommendations(
            latitude, longitude, district, current_date
        )
        if results['status'] != 'success':
            raise RuntimeError(results['message'])
        result = format_web_recommendation(results)
        print("Recommendation generated successfully.")
    except Exception as e:
        print("Error running market_integrated.py:", e)
        return jsonify({'error': str(e)}), 500

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
