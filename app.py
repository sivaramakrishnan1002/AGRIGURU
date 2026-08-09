from pathlib import Path
import secrets

from flask import Flask, render_template, request, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import subprocess
import sys

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', districts=DISTRICTS, taluks=TALUKS)

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
        result = subprocess.check_output([
            sys.executable, str(BASE_DIR / 'market_integrated.py'),
            str(latitude), str(longitude), str(district), str(current_date)
        ], text=True, encoding='utf-8', stderr=subprocess.STDOUT, timeout=90)
        print("Recommendation generated successfully.")
    except Exception as e:
        print("Error running market_integrated.py:", e)
        return jsonify({'error': str(e)}), 500

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
