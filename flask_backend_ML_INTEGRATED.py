#!/usr/bin/env python3
"""
COMPLETE FLASK BACKEND FOR GRIP STRENGTH MONITORING SYSTEM
Updated for 6 FSR Sensors Support
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import sqlite3
import jwt
import hashlib
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)


# ==========================================
# MACHINE LEARNING MODEL LOADER
# ==========================================

class MLModelPredictor:
    """
    Wrapper class for the trained ML model
    Handles loading model, scaler, and normative data
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.norms_data = None
        self.load_model_and_data()

    def load_model_and_data(self):
        """Load ML model, scaler, and normative data"""
        try:
            print("Loading ML model...")
            self.model = keras.models.load_model('grip_recovery_model.h5')
            print("✓ ML model loaded successfully")

            print("Loading scaler...")
            with open('grip_recovery_model_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("✓ Scaler loaded successfully")

            print("Loading normative data...")
            with open('grip_strength_norms.json', 'r') as f:
                self.norms_data = json.load(f)
            print("✓ Normative data loaded successfully")

            print("\n" + "=" * 60)
            print("✓ ML MODEL READY FOR PREDICTIONS")
            print("=" * 60 + "\n")

        except FileNotFoundError as e:
            print(f"ERROR: Required file not found - {e}")
            print("\nPlease ensure these files are in the same folder:")
            print("  - grip_recovery_model.h5")
            print("  - grip_recovery_model_scaler.pkl")
            print("  - grip_strength_norms.json")
            raise
        except Exception as e:
            print(f"ERROR loading ML model: {e}")
            raise

    def get_age_group(self, age):
        """Map age to age group string"""
        if age < 20:
            return "15-19"
        elif age < 25:
            return "20-24"
        elif age < 30:
            return "25-29"
        elif age < 35:
            return "30-34"
        elif age < 40:
            return "35-39"
        elif age < 45:
            return "40-44"
        elif age < 50:
            return "45-49"
        elif age < 55:
            return "50-54"
        elif age < 60:
            return "55-59"
        elif age < 65:
            return "60-64"
        elif age < 70:
            return "65-69"
        else:
            return "70+"

    def predict_recovery(self, age, gender, current_grip, baseline_grip, days_in_therapy):
        """
        Predict recovery stage using trained ML model
        """
        try:
            # Convert gender to numeric if needed
            if isinstance(gender, str):
                gender_numeric = 1 if gender.lower() == 'male' else 0
            else:
                gender_numeric = gender

            gender_str = 'male' if gender_numeric == 1 else 'female'

            # Validate inputs
            if age is None or age <= 0:
                print(f"WARNING: Invalid age {age}, using default 30")
                age = 30
            if baseline_grip is None or baseline_grip <= 0:
                print(f"WARNING: Invalid baseline {baseline_grip}, using default 20.0")
                baseline_grip = 20.0

            print(
                f"ML Input: age={age}, gender={gender_str}, current={current_grip:.2f}, baseline={baseline_grip:.2f}, days={days_in_therapy}")

            # Prepare input features
            X = np.array([[age, gender_numeric, current_grip, baseline_grip, days_in_therapy]])

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Predict using ML model
            predictions = self.model.predict(X_scaled, verbose=0)
            stage = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][stage]) * 100

            # Calculate recovery percentage
            age_group = self.get_age_group(age)
            try:
                expected_grip = self.norms_data["norms"][gender_str][age_group]["mean"]
                print(f"Normative data: {gender_str} {age_group} expected grip = {expected_grip} kg")
            except KeyError as e:
                print(f"ERROR: Could not find normative data for {gender_str} {age_group}: {e}")
                expected_grip = 45.0

            if expected_grip > baseline_grip:
                recovery_percent = ((current_grip - baseline_grip) /
                                    (expected_grip - baseline_grip)) * 100
                recovery_percent = max(0, min(100, recovery_percent))
            else:
                recovery_percent = min((current_grip / expected_grip) * 100, 100)

            stage_names = [
                "Severe Paralysis",
                "Poor Recovery",
                "Fair Recovery",
                "Good Recovery",
                "Excellent Recovery"
            ]

            print(
                f"ML Output: Stage={stage} ({stage_names[stage]}), Confidence={confidence:.1f}%, Recovery={recovery_percent:.1f}%")

            return {
                'stage': stage,
                'stage_name': stage_names[stage],
                'confidence': confidence,
                'recovery_percent': recovery_percent
            }
        except Exception as e:
            print(f"ERROR in ML prediction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'stage': 0,
                'stage_name': 'Unknown',
                'confidence': 0.0,
                'recovery_percent': 0.0
            }


# Initialize ML model predictor
print("\n" + "=" * 60)
print("INITIALIZING MACHINE LEARNING MODEL")
print("=" * 60)
ml_predictor = MLModelPredictor()

SECRET_KEY = 'grip_strength_secret_key_2025'
DATABASE = 'grip_strength_production.db'


# ==============================================
# DATABASE INITIALIZATION
# ==============================================

def init_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'patient',
            baseline_grip REAL DEFAULT 0,
            assigned_device_id TEXT,
            created_at TEXT NOT NULL
        )
    ''')

    # Devices table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            device_name TEXT NOT NULL,
            device_type TEXT NOT NULL,
            location TEXT,
            firmware_version TEXT,
            last_seen TEXT,
            created_at TEXT NOT NULL
        )
    ''')

    # Measurements table - UPDATED FOR 6 SENSORS
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            device_id TEXT,
            session_id TEXT,
            timestamp TEXT NOT NULL,
            sensor1 REAL NOT NULL,
            sensor2 REAL NOT NULL,
            sensor3 REAL NOT NULL,
            sensor4 REAL NOT NULL,
            sensor5 REAL DEFAULT 0,
            sensor6 REAL DEFAULT 0,
            total_grip REAL NOT NULL,
            recovery_percent REAL,
            recovery_stage INTEGER,
            health_status TEXT,
            ml_confidence REAL DEFAULT 0.0,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (device_id) REFERENCES devices (device_id)
        )
    ''')

    # Check if sensor5 and sensor6 columns exist, add them if not
    cursor.execute("PRAGMA table_info(measurements)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'sensor5' not in columns:
        print("Adding sensor5 column to measurements table...")
        cursor.execute('ALTER TABLE measurements ADD COLUMN sensor5 REAL DEFAULT 0')

    if 'sensor6' not in columns:
        print("Adding sensor6 column to measurements table...")
        cursor.execute('ALTER TABLE measurements ADD COLUMN sensor6 REAL DEFAULT 0')

    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            device_id TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            session_type TEXT,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Treatment notes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS treatment_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES users (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✓ Database initialized with 6-sensor support")


# ==============================================
# HELPER FUNCTIONS
# ==============================================

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def generate_token(user_id, role):
    payload = {
        'user_id': user_id,
        'role': role,
        'exp': datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except:
        return None


def require_auth(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        payload = verify_token(token)

        if not payload:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401

        request.user_id = payload['user_id']
        request.user_role = payload['role']
        return f(*args, **kwargs)

    wrapper.__name__ = f.__name__
    return wrapper


def require_admin(f):
    def wrapper(*args, **kwargs):
        if request.user_role not in ['admin', 'doctor']:
            return jsonify({'success': False, 'error': 'Admin access required'}), 403
        return f(*args, **kwargs)

    wrapper.__name__ = f.__name__
    return wrapper


# ==============================================
# AUTH ENDPOINTS
# ==============================================

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json

    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM users WHERE email = ?', (data['email'],))
        if cursor.fetchone():
            return jsonify({'success': False, 'error': 'Email already registered'})

        cursor.execute('''
            INSERT INTO users (name, email, password_hash, age, gender, role, baseline_grip, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['name'],
            data['email'],
            hash_password(data['password']),
            data['age'],
            data['gender'],
            data.get('role', 'patient'),
            data.get('baseline_grip', 0),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Registration successful'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, name, email, age, gender, role, baseline_grip, assigned_device_id
        FROM users
        WHERE email = ? AND password_hash = ?
    ''', (data['email'], hash_password(data['password'])))

    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({'success': False, 'error': 'Invalid credentials'})

    user_dict = dict(user)
    token = generate_token(user_dict['id'], user_dict['role'])

    return jsonify({
        'success': True,
        'token': token,
        'user': user_dict
    })


# ==============================================
# PATIENT ENDPOINTS
# ==============================================

@app.route('/api/progress', methods=['GET'])
@require_auth
def get_progress():
    user_id = request.user_id

    print(f"\n📈 /api/progress called for user_id: {user_id}")

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        print(f"   ERROR: User {user_id} not found!")
        return jsonify({'error': 'User not found'}), 404

    user = dict(user)
    print(f"   User: {user['name']} (baseline: {user.get('baseline_grip', 0)} kg)")

    cursor.execute('''
        SELECT total_grip, recovery_percent, recovery_stage, timestamp, ml_confidence
        FROM measurements
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 30
    ''', (user_id,))

    measurements = cursor.fetchall()
    print(f"   Found {len(measurements)} measurements")

    conn.close()

    progress_history = []
    for m in measurements:
        progress_history.append({
            'grip': m['total_grip'],
            'recovery': m['recovery_percent'] or 0,
            'timestamp': m['timestamp']
        })

    current_grip = progress_history[0]['grip'] if progress_history else 0
    current_recovery = progress_history[0]['recovery'] if progress_history else 0
    current_stage = measurements[0]['recovery_stage'] if measurements else 0

    print(f"   Returning: grip={current_grip:.2f}kg, stage={current_stage}, recovery={current_recovery:.1f}%")

    return jsonify({
        'user': user,
        'current': {
            'grip': current_grip,
            'recovery_percent': current_recovery,
            'recovery_stage': current_stage
        },
        'progress_history': list(reversed(progress_history))
    })


@app.route('/api/measurements/user', methods=['GET'])
@require_auth
def get_user_measurements():
    user_id = request.user_id
    limit = request.args.get('limit', 50, type=int)

    conn = get_db()
    cursor = conn.cursor()

    # Include sensor5 and sensor6 in query
    cursor.execute('''
        SELECT *
        FROM measurements
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (user_id, limit))

    measurements = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({'measurements': measurements})


@app.route('/api/session/start', methods=['POST'])
@require_auth
def start_session():
    import uuid

    data = request.json or {}
    session_id = str(uuid.uuid4())
    user_id = request.user_id
    device_id = data.get('device_id')

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT id, name, email FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()

    if not device_id:
        cursor.execute('SELECT assigned_device_id FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
        if user_data and user_data['assigned_device_id']:
            device_id = user_data['assigned_device_id']

    if device_id:
        cursor.execute('''
            SELECT session_id, user_id
            FROM sessions
            WHERE device_id = ? AND status = 'active'
        ''', (device_id,))
        existing_session = cursor.fetchone()

        if existing_session:
            cursor.execute('''
                UPDATE sessions
                SET ended_at = ?, status = 'completed'
                WHERE device_id = ? AND status = 'active'
            ''', (datetime.now().isoformat(), device_id))
            print(f"⚠️  Auto-closed previous active session for device {device_id}")

    cursor.execute('''
        INSERT INTO sessions (session_id, user_id, device_id, started_at, status, session_type)
        VALUES (?, ?, ?, ?, 'active', ?)
    ''', (session_id, user_id, device_id, datetime.now().isoformat(), data.get('session_type', 'manual')))

    conn.commit()
    conn.close()

    print(f"\n▶️  SESSION STARTED")
    print(f"   Session ID: {session_id[:8]}...")
    print(f"   User: [{user_id}] {user['name']}")
    print(f"   Device: {device_id or 'Not assigned'}")
    print(f"   Started: {datetime.now().isoformat()}")
    print(f"   ✅ Device can now send measurements\n")

    return jsonify({
        'success': True,
        'session_id': session_id,
        'device_id': device_id,
        'started_at': datetime.now().isoformat()
    })


@app.route('/api/session/stop', methods=['POST'])
@require_auth
def stop_session():
    data = request.json
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({'success': False, 'error': 'Session ID required'}), 400

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT s.session_id, s.user_id, u.name, s.device_id, s.started_at
        FROM sessions s
        LEFT JOIN users u ON s.user_id = u.id
        WHERE s.session_id = ? AND s.status = 'active'
    ''', (session_id,))

    session = cursor.fetchone()

    if not session:
        conn.close()
        return jsonify({'success': False, 'error': 'Session not found or already stopped'}), 404

    cursor.execute('''
        UPDATE sessions
        SET ended_at = ?, status = 'completed'
        WHERE session_id = ?
    ''', (datetime.now().isoformat(), session_id))

    conn.commit()
    conn.close()

    print(f"\n🛑 SESSION STOPPED")
    print(f"   Session ID: {session_id[:8]}...")
    print(f"   User: [{session['user_id']}] {session['name']}")
    print(f"   Device: {session['device_id']}")
    print(f"   Started: {session['started_at']}")
    print(f"   Ended: {datetime.now().isoformat()}")
    print(f"   ⚠️  Device will no longer accept measurements until new session started\n")

    return jsonify({'success': True, 'message': 'Session stopped successfully'})


@app.route('/api/user/update', methods=['PUT'])
@require_auth
def update_current_user():
    data = request.json
    user_id = request.user_id

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE users
        SET name = ?, age = ?, gender = ?
        WHERE id = ?
    ''', (data['name'], data['age'], data['gender'], user_id))

    conn.commit()
    conn.close()

    return jsonify({'success': True})


# ==============================================
# DOCTOR ENDPOINTS
# ==============================================

@app.route('/api/doctor/note', methods=['POST'])
@require_auth
def add_treatment_note():
    if request.user_role not in ['doctor', 'admin']:
        return jsonify({'success': False, 'error': 'Doctor access required'}), 403

    data = request.json
    doctor_id = request.user_id
    patient_id = data['patient_id']
    note = data['note']

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO treatment_notes (patient_id, doctor_id, note, created_at)
        VALUES (?, ?, ?, ?)
    ''', (patient_id, doctor_id, note, datetime.now().isoformat()))

    conn.commit()
    conn.close()

    return jsonify({'success': True})


@app.route('/api/doctor/notes', methods=['GET'])
@require_auth
def get_doctor_notes():
    if request.user_role not in ['doctor', 'admin']:
        return jsonify({'success': False, 'error': 'Doctor access required'}), 403

    doctor_id = request.user_id

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT tn.*, u.name as patient_name
        FROM treatment_notes tn
        LEFT JOIN users u ON tn.patient_id = u.id
        WHERE tn.doctor_id = ?
        ORDER BY tn.created_at DESC
        LIMIT 50
    ''', (doctor_id,))

    notes = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({'notes': notes})


@app.route('/api/measurements/user/<int:patient_id>', methods=['GET'])
@require_auth
def get_patient_measurements(patient_id):
    if request.user_role not in ['doctor', 'admin']:
        return jsonify({'success': False, 'error': 'Access denied'}), 403

    limit = request.args.get('limit', 50, type=int)

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT *
        FROM measurements
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (patient_id, limit))

    measurements = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({'measurements': measurements})


# ==============================================
# ADMIN ENDPOINTS
# ==============================================

@app.route('/api/admin/stats', methods=['GET'])
@require_auth
@require_admin
def get_admin_stats():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) as count FROM users WHERE role = "patient"')
    total_patients = cursor.fetchone()['count']

    cursor.execute('SELECT COUNT(*) as count FROM users WHERE role = "doctor"')
    total_doctors = cursor.fetchone()['count']

    cursor.execute('SELECT COUNT(*) as count FROM devices')
    total_devices = cursor.fetchone()['count']

    today = datetime.now().date().isoformat()
    cursor.execute('SELECT COUNT(*) as count FROM measurements WHERE DATE(timestamp) = ?', (today,))
    measurements_today = cursor.fetchone()['count']

    conn.close()

    return jsonify({
        'total_patients': total_patients,
        'total_doctors': total_doctors,
        'total_devices': total_devices,
        'measurements_today': measurements_today
    })


@app.route('/api/admin/users', methods=['GET'])
@require_auth
@require_admin
def get_all_users():
    role_filter = request.args.get('role')
    search_query = request.args.get('query', '')

    conn = get_db()
    cursor = conn.cursor()

    if role_filter and search_query:
        cursor.execute('''
            SELECT id, name, email, role, age, gender, baseline_grip, created_at
            FROM users
            WHERE role = ? AND (name LIKE ? OR email LIKE ? OR CAST(id AS TEXT) LIKE ?)
            ORDER BY created_at DESC
        ''', (role_filter, f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'))
    elif role_filter:
        cursor.execute('''
            SELECT id, name, email, role, age, gender, baseline_grip, created_at
            FROM users
            WHERE role = ?
            ORDER BY created_at DESC
        ''', (role_filter,))
    elif search_query:
        cursor.execute('''
            SELECT id, name, email, role, age, gender, baseline_grip, created_at
            FROM users
            WHERE name LIKE ? OR email LIKE ? OR CAST(id AS TEXT) LIKE ?
            ORDER BY created_at DESC
        ''', (f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'))
    else:
        cursor.execute('''
            SELECT id, name, email, role, age, gender, baseline_grip, created_at
            FROM users
            ORDER BY created_at DESC
        ''')

    users = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({'users': users})


@app.route('/api/users/<int:user_id>', methods=['GET'])
@require_auth
def get_user(user_id):
    if request.user_role != 'admin' and request.user_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, name, email, role, age, gender, baseline_grip, created_at
        FROM users
        WHERE id = ?
    ''', (user_id,))

    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'success': True, 'user': dict(user)})


@app.route('/api/users/<int:user_id>', methods=['PUT'])
@require_auth
def update_user(user_id):
    if request.user_role != 'admin' and request.user_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.json

    conn = get_db()
    cursor = conn.cursor()

    updates = []
    values = []

    if 'name' in data:
        updates.append('name = ?')
        values.append(data['name'])
    if 'email' in data:
        cursor.execute('SELECT id FROM users WHERE email = ? AND id != ?', (data['email'], user_id))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'Email already exists'}), 400
        updates.append('email = ?')
        values.append(data['email'])
    if 'age' in data:
        updates.append('age = ?')
        values.append(data['age'])
    if 'gender' in data:
        updates.append('gender = ?')
        values.append(data['gender'])
    if 'baseline_grip' in data:
        updates.append('baseline_grip = ?')
        values.append(data['baseline_grip'])
    if 'role' in data and request.user_role == 'admin':
        updates.append('role = ?')
        values.append(data['role'])
    if 'password' in data:
        updates.append('password_hash = ?')
        values.append(hash_password(data['password']))

    if not updates:
        conn.close()
        return jsonify({'error': 'No fields to update'}), 400

    values.append(user_id)

    cursor.execute(f'''
        UPDATE users
        SET {', '.join(updates)}
        WHERE id = ?
    ''', values)

    conn.commit()
    conn.close()

    return jsonify({'success': True, 'message': 'User updated successfully'})


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@require_auth
def delete_user(user_id):
    if request.user_role != 'admin':
        return jsonify({'error': 'Unauthorized. Only administrators can delete users.'}), 403

    if request.user_id == user_id:
        return jsonify({'error': 'Cannot delete your own account'}), 400

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT id, name, email FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    try:
        cursor.execute('DELETE FROM measurements WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM treatment_notes WHERE patient_id = ?', (user_id,))
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))

        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'message': f'User {user["name"]} ({user["email"]}) has been deleted successfully'
        })
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({'error': f'Failed to delete user: {str(e)}'}), 500


@app.route('/api/admin/user/<int:user_id>/update', methods=['PUT'])
@require_auth
@require_admin
def admin_update_user(user_id):
    return update_user(user_id)


@app.route('/api/admin/device/<device_id>/test', methods=['POST'])
@require_auth
@require_admin
def test_device(device_id):
    conn = get_db()
    cursor = conn.cursor()

    # Get most recent measurement - include sensor5 and sensor6
    cursor.execute('''
        SELECT sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, timestamp
        FROM measurements
        WHERE device_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (device_id,))

    latest = cursor.fetchone()
    conn.close()

    if latest:
        sensor_status = []
        sensors_ok = 0

        # Check all 6 sensors
        for i in range(1, 7):
            sensor_val = latest[f'sensor{i}'] if latest[f'sensor{i}'] is not None else 0
            if sensor_val > 0.1:
                sensors_ok += 1
                sensor_status.append(f"✅ Sensor {i}: {sensor_val:.2f} kg - OK")
            else:
                sensor_status.append(f"⚠️ Sensor {i}: {sensor_val:.2f} kg - Low/No signal")

        if sensors_ok == 6:
            overall_status = '✅ All 6 sensors operational'
        elif sensors_ok >= 3:
            overall_status = f'⚠️ {sensors_ok}/6 sensors operational'
        else:
            overall_status = '❌ Multiple sensor failures'

        return jsonify({
            'success': True,
            'sensor1': round(latest['sensor1'] if latest['sensor1'] else 0, 2),
            'sensor2': round(latest['sensor2'] if latest['sensor2'] else 0, 2),
            'sensor3': round(latest['sensor3'] if latest['sensor3'] else 0, 2),
            'sensor4': round(latest['sensor4'] if latest['sensor4'] else 0, 2),
            'sensor5': round(latest['sensor5'] if latest['sensor5'] else 0, 2),
            'sensor6': round(latest['sensor6'] if latest['sensor6'] else 0, 2),
            'status': overall_status,
            'sensor_status': sensor_status,
            'last_reading': latest['timestamp'],
            'message': f'Showing real-time data from 6-sensor device (last reading: {latest["timestamp"]})'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'No sensor data available',
            'message': 'Device has not sent any measurements yet. Please start a measurement session.'
        }), 404


# ==============================================
# DEVICE ENDPOINTS
# ==============================================

@app.route('/api/devices/auto-register', methods=['POST'])
def auto_register_device():
    try:
        data = request.json

        if not data or 'device_id' not in data:
            return jsonify({'success': False, 'error': 'Device ID is required'}), 400

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('SELECT device_id FROM devices WHERE device_id = ?', (data['device_id'],))
        existing = cursor.fetchone()

        if existing:
            cursor.execute('''
                UPDATE devices
                SET last_seen = ?, firmware_version = ?
                WHERE device_id = ?
            ''', (datetime.now().isoformat(), data.get('firmware_version', '1.0'), data['device_id']))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'message': 'Device already registered', 'is_new': False})

        cursor.execute('''
            INSERT INTO devices (device_id, device_name, device_type, location, firmware_version, created_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['device_id'],
            data.get('device_name', data['device_id']),
            data.get('device_type', 'Unknown'),
            data.get('location', 'Unassigned'),
            data.get('firmware_version', '1.0'),
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Device registered successfully', 'is_new': True})
    except Exception as e:
        print(f"Error auto-registering device: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/devices/register', methods=['POST'])
@require_auth
@require_admin
def register_device():
    try:
        data = request.json

        if not data or 'device_id' not in data:
            return jsonify({'success': False, 'error': 'Device ID is required'}), 400

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('SELECT device_id FROM devices WHERE device_id = ?', (data['device_id'],))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'Device ID already exists'}), 400

        cursor.execute('''
            INSERT INTO devices (device_id, device_name, device_type, location, firmware_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['device_id'],
            data.get('device_name', data['device_id']),
            data.get('device_type', 'Unknown'),
            data.get('location', 'Unassigned'),
            data.get('firmware_version', '1.0'),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Device registered successfully'})
    except Exception as e:
        print(f"Error registering device: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/devices/<device_id>', methods=['PUT'])
@require_auth
@require_admin
def update_device(device_id):
    try:
        data = request.json

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('SELECT device_id FROM devices WHERE device_id = ?', (device_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'Device not found'}), 404

        cursor.execute('''
            UPDATE devices
            SET device_name = ?, device_type = ?, location = ?
            WHERE device_id = ?
        ''', (
            data.get('device_name', ''),
            data.get('device_type', 'hospital'),
            data.get('location', ''),
            device_id
        ))

        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Device updated successfully'})
    except Exception as e:
        print(f"Error updating device: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/devices/<device_id>', methods=['DELETE'])
@require_auth
@require_admin
def delete_device(device_id):
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('SELECT device_id FROM devices WHERE device_id = ?', (device_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'Device not found'}), 404

        cursor.execute('SELECT COUNT(*) as count FROM users WHERE assigned_device_id = ?', (device_id,))
        assigned_count = cursor.fetchone()['count']

        if assigned_count > 0:
            conn.close()
            return jsonify({
                'success': False,
                'error': f'Cannot delete device. It is currently assigned to {assigned_count} user(s). Unassign first.'
            }), 400

        cursor.execute('DELETE FROM devices WHERE device_id = ?', (device_id,))

        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Device deleted successfully'})
    except Exception as e:
        print(f"Error deleting device: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/devices/list', methods=['GET'])
@require_auth
def list_devices():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM devices ORDER BY created_at DESC')
    devices = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({'devices': devices})


@app.route('/api/devices/assign', methods=['POST'])
@require_auth
@require_admin
def assign_device():
    data = request.json

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE users
        SET assigned_device_id = ?
        WHERE id = ?
    ''', (data['device_id'], data['user_id']))

    conn.commit()
    conn.close()

    return jsonify({'success': True})


@app.route('/api/devices/unassign', methods=['POST'])
@require_auth
@require_admin
def unassign_device():
    data = request.json

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE users
        SET assigned_device_id = NULL
        WHERE id = ?
    ''', (data['user_id'],))

    conn.commit()
    conn.close()

    return jsonify({'success': True, 'message': 'Device unassigned successfully'})


# ==============================================
# DATA INGESTION - UPDATED FOR 6 SENSORS
# ==============================================

@app.route('/api/data/ingest', methods=['POST'])
def ingest_data():
    data = request.json

    device_id = data.get('device_id')

    if not device_id:
        return jsonify({'success': False, 'error': 'Device ID is required'}), 400

    conn = get_db()
    cursor = conn.cursor()

    # Auto-register device if not exists
    cursor.execute('SELECT device_id FROM devices WHERE device_id = ?', (device_id,))
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO devices (device_id, device_name, device_type, location, firmware_version, created_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id,
            f"Device {device_id[-6:]}",
            'Unknown',
            'Unassigned',
            '1.0',
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()
        print(f"✓ Auto-registered new device: {device_id}")

    cursor.execute('''
        SELECT user_id, session_id
        FROM sessions
        WHERE device_id = ? AND status = 'active'
        ORDER BY started_at DESC
        LIMIT 1
    ''', (device_id,))

    session = cursor.fetchone()

    if not session:
        conn.close()
        print(f"⚠️  No active session for device {device_id}")
        return jsonify({
            'success': False,
            'error': 'No active session. Please START MEASUREMENT in dashboard first.',
            'action_required': 'start_session'
        }), 400

    user_id = session['user_id']
    session_id = session['session_id']

    cursor.execute('SELECT id, name, age, gender, baseline_grip, created_at FROM users WHERE id = ?', (user_id,))
    user_data = cursor.fetchone()

    if not user_data:
        conn.close()
        return jsonify({'success': False, 'error': 'User not found'}), 404

    print(f"\n📊 Processing 6-sensor measurement for User ID {user_id} ({user_data['name']})")
    print(
        f"   Patient age: {user_data['age']}, gender: {user_data['gender']}, baseline: {user_data['baseline_grip']} kg")

    age = user_data['age'] if user_data['age'] else 30
    gender = user_data['gender'] if user_data['gender'] else 'male'
    baseline_grip = user_data['baseline_grip'] if user_data['baseline_grip'] else 5.0

    cursor.execute('''
        SELECT MIN(timestamp) as first_measurement
        FROM measurements
        WHERE user_id = ?
    ''', (user_id,))
    first_measurement = cursor.fetchone()['first_measurement']

    if first_measurement:
        first_date = datetime.fromisoformat(first_measurement)
        days_in_therapy = (datetime.now() - first_date).days
    else:
        days_in_therapy = 0

    # Get all 6 sensor values
    sensor1 = data.get('sensor1', 0)
    sensor2 = data.get('sensor2', 0)
    sensor3 = data.get('sensor3', 0)
    sensor4 = data.get('sensor4', 0)
    sensor5 = data.get('sensor5', 0)
    sensor6 = data.get('sensor6', 0)

    total_grip = data.get('total_grip', sensor1 + sensor2 + sensor3 + sensor4 + sensor5 + sensor6)

    print(
        f"   6-Sensor Grip: S1={sensor1:.2f}, S2={sensor2:.2f}, S3={sensor3:.2f}, S4={sensor4:.2f}, S5={sensor5:.2f}, S6={sensor6:.2f}")
    print(f"   Total: {total_grip:.2f} kg, Days in therapy: {days_in_therapy}")

    # Use ML model for prediction
    try:
        print(f"\n🤖 Calling ML model...")
        ml_prediction = ml_predictor.predict_recovery(
            age=age,
            gender=gender,
            current_grip=total_grip,
            baseline_grip=baseline_grip,
            days_in_therapy=days_in_therapy
        )

        recovery_stage = ml_prediction['stage']
        recovery_percent = ml_prediction['recovery_percent']
        ml_confidence = ml_prediction['confidence']

        recovery_percent = min(recovery_percent, 100.0)

        print(
            f"ML Prediction: Stage {recovery_stage}, Confidence {ml_confidence:.1f}%, Recovery {recovery_percent:.1f}%")
    except Exception as e:
        print(f"ML prediction failed: {e}, using fallback calculation")
        recovery_percent = min((total_grip / baseline_grip * 100) if baseline_grip > 0 else 0, 100.0)

        if recovery_percent < 25:
            recovery_stage = 0
        elif recovery_percent < 50:
            recovery_stage = 1
        elif recovery_percent < 75:
            recovery_stage = 2
        elif recovery_percent < 90:
            recovery_stage = 3
        else:
            recovery_stage = 4

        ml_confidence = 0.0

    if recovery_stage <= 1:
        health_status = 'Critical'
    elif recovery_stage == 2:
        health_status = 'Recovering'
    elif recovery_stage == 3:
        health_status = 'Good'
    else:
        health_status = 'Excellent'

    # Insert measurement with ALL 6 sensors
    cursor.execute('''
        INSERT INTO measurements (user_id, device_id, session_id, timestamp,
                                  sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, total_grip,
                                  recovery_percent, recovery_stage, health_status, ml_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, device_id, session_id, datetime.now().isoformat(),
        sensor1, sensor2, sensor3, sensor4, sensor5, sensor6,
        total_grip, recovery_percent, recovery_stage, health_status, ml_confidence
    ))

    measurement_id = cursor.lastrowid

    cursor.execute('''
        UPDATE devices
        SET last_seen = ?
        WHERE device_id = ?
    ''', (datetime.now().isoformat(), device_id))

    conn.commit()
    conn.close()

    print(f"\n✅ 6-Sensor Measurement #{measurement_id} saved to database")
    print(f"   User: {user_id}, Grip: {total_grip:.2f}kg, Stage: {recovery_stage}, Recovery: {recovery_percent:.1f}%\n")

    return jsonify({
        'success': True,
        'measurement_id': measurement_id,
        'recovery_percent': recovery_percent,
        'recovery_stage': recovery_stage,
        'health_status': health_status,
        'ml_confidence': ml_confidence
    })


# ==============================================
# HEALTH CHECK
# ==============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'ml_model': 'loaded' if ml_predictor.model else 'not loaded',
        'database': 'connected',
        'server': 'running',
        'sensors': '6-sensor support enabled'
    })


@app.route('/api/ml/info', methods=['GET'])
def get_ml_info():
    return jsonify({
        'model_loaded': ml_predictor.model is not None,
        'scaler_loaded': ml_predictor.scaler is not None,
        'norms_loaded': ml_predictor.norms_data is not None,
        'model_type': 'TensorFlow Keras Neural Network',
        'sensors_supported': 6,
        'recovery_stages': [
            'Stage 0: Severe Paralysis (0-10%)',
            'Stage 1: Poor Recovery (10-25%)',
            'Stage 2: Fair Recovery (25-50%)',
            'Stage 3: Good Recovery (50-75%)',
            'Stage 4: Excellent Recovery (75-100%)'
        ]
    })


if __name__ == '__main__':
    init_database()
    print("✓ Starting Flask server with 6-sensor support...")
    print("✓ Server running on http://localhost:5000")
    print("✓ All endpoints active for Patient, Doctor, and Admin roles")
    print("✓ 6 FSR sensors supported")
    app.run(debug=True, host='0.0.0.0', port=5000)