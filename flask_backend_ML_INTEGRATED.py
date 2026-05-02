"""
Flask Backend - HYBRID CALIBRATION APPROACH
Shows only ONLINE devices (last seen within 30 seconds)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import jwt
import hashlib
from datetime import datetime, timedelta
import json
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
SECRET_KEY = 'your-secret-key-change-in-production'
DB_NAME = 'grip_strength_production.db'
CALIBRATION_FACTOR = 1.28
ONLINE_THRESHOLD_SECONDS = 30  # Device is "online" if seen within last 30 seconds


# ==========================================
# MACHINE LEARNING MODEL LOADER kaween
# ==========================================

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS users
                   (
                       id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                       username           TEXT,
                       name               TEXT        NOT NULL,
                       password_hash      TEXT        NOT NULL,
                       email              TEXT UNIQUE NOT NULL,
                       role               TEXT      DEFAULT 'patient',
                       age                INTEGER,
                       gender             TEXT,
                       baseline_grip      REAL      DEFAULT 0,
                       assigned_device_id TEXT,
                       created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS sessions
                   (
                       id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                       session_id               TEXT UNIQUE NOT NULL,
                       user_id                  INTEGER     NOT NULL,
                       device_id                TEXT        NOT NULL,
                       baseline_grip            REAL,
                       baseline_equivalent_grip REAL,
                       started_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       ended_at                 TIMESTAMP,
                       status                   TEXT      DEFAULT 'active',
                       session_type             TEXT,
                       FOREIGN KEY (user_id) REFERENCES users (id)
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS measurements
                   (
                       id               INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id          INTEGER NOT NULL,
                       device_id        TEXT,
                       session_id       TEXT,
                       timestamp        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       sensor1          REAL    NOT NULL,
                       sensor2          REAL    NOT NULL,
                       sensor3          REAL    NOT NULL,
                       sensor4          REAL    NOT NULL,
                       sensor5          REAL      DEFAULT 0,
                       sensor6          REAL      DEFAULT 0,
                       total_grip       REAL    NOT NULL,
                       equivalent_grip  REAL,
                       recovery_percent REAL,
                       recovery_stage   INTEGER,
                       health_status    TEXT,
                       ml_confidence    REAL      DEFAULT 0.0,
                       FOREIGN KEY (user_id) REFERENCES users (id)
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS devices
                   (
                       device_id        TEXT PRIMARY KEY,
                       device_name      TEXT NOT NULL,
                       device_type      TEXT NOT NULL,
                       location         TEXT,
                       firmware_version TEXT,
                       last_seen        TEXT,
                       created_at       TEXT NOT NULL
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS treatment_notes
                   (
                       id         INTEGER PRIMARY KEY AUTOINCREMENT,
                       patient_id INTEGER NOT NULL,
                       doctor_id  INTEGER NOT NULL,
                       note       TEXT    NOT NULL,
                       created_at TEXT    NOT NULL,
                       FOREIGN KEY (patient_id) REFERENCES users (id),
                       FOREIGN KEY (doctor_id) REFERENCES users (id)
                   )
                   ''')

    cursor.execute("PRAGMA table_info(measurements)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'equivalent_grip' not in columns:
        cursor.execute("ALTER TABLE measurements ADD COLUMN equivalent_grip REAL")
    if 'sensor5' not in columns:
        cursor.execute("ALTER TABLE measurements ADD COLUMN sensor5 REAL DEFAULT 0")
    if 'sensor6' not in columns:
        cursor.execute("ALTER TABLE measurements ADD COLUMN sensor6 REAL DEFAULT 0")

    cursor.execute("PRAGMA table_info(sessions)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'baseline_equivalent_grip' not in columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN baseline_equivalent_grip REAL")

    conn.commit()
    conn.close()
    print("✓ Database initialized")


def get_db():
    conn = sqlite3.connect(DB_NAME)
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
        return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
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


def is_device_online(last_seen):
    """Check if device is currently online (sent data within threshold)"""
    if not last_seen:
        return False
    try:
        last_seen_dt = datetime.fromisoformat(last_seen)
        diff = (datetime.now() - last_seen_dt).total_seconds()
        return diff <= ONLINE_THRESHOLD_SECONDS
    except:
        return False


# ==================== RECOVERY CALCULATION ====================

def calculate_recovery(user_id, current_equivalent_grip, baseline_equivalent_grip=None):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT age, gender FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return {'stage': 0, 'percent': 0.0}

        age, gender = user['age'], user['gender']

        if gender == 'male':
            if age < 30: expected_grip = 48.0
            elif age < 50: expected_grip = 45.0
            elif age < 70: expected_grip = 40.0
            else: expected_grip = 35.0
        else:
            if age < 30: expected_grip = 32.0
            elif age < 50: expected_grip = 30.0
            elif age < 70: expected_grip = 26.0
            else: expected_grip = 22.0

        reference_grip = baseline_equivalent_grip if baseline_equivalent_grip else expected_grip
        recovery_percent = (current_equivalent_grip / reference_grip) * 100

        if recovery_percent < 10: stage = 0
        elif recovery_percent < 25: stage = 1
        elif recovery_percent < 50: stage = 2
        elif recovery_percent < 75: stage = 3
        else: stage = 4

        return {'stage': stage, 'percent': min(recovery_percent, 100.0), 'expected_grip': expected_grip}
    except Exception as e:
        print(f"Error: {e}")
        return {'stage': 0, 'percent': 0.0}


# ==================== AUTH ====================

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'success': False, 'error': 'Missing credentials'}), 400

    password_hash = hash_password(password)
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ? AND password_hash = ?', (email, password_hash))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    token = generate_token(user['id'], user['role'])
    return jsonify({'success': True, 'token': token, 'user': dict(user)}), 200


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'patient')
    age = data.get('age')
    gender = data.get('gender')
    baseline_grip = data.get('baseline_grip', 0)

    if not all([name, email, password]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    password_hash = hash_password(password)
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute('''INSERT INTO users (name, email, password_hash, role, age, gender, baseline_grip)
                          VALUES (?, ?, ?, ?, ?, ?, ?)''',
                       (name, email, password_hash, role, age, gender, baseline_grip))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Registration successful'}), 201
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'success': False, 'error': 'Email already registered'}), 409


# ==================== PATIENT ENDPOINTS ====================

@app.route('/api/progress', methods=['GET'])
@require_auth
def get_progress():
    user_id = request.user_id
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    cursor.execute('''SELECT total_grip, recovery_percent, recovery_stage, timestamp, ml_confidence
                      FROM measurements WHERE user_id = ? ORDER BY timestamp DESC LIMIT 30''', (user_id,))
    measurements = cursor.fetchall()
    conn.close()

    progress_history = [{'grip': m['total_grip'], 'recovery': m['recovery_percent'] or 0, 'timestamp': m['timestamp']}
                        for m in measurements]

    current_grip = progress_history[0]['grip'] if progress_history else 0
    current_recovery = progress_history[0]['recovery'] if progress_history else 0
    current_stage = measurements[0]['recovery_stage'] if measurements else 0

    return jsonify({
        'user': dict(user),
        'current': {'grip': current_grip, 'recovery_percent': current_recovery, 'recovery_stage': current_stage},
        'progress_history': list(reversed(progress_history))
    })


@app.route('/api/measurements/user', methods=['GET'])
@require_auth
def get_user_measurements():
    user_id = request.user_id
    limit = request.args.get('limit', 50, type=int)
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM measurements WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, limit))
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

    cursor.execute('SELECT id, name, assigned_device_id FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()

    if not device_id and user['assigned_device_id']:
        device_id = user['assigned_device_id']

    if not device_id:
        device_id = 'ESP32-DEFAULT'

    cursor.execute('UPDATE sessions SET ended_at = ?, status = "completed" WHERE device_id = ? AND status = "active"',
                   (datetime.now().isoformat(), device_id))
    conn.commit()

    cursor.execute('''INSERT INTO sessions (session_id, user_id, device_id, started_at, status, session_type)
                      VALUES (?, ?, ?, ?, 'active', ?)''',
                   (session_id, user_id, device_id, datetime.now().isoformat(), data.get('session_type', 'manual')))
    conn.commit()
    conn.close()

    print(f"\n▶️  SESSION STARTED: User [{user_id}] {user['name']} | Device: {device_id}\n")

    return jsonify({'success': True, 'session_id': session_id, 'device_id': device_id,
                    'started_at': datetime.now().isoformat()}), 200


@app.route('/api/session/stop', methods=['POST'])
@require_auth
def stop_session():
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'error': 'Session ID required'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE sessions SET ended_at = ?, status = "completed" WHERE session_id = ?',
                   (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Session stopped successfully'})


@app.route('/api/user/update', methods=['PUT'])
@require_auth
def update_current_user():
    data = request.json
    user_id = request.user_id
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET name = ?, age = ?, gender = ? WHERE id = ?',
                   (data['name'], data['age'], data['gender'], user_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ==================== DATA INGESTION (auto-registers device) ====================

@app.route('/api/data/ingest', methods=['POST'])
def ingest_data():
    """Receive data from ESP32 - AUTO-REGISTERS device (only online devices visible in UI)"""
    try:
        data = request.json
        device_id = data.get('device_id')

        if not device_id:
            return jsonify({'success': False, 'error': 'device_id required'}), 400

        total_grip = data.get('total_grip')
        equivalent_grip = data.get('equivalent_grip')

        if equivalent_grip is None and total_grip is not None:
            equivalent_grip = total_grip * CALIBRATION_FACTOR

        sensor1 = data.get('sensor1', 0)
        sensor2 = data.get('sensor2', 0)
        sensor3 = data.get('sensor3', 0)
        sensor4 = data.get('sensor4', 0)
        sensor5 = data.get('sensor5', 0)
        sensor6 = data.get('sensor6', 0)

        conn = get_db()
        cursor = conn.cursor()

        # AUTO-REGISTER device when ESP32 sends data
        cursor.execute('SELECT device_id FROM devices WHERE device_id = ?', (device_id,))
        if not cursor.fetchone():
            cursor.execute('''INSERT INTO devices (device_id, device_name, device_type, location, firmware_version, created_at, last_seen)
                              VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (device_id, f'Device {device_id}', 'hospital', 'Auto',
                            '1.0', datetime.now().isoformat(), datetime.now().isoformat()))
            print(f"✓ Auto-registered device: {device_id}")

        # ALWAYS update last_seen (this is critical for "online" detection)
        cursor.execute('UPDATE devices SET last_seen = ? WHERE device_id = ?',
                       (datetime.now().isoformat(), device_id))

        # Find active session
        cursor.execute('''SELECT session_id, user_id, baseline_equivalent_grip FROM sessions
                          WHERE device_id = ? AND status = 'active' ORDER BY started_at DESC LIMIT 1''', (device_id,))
        session = cursor.fetchone()

        if not session:
            conn.commit()  # Save the last_seen update even without session
            conn.close()
            return jsonify({
                'success': False,
                'error': 'No active session. Please START MEASUREMENT in dashboard first.'
            }), 400

        session_id = session['session_id']
        user_id = session['user_id']
        baseline_equiv = session['baseline_equivalent_grip']

        recovery_data = calculate_recovery(user_id, equivalent_grip, baseline_equiv)

        if recovery_data['stage'] <= 1: health_status = 'Critical'
        elif recovery_data['stage'] == 2: health_status = 'Recovering'
        elif recovery_data['stage'] == 3: health_status = 'Good'
        else: health_status = 'Excellent'

        cursor.execute('''INSERT INTO measurements (user_id, device_id, session_id, timestamp,
                          sensor1, sensor2, sensor3, sensor4, sensor5, sensor6,
                          total_grip, equivalent_grip, recovery_percent, recovery_stage, health_status, ml_confidence)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (user_id, device_id, session_id, datetime.now().isoformat(),
                        sensor1, sensor2, sensor3, sensor4, sensor5, sensor6,
                        total_grip, equivalent_grip, recovery_data['percent'], recovery_data['stage'],
                        health_status, 95.0))

        conn.commit()
        conn.close()

        print(f"✓ Data: {device_id} | user={user_id} | grip={total_grip:.2f}kg | stage={recovery_data['stage']}")

        return jsonify({
            'success': True,
            'total_grip': total_grip,
            'equivalent_grip': equivalent_grip,
            'recovery_stage': recovery_data['stage'],
            'recovery_percent': recovery_data['percent']
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== DOCTOR ENDPOINTS ====================

@app.route('/api/doctor/note', methods=['POST'])
@require_auth
def add_treatment_note():
    if request.user_role not in ['doctor', 'admin']:
        return jsonify({'success': False, 'error': 'Doctor access required'}), 403

    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO treatment_notes (patient_id, doctor_id, note, created_at) VALUES (?, ?, ?, ?)',
                   (data['patient_id'], request.user_id, data['note'], datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/doctor/notes', methods=['GET'])
@require_auth
def get_doctor_notes():
    if request.user_role not in ['doctor', 'admin']:
        return jsonify({'success': False, 'error': 'Doctor access required'}), 403

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''SELECT tn.*, u.name as patient_name FROM treatment_notes tn
                      LEFT JOIN users u ON tn.patient_id = u.id WHERE tn.doctor_id = ?
                      ORDER BY tn.created_at DESC LIMIT 50''', (request.user_id,))
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
    cursor.execute('SELECT * FROM measurements WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (patient_id, limit))
    measurements = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({'measurements': measurements})


# ==================== ADMIN ENDPOINTS ====================

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

    # Count only ONLINE devices
    cursor.execute('SELECT * FROM devices')
    all_devices = cursor.fetchall()
    online_count = sum(1 for d in all_devices if is_device_online(d['last_seen']))

    today = datetime.now().date().isoformat()
    cursor.execute('SELECT COUNT(*) as count FROM measurements WHERE DATE(timestamp) = ?', (today,))
    measurements_today = cursor.fetchone()['count']

    conn.close()

    return jsonify({
        'total_patients': total_patients,
        'total_doctors': total_doctors,
        'total_devices': online_count,  # Show online count
        'measurements_today': measurements_today
    })


@app.route('/api/admin/users', methods=['GET'])
@require_auth
@require_admin
def get_all_users():
    role_filter = request.args.get('role')
    conn = get_db()
    cursor = conn.cursor()

    if role_filter:
        cursor.execute('SELECT * FROM users WHERE role = ? ORDER BY created_at DESC', (role_filter,))
    else:
        cursor.execute('SELECT * FROM users ORDER BY created_at DESC')

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
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
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

    if 'name' in data: updates.append('name = ?'); values.append(data['name'])
    if 'age' in data: updates.append('age = ?'); values.append(data['age'])
    if 'gender' in data: updates.append('gender = ?'); values.append(data['gender'])
    if 'baseline_grip' in data: updates.append('baseline_grip = ?'); values.append(data['baseline_grip'])
    if 'role' in data and request.user_role == 'admin': updates.append('role = ?'); values.append(data['role'])
    if 'email' in data: updates.append('email = ?'); values.append(data['email'])
    if 'password' in data: updates.append('password_hash = ?'); values.append(hash_password(data['password']))

    if not updates:
        conn.close()
        return jsonify({'error': 'No fields to update'}), 400

    values.append(user_id)
    cursor.execute(f'UPDATE users SET {", ".join(updates)} WHERE id = ?', values)
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'User updated successfully'})


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@require_auth
def delete_user(user_id):
    if request.user_role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    if request.user_id == user_id:
        return jsonify({'error': 'Cannot delete your own account'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    cursor.execute('DELETE FROM measurements WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM treatment_notes WHERE patient_id = ?', (user_id,))
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))

    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': f'User {user["name"]} deleted'})


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

    cursor.execute('''SELECT sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, timestamp
                      FROM measurements WHERE device_id = ? ORDER BY timestamp DESC LIMIT 1''', (device_id,))
    latest = cursor.fetchone()
    conn.close()

    if latest:
        sensors_ok = sum(1 for i in range(1, 7) if latest[f'sensor{i}'] and latest[f'sensor{i}'] > 0.1)

        if sensors_ok == 6: overall_status = '✅ All 6 sensors operational'
        elif sensors_ok >= 3: overall_status = f'⚠️ {sensors_ok}/6 sensors operational'
        else: overall_status = '❌ Multiple sensor failures'

        return jsonify({
            'success': True,
            'sensor1': round(latest['sensor1'] or 0, 2),
            'sensor2': round(latest['sensor2'] or 0, 2),
            'sensor3': round(latest['sensor3'] or 0, 2),
            'sensor4': round(latest['sensor4'] or 0, 2),
            'sensor5': round(latest['sensor5'] or 0, 2),
            'sensor6': round(latest['sensor6'] or 0, 2),
            'status': overall_status,
            'last_reading': latest['timestamp']
        })
    else:
        return jsonify({'success': False, 'error': 'No sensor data available'}), 404


# ==================== DEVICE ENDPOINTS - ONLY ONLINE DEVICES ====================

@app.route('/api/devices/list', methods=['GET'])
@require_auth
def list_devices():
    """Returns ONLY devices that are currently online (sent data within last 30 seconds)"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM devices ORDER BY last_seen DESC')
    all_devices = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Filter only ONLINE devices
    online_devices = [d for d in all_devices if is_device_online(d['last_seen'])]

    print(f"✓ Online devices: {len(online_devices)}/{len(all_devices)}")
    for d in online_devices:
        print(f"   - {d['device_id']} (last seen: {d['last_seen']})")

    return jsonify({'devices': online_devices})


@app.route('/api/devices/register', methods=['POST'])
@require_auth
@require_admin
def register_device():
    data = request.json
    if not data or 'device_id' not in data:
        return jsonify({'success': False, 'error': 'Device ID required'}), 400

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute('''INSERT INTO devices (device_id, device_name, device_type, location, firmware_version, created_at)
                          VALUES (?, ?, ?, ?, ?, ?)''',
                       (data['device_id'], data.get('device_name', data['device_id']),
                        data.get('device_type', 'Unknown'), data.get('location', 'Unassigned'),
                        data.get('firmware_version', '1.0'), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Device registered'})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'success': False, 'error': 'Device ID already exists'}), 400


@app.route('/api/devices/<device_id>', methods=['PUT'])
@require_auth
@require_admin
def update_device(device_id):
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE devices SET device_name = ?, device_type = ?, location = ? WHERE device_id = ?',
                   (data.get('device_name', ''), data.get('device_type', 'hospital'),
                    data.get('location', ''), device_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Device updated'})


@app.route('/api/devices/<device_id>', methods=['DELETE'])
@require_auth
@require_admin
def delete_device(device_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM devices WHERE device_id = ?', (device_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Device deleted'})


@app.route('/api/devices/assign', methods=['POST'])
@require_auth
@require_admin
def assign_device():
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET assigned_device_id = ? WHERE id = ?',
                   (data['device_id'], data['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'calibration_factor': CALIBRATION_FACTOR,
        'online_threshold_seconds': ONLINE_THRESHOLD_SECONDS
    }), 200


# ==================== MAIN ====================

if __name__ == '__main__':
    init_db()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM users WHERE role = 'admin'")
    if cursor.fetchone()['count'] == 0:
        admin_hash = hash_password('admin123')
        cursor.execute('''INSERT INTO users (name, email, password_hash, role, age, gender)
                          VALUES ('System Administrator', 'admin@demo.com', ?, 'admin', 30, 'male')''', (admin_hash,))
        conn.commit()
        print("✓ Created default admin (admin@demo.com / admin123)")
    conn.close()

    PORT = int(os.environ.get('PORT', 5000))
    print(f"\n{'=' * 60}")
    print(f"Flask Backend - HYBRID CALIBRATION APPROACH")
    print(f"Calibration Factor: {CALIBRATION_FACTOR}")
    print(f"Online Threshold: {ONLINE_THRESHOLD_SECONDS} seconds")
    print(f"Listening on port {PORT}")
    print(f"{'=' * 60}\n")

    app.run(host='0.0.0.0', port=PORT, debug=False)