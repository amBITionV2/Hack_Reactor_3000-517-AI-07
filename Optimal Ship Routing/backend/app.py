import os
import math
import threading
import time
import atexit
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging
from heapq import heappush, heappop

from shapely.geometry import Point
import cartopy.feature as cfeature

# Import custom modules
from ml_model import ShipPerformancePredictor
from weather_api import WeatherOceanAPI, get_cached_weather

# --- App Initialization ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Global Variables ---
OPENWEATHER_API_KEY = "81da745c6171d7297c8d6943dd0d240e"
ML_MODEL_PATH = 'models/ship_performance_model'
RECOMPUTE_INTERVAL = 15 * 60

ml_predictor, weather_api, route_monitor_thread = None, None, None
active_routes, shutdown_flag = {}, False

# --- Geospatial & Route Utilities ---
R_EARTH_KM = 6371.0088
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
land_feature = cfeature.NaturalEarthFeature('physical', 'land', '110m')
land_geometries = list(land_feature.geometries())

def haversine_km(a, b):
    lat1, lon1 = map(math.radians, a); lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    h = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(h))

def is_land(lat, lon):
    point = Point(lon, lat)
    return any(poly.contains(point) for poly in land_geometries)

def get_navigational_hazard_penalty(lat, lon):
    palk_strait_box = [9.0, 10.5, 78.8, 80.5]
    if (palk_strait_box[0] <= lat <= palk_strait_box[1] and
        palk_strait_box[2] <= lon <= palk_strait_box[3]):
        return 999999
    return 0

def clamp_lon(lon):
    while lon < -180: lon += 360
    while lon >= 180: lon -= 360
    return lon

class Grid:
    def __init__(self, bounds, step_deg):
        self.lat_min, self.lat_max = bounds['lat_min'], bounds['lat_max']
        self.lon_min, self.lon_max = bounds['lon_min'], bounds['lon_max']
        self.step = step_deg
        self.n_lat = int(round((self.lat_max - self.lat_min) / self.step)) + 1
        self.n_lon = int(round((self.lon_max - self.lon_min) / self.step)) + 1
    def to_cell(self, lat, lon): return (round((lat - self.lat_min) / self.step), round((lon - self.lon_min) / self.step))
    def to_coord(self, cell): i, j = cell; lat = self.lat_min + i * self.step; lon = self.lon_min + j * self.step; return (lat, clamp_lon(lon))
    def in_bounds(self, cell): i, j = cell; return (0 <= i < self.n_lat) and (0 <= j < self.n_lon)
    def neighbors(self, cell):
        i, j = cell
        for di, dj in NEIGHBOR_OFFSETS:
            ni, nj = i + di, j + dj
            if self.in_bounds((ni, nj)): yield (ni, nj)

# --- Cost Functions ---
def get_environmental_penalty(lat, lon, conditions=None):
    if conditions is None and weather_api: conditions = get_cached_weather(weather_api, lat, lon)
    if not conditions: return 0.0
    penalty = 0.0; wind_speed = conditions.get('wind_speed', 0); wave_height = conditions.get('wave_height', 0); current_speed = conditions.get('current_speed', 0); visibility = conditions.get('visibility', 10)
    if wind_speed > 25: penalty += (wind_speed - 25) * 0.5
    if wave_height > 3: penalty += (wave_height - 3) * 2.0
    if visibility < 2: penalty += (2 - visibility) * 3.0
    penalty -= current_speed * 0.3
    return max(0, penalty)

def get_ml_speed_penalty(lat, lon):
    if not ml_predictor or not weather_api: return 0.0
    try:
        conditions = get_cached_weather(weather_api, lat, lon)
        features = np.array([[conditions.get('wind_speed', 10), conditions.get('wind_direction', 180), conditions.get('wave_height', 2), conditions.get('wave_period', 8), conditions.get('current_speed', 0.5), conditions.get('current_direction', 90), conditions.get('sea_temp', 22), conditions.get('air_temp', 20), conditions.get('pressure', 1013), conditions.get('visibility', 10), 180, 12]])
        prediction = ml_predictor.predict(features)
        speed_penalty = max(0, (12.0 - prediction['speed']) * 2.0)
        fuel_penalty = max(0, (prediction['fuel_consumption'] - 25.0) * 0.1)
        return speed_penalty + fuel_penalty
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        return 0.0

def cost_penalty(lat, lon, conditions=None):
    hazard_penalty = get_navigational_hazard_penalty(lat, lon)
    if hazard_penalty > 0:
        return hazard_penalty
    return get_environmental_penalty(lat, lon, conditions) + get_ml_speed_penalty(lat, lon)

# --- A* Routing Algorithm ---
def astar_route_with_conditions(grid, start_ll, end_ll):
    start, goal = grid.to_cell(*start_ll), grid.to_cell(*end_ll)
    g, parent, route_conditions = {start: 0.0}, {start: None}, {}
    if weather_api:
        try:
            dist = haversine_km(start_ll, end_ll); n_samples = min(20, max(5, int(dist / 100)))
            for i in range(n_samples):
                t = i / (n_samples - 1) if n_samples > 1 else 0
                s_lat, s_lon = start_ll[0] + t * (end_ll[0] - start_ll[0]), start_ll[1] + t * (end_ll[1] - start_ll[1])
                s_coord = (round(s_lat, 2), round(s_lon, 2))
                if s_coord not in route_conditions: route_conditions[s_coord] = get_cached_weather(weather_api, s_lat, s_lon)
        except Exception as e: logger.warning(f"Could not fetch route conditions: {e}")
    openq, visited = [], set(); heappush(openq, (0.0, start))
    while openq:
        _, current = heappop(openq)
        if current in visited: continue
        visited.add(current)
        if current == goal: break
        cur_ll = grid.to_coord(current)
        for nb in grid.neighbors(current):
            nb_ll = grid.to_coord(nb)
            if is_land(nb_ll[0], nb_ll[1]): continue
            step_dist = haversine_km(cur_ll, nb_ll); conditions = route_conditions.get((round(nb_ll[0], 2), round(nb_ll[1], 2)))
            tentative_g = g[current] + step_dist + cost_penalty(*nb_ll, conditions)
            if nb not in g or tentative_g < g[nb]:
                g[nb], parent[nb] = tentative_g, current
                h = haversine_km(nb_ll, end_ll); heappush(openq, (tentative_g + h, nb))
    if goal not in parent: return [], 0.0, {}
    path_cells, c = [], goal
    while c is not None: path_cells.append(c); c = parent.get(c)
    path_cells.reverse()
    coords = [grid.to_coord(c) for c in path_cells]
    total_distance, route_details = 0.0, []
    for i, coord in enumerate(coords):
        if i > 0: total_distance += haversine_km(coords[i - 1], coord)
        conditions = route_conditions.get((round(coord[0], 2), round(coord[1], 2)), {})
        route_details.append({'lat': coord[0], 'lon': coord[1], 'distance_from_start': total_distance, 'conditions': conditions})
    return coords, total_distance, {'route_details': route_details}

# --- Background Monitoring, Initialization, API Endpoints ---
def monitor_active_routes():
    global shutdown_flag
    while not shutdown_flag:
        try:
            routes_to_update = [rid for rid, rinfo in list(active_routes.items()) if (datetime.now() - rinfo['last_computed']).total_seconds() >= RECOMPUTE_INTERVAL]
            for route_id in routes_to_update:
                try:
                    rinfo = active_routes[route_id]; logger.info(f"Re-computing route {route_id}")
                    # Use the same dynamic logic for monitoring
                    direct_dist = haversine_km(rinfo['start'], rinfo['end'])
                    if direct_dist > 1000: step_deg = 1.0
                    elif direct_dist > 300: step_deg = 0.5
                    else: step_deg = 0.25
                    grid = Grid(rinfo['grid_config']['bounds'], step_deg)
                    new_path, new_dist, new_details = astar_route_with_conditions(grid, rinfo['start'], rinfo['end'])
                    if path_changed_significantly(rinfo['path_history'][-1], new_path):
                        logger.info(f"Route {route_id} changed significantly - updating"); rinfo['path_history'].append(new_path); rinfo.update({'distance': new_dist, 'details': new_details['route_details'], 'route_updated': True})
                    else:
                        rinfo['route_updated'] = False
                    rinfo['last_computed'] = datetime.now()
                except Exception as e: logger.error(f"Error updating route {route_id}: {e}")
            time.sleep(60)
        except Exception as e: logger.error(f"Error in route monitoring thread: {e}"); time.sleep(60)
def path_changed_significantly(old_path, new_path, threshold_km=10.0):
    if len(old_path) != len(new_path): return True
    total_dev = sum(haversine_km(p1, p2) for p1, p2 in zip(old_path, new_path))
    return (total_dev / len(old_path) if old_path else 0) > threshold_km
def initialize_ml_model():
    global ml_predictor
    try:
        ml_predictor = ShipPerformancePredictor()
        ml_predictor.load_model(ML_MODEL_PATH)
        logger.info("Loaded existing ML model")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize ML model: {e}")
        ml_predictor = None
        return False
def initialize_weather_api(): global weather_api; weather_api = WeatherOceanAPI(OPENWEATHER_API_KEY); logger.info("Weather API initialized"); return True
def start_background_monitoring(): global route_monitor_thread; route_monitor_thread = threading.Thread(target=monitor_active_routes, daemon=True); route_monitor_thread.start(); logger.info("Started background route monitoring")

@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok", "ml_model": ml_predictor is not None, "weather_api": weather_api is not None, "active_routes": len(active_routes)})

@app.route('/route', methods=['POST'])
def create_route():
    data = request.get_json(force=True)
    start, end = data['start'], data['end']
    
    # DYNAMIC BOUNDING BOX LOGIC
    direct_dist_km = haversine_km(start, end)
    if direct_dist_km > 1000:
        step_deg, padding = 1.0, 10.0
    elif direct_dist_km > 300:
        step_deg, padding = 0.5, 7.0
    else:
        step_deg, padding = 0.25, 5.0
    
    logger.info(f"Route distance {direct_dist_km:.0f} km. Using grid step: {step_deg}")

    bounds = {
        'lat_min': max(min(start[0], end[0]) - padding, -90),
        'lat_max': min(max(start[0], end[0]) + padding, 90),
        'lon_min': max(min(start[1], end[1]) - padding, -180),
        'lon_max': min(max(start[1], end[1]) + padding, 180)
    }
    logger.info(f"Using dynamic bounds: {bounds}")
    
    try:
        grid = Grid(bounds, step_deg)
        path, dist_km, details = astar_route_with_conditions(grid, start, end)
        
        route_id = f"route_{int(time.time() * 1000)}"
        route_details_data = details.get('route_details', [])
        
        active_routes[route_id] = {
            'path_history': [path], 'start': start, 'end': end, 
            'distance': dist_km, 'details': route_details_data, 
            'last_computed': datetime.now(), 'grid_config': {'bounds': bounds, 'step_deg': step_deg}
        }
        
        return jsonify({
            'route_id': route_id,
            'path_history': [path],
            'distance_km': dist_km,
            'route_details': route_details_data,
            'grid_step_used': step_deg
        }), 200
    except Exception as e:
        logger.error(f"Error computing route: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/route/<route_id>', methods=['GET', 'POST', 'DELETE'])
def handle_route(route_id):
    if route_id not in active_routes: return jsonify({'error': 'Route not found'}), 404
    if request.method == 'DELETE':
        del active_routes[route_id]; logger.info(f"Route {route_id} cleared by user.")
        return jsonify({'message': f'Route {route_id} cleared successfully'}), 200
    if request.method == 'GET':
        rinfo = active_routes[route_id]; rinfo.setdefault('route_updated', False)
        return jsonify({'route_id': route_id, 'last_computed': rinfo['last_computed'].isoformat(), 'route_updated': rinfo['route_updated'], 'path_history': rinfo['path_history'], 'distance_km': rinfo.get('distance', 0), 'details': rinfo.get('details', [])}), 200
    if request.method == 'POST':
        rinfo = active_routes[route_id]
        direct_dist = haversine_km(rinfo['start'], rinfo['end'])
        if direct_dist > 1000: step_deg = 1.0
        elif direct_dist > 300: step_deg = 0.5
        else: step_deg = 0.25
        grid = Grid(rinfo['grid_config']['bounds'], step_deg)
        new_path, new_dist, new_details = astar_route_with_conditions(grid, rinfo['start'], rinfo['end'])
        rinfo['path_history'].append(new_path); rinfo.update({'distance': new_dist, 'details': new_details['route_details'], 'last_computed': datetime.now(), 'route_updated': True})
        return jsonify({'route_id': route_id, 'updated': True, 'path_history': rinfo['path_history'], 'distance_km': new_dist}), 200

@app.route('/weather', methods=['GET'])
def get_weather():
    lat, lon = float(request.args.get('lat', 0)), float(request.args.get('lon', 0))
    if not weather_api: return jsonify({'error': 'Weather API not available'}), 503
    try:
        conditions = get_cached_weather(weather_api, lat, lon)
        if 'timestamp' in conditions and isinstance(conditions['timestamp'], datetime): conditions['timestamp'] = conditions['timestamp'].isoformat()
        return jsonify(conditions)
    except Exception as e: logger.error(f"Error fetching weather: {e}"); return jsonify({'error': str(e)}), 500
@app.route('/predict', methods=['POST'])
def predict_performance():
    if not ml_predictor: return jsonify({'error': 'ML model not available'}), 503
    try:
        conditions = request.get_json().get('conditions', {})
        features = np.array([[conditions.get('wind_speed', 10), conditions.get('wind_direction', 180), conditions.get('wave_height', 2), conditions.get('wave_period', 8), conditions.get('current_speed', 0.5), conditions.get('current_direction', 90), conditions.get('sea_temp', 22), conditions.get('air_temp', 20), conditions.get('pressure', 1013), conditions.get('visibility', 10), 180, 12]])
        return jsonify(ml_predictor.predict(features))
    except Exception as e: logger.error(f"Error making prediction: {e}"); return jsonify({'error': str(e)}), 500
def cleanup_on_exit(): global shutdown_flag; shutdown_flag = True; logger.info("Shutting down background threads.")

if __name__ == '__main__':
    atexit.register(cleanup_on_exit)
    logger.info("Initializing Ship Route Optimization Backend...")
    ml_initialized = initialize_ml_model()
    weather_initialized = initialize_weather_api()
    start_background_monitoring()
    logger.info(f"Backend ready - ML: {ml_initialized}, Weather: {weather_initialized}")
    app.run(host='0.0.0.0', port=5001, debug=False)

