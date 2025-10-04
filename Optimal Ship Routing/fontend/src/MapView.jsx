import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, useMapEvents, LayersControl, useMap } from 'react-leaflet';
import L from 'leaflet';

// Helper component to resize the map when the sidebar toggles
function MapResizer({ isPanelOpen }) {
  const map = useMap();
  useEffect(() => {
    // Wait for the sidebar transition to finish before resizing
    const timer = setTimeout(() => {
      map.invalidateSize();
    }, 500); // This duration should match the CSS transition in App.jsx

    return () => clearTimeout(timer);
  }, [isPanelOpen, map]);

  return null;
}

const { BaseLayer, Overlay } = LayersControl;

function ClickToSet({ onSet }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      onSet([lat, lng]);
    }
  });
  return null;
}

const markerIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

export default function MapView({ start, end, setStart, setEnd, pathHistory, gridStep, setGridStep, isPanelOpen }) {
  const windLayer = `https://tile.openweathermap.org/map/wind/{z}/{x}/{y}.png?appid=81da745c6171d7297c8d6943dd0d240e`;
  const clouds = `https://tile.openweathermap.org/map/clouds/{z}/{x}/{y}.png?appid=81da745c6171d7297c8d6943dd0d240e`;

  // Derive current and historical paths from the pathHistory prop
  const currentPath = pathHistory.length > 0 ? pathHistory[pathHistory.length - 1] : null;
  const historicalPaths = pathHistory.length > 1 ? pathHistory.slice(0, -1) : [];

  return (
    <MapContainer
      center={[20.5937, 78.9629]}
      zoom={5}
      style={{ height: "100vh", width: "100%" }}
      scrollWheelZoom={true}
      maxBounds={[[-40, 40], [40, 120]]}
      maxBoundsViscosity={1.0}
    >
      <MapResizer isPanelOpen={isPanelOpen} />

      <LayersControl position="topright">
          <BaseLayer checked name="OSM">
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          </BaseLayer>
          <Overlay checked name="Wind">
              <TileLayer url={windLayer} opacity={0.5} />
          </Overlay>
          <Overlay name="Clouds">
              <TileLayer url={clouds} opacity={0.7} />
          </Overlay>
      </LayersControl>

      <ClickToSet onSet={(pt) => {
        if (!start) setStart(pt);
        else if (!end) setEnd(pt);
        else { setStart(pt); setEnd(null); }
      }} />

      {start && <Marker position={start} icon={markerIcon} />}
      {end && <Marker position={end} icon={markerIcon} />}

      {start && end && (
        <>
          {/* Direct GPS Path */}
          <Polyline positions={[start, end]} pathOptions={{ color: "blue", weight: 2, dashArray: "5, 10" }} />
          
          {/* Historical Paths */}
          {historicalPaths.map((histPath, index) => (
            <Polyline
              key={`hist-${index}`}
              positions={[start, ...histPath, end]}
              pathOptions={{ color: "#6c757d", weight: 3, opacity: 0.7, dashArray: "1, 10" }}
            />
          ))}

          {/* Current Optimized Path */}
          {currentPath && (
            <Polyline 
              key="current"
              positions={[start, ...currentPath, end]}
              pathOptions={{ color: "green", weight: 5 }} 
            />
          )}
        </>
      )}
      
      <div className="leaflet-bottom leaflet-right" style={{ padding: 8, pointerEvents: 'auto' }}>
        <div style={{ background: 'white', padding: 8, borderRadius: 8, boxShadow: '0 2px 6px rgba(0,0,0,0.2)' }}>
          <div><strong>Click map:</strong> start → end</div>
          <div>Grid step (deg): 
            <select value={gridStep} onChange={(e) => setGridStep(parseFloat(e.target.value))}>
              <option value={0.5}>0.5</option>
              <option value={0.25}>0.25</option>
              <option value={0.1}>0.1</option>
            </select>
          </div>
        </div>
      </div>
      
      <div className="leaflet-bottom leaflet-left" style={{ padding: 8, pointerEvents: 'auto' }}>
        <div style={{ background: "white", padding: 6, borderRadius: 6, fontSize: 12, boxShadow: "0 2px 6px rgba(0,0,0,0.3)" }}>
          <div><span style={{color:"blue"}}>▬</span> GPS Route</div>
          {historicalPaths.length > 0 && (
            <div><span style={{color:"#6c757d", letterSpacing: '-1px'}}>▬ ▬</span> Previous Route</div>
          )}
          {currentPath && (
            <div><span style={{color:"green"}}>▬</span> Current Optimized Route</div>
          )}
        </div>
      </div>
    </MapContainer>
  );
}
