import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const USAMap = ({ pollingData }) => {
  const [mapData, setMapData] = useState(null);

  useEffect(() => {
    // Fetch the GeoJSON data for USA states
    const fetchMapData = async () => {
      try {
        const response = await fetch('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json');
        const data = await response.json();
        setMapData(data);
      } catch (error) {
        console.error('Error fetching map data:', error);
      }
    };

    fetchMapData();
  }, []);

  const onEachFeature = (feature, layer) => {
    // Set the state color based on polling data
    const statePollingData = pollingData.find(
      (state) => state.name === feature.properties.name
    );
    const color = statePollingData ? statePollingData.color : 'white';

    layer.setStyle({
      fillColor: color,
      weight: 1,
      color: 'black',
      fillOpacity: 0.7,
    });
  };

  return (
    <MapContainer center={[37.8, -96]} zoom={4} scrollWheelZoom={false} style={{ height: '500px', width: '100%' }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {mapData && <GeoJSON data={mapData} onEachFeature={onEachFeature} />}
    </MapContainer>
  );
};

export default USAMap;
