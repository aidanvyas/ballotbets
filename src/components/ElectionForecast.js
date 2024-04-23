import React, { useState, useEffect } from 'react';
import { Box, Heading, Text, List, ListItem } from '@chakra-ui/react';

const ElectionForecast = () => {
  const [forecastData, setForecastData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/api/forecast');
        const data = await response.json();
        setForecastData(data);
      } catch (error) {
        console.error('Error fetching election forecast data:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <Box p={5} shadow="md" borderWidth="1px">
      <Heading fontSize="xl">Election Forecast 2024</Heading>
      {forecastData ? (
        <>
          <Text mt={4}>National Forecast</Text>
          <Text>Chance to Win: {forecastData.national.chanceToWin * 100}%</Text>
          <Text>Total Electoral Votes: {forecastData.national.totalElectoralVotes}</Text>
          <Text mt={4}>State Forecasts</Text>
          <List spacing={3}>
            {forecastData.states.map((state, index) => (
              <ListItem key={index}>
                {state.name}: {state.chanceToWin * 100}% chance to win, {state.electoralVotes} electoral votes
              </ListItem>
            ))}
          </List>
        </>
      ) : (
        <Text mt={4}>Loading election forecast data...</Text>
      )}
    </Box>
  );
};

export default ElectionForecast;
