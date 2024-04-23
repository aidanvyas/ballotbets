import React from 'react';
import { Box, Heading, Text } from '@chakra-ui/react';

const ElectionForecast = () => {
  return (
    <Box p={5} shadow="md" borderWidth="1px">
      <Heading fontSize="xl">Election Forecast 2024</Heading>
      <Text mt={4}>
        This is a placeholder for the election forecast data visualization.
      </Text>
    </Box>
  );
};

export default ElectionForecast;
