import React from 'react';
import { ChakraProvider } from '@chakra-ui/react';
import ElectionForecast from './components/ElectionForecast';

function App() {
  return (
    <ChakraProvider>
      <ElectionForecast />
    </ChakraProvider>
  );
}

export default App;
