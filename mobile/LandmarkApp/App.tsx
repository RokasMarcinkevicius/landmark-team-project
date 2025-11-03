import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Provider as PaperProvider } from 'react-native-paper';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import CameraScreen from './src/screens/CameraScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import { enableScreens } from 'react-native-screens';

enableScreens();
const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <PaperProvider>
        <NavigationContainer>
          <Stack.Navigator
            initialRouteName="Camera"
            screenOptions={{
              headerStyle: {
                backgroundColor: '#6200ea',
              },
              headerTintColor: '#fff',
              headerTitleStyle: {
                fontWeight: 'bold',
              },
            }}
          >
            <Stack.Screen
              name="Camera"
              component={CameraScreen}
              options={{ title: 'Landmark Identifier' }}
            />
            <Stack.Screen
              name="Results"
              component={ResultsScreen}
              options={{ title: 'Prediction Results' }}
            />
            <Stack.Screen
              name="History"
              component={HistoryScreen}
              options={{ title: 'Prediction History' }}
            />
          </Stack.Navigator>
        </NavigationContainer>
      </PaperProvider>
    </SafeAreaProvider>
  );
}
