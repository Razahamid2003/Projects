// client/src/contexts/SocketContext.js

import { createContext } from 'react';
import { io } from 'socket.io-client';

export const socket = io('http://localhost:8000', {
  transports: ['websocket'],
  auth: { token: localStorage.getItem('token') },
  reconnection: false,
});

export const SocketContext = createContext(socket);
