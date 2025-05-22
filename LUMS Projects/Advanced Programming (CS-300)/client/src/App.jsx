// client/src/App.jsx

import { useEffect } from 'react';
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useLocation
} from 'react-router-dom';
import { SocketContext, socket } from './contexts/SocketContext';

import Welcome       from './pages/Welcome';
import Login         from './pages/Login';
import Signup        from './pages/Signup';
import Home          from './pages/Home';
import Waiting       from './pages/Waiting';
import MatchFound    from './pages/MatchFound';
import Game          from './pages/Game';
import History       from './pages/History';
import HistoryDetail from './pages/HistoryDetail';
import UpdateProfile from './pages/UpdateProfile';
import Leaderboard   from './pages/Leaderboard';

function AppRoutes() {
  const { pathname } = useLocation();
  const token = localStorage.getItem('token');

  const isPublicPage =
    pathname === '/' ||
    pathname === '/login' ||
    pathname === '/signup';

  return (
    <>
      <Routes>
        {/* PUBLIC */}
        <Route
          path="/"
          element={token ? <Navigate to="/home" /> : <Welcome />}
        />
        <Route
          path="/login"
          element={token ? <Navigate to="/home" /> : <Login />}
        />
        <Route
          path="/signup"
          element={token ? <Navigate to="/home" /> : <Signup />}
        />

        {/* PROTECTED */}
        <Route
          path="/home"
          element={token ? <Home /> : <Navigate to="/login" />}
        />
        <Route
          path="/newgame/waiting"
          element={token ? <Waiting /> : <Navigate to="/login" />}
        />
        <Route
          path="/newgame/found"
          element={token ? <MatchFound /> : <Navigate to="/login" />}
        />
        <Route
          path="/newgame/:gameId"
          element={token ? <Game /> : <Navigate to="/login" />}
        />
        <Route
          path="/history"
          element={token ? <History /> : <Navigate to="/login" />}
        />
        <Route
          path="/history/:gameId"
          element={token ? <HistoryDetail /> : <Navigate to="/login" />}
        />
        <Route
          path="/update-profile"
          element={token ? <UpdateProfile /> : <Navigate to="/login" />}
        />
        <Route
          path="/leaderboard"
          element={token ? <Leaderboard /> : <Navigate to="/login" />}
        />

        {/* FALLBACK */}
        <Route
          path="*"
          element={<Navigate to={token ? '/home' : '/'} />}
        />
      </Routes>
    </>
  );
}

export default function App() {
  useEffect(() => {
    const handleUnload = () => {
      navigator.sendBeacon(
        'http://localhost:8000/api/cancel-match',
        ''
      );
    };
    window.addEventListener('beforeunload', handleUnload);
    return () => {
      window.removeEventListener('beforeunload', handleUnload);
    };
  }, []);

  return (
    <SocketContext.Provider value={socket}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </SocketContext.Provider>
  );
}
