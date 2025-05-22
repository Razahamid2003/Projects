import React, { useContext, useEffect, useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import api from '../utils/api';
import { SocketContext } from '../contexts/SocketContext';
import './matchfound.css';

export default function MatchFound() {
  const socket   = useContext(SocketContext);
  const navigate = useNavigate();
  const { state } = useLocation();
  const [user, setUser]             = useState(null);
  const [opponent, setOpponent]     = useState(state?.opponentInfo || null);
  const [gameId, setGameId]         = useState(state?.gameId || null);

  // Load current user for navbar
  useEffect(() => {
    api.get('/auth/me')
      .then(({ data }) => setUser(data))
      .catch(() => {
        localStorage.removeItem('token');
        navigate('/login');
      });
  }, [navigate]);

  // Matchmaking → start_game socket flow
  useEffect(() => {
    // If we don't yet have opponent info, listen for start_game
    if (!opponent) {
      socket.on('start_game', ({ gameId: newGameId, opponentInfo }) => {
        setGameId(newGameId);
        setOpponent(opponentInfo);
      });
    }

    // Once matched, navigate to /newgame/:gameId after a brief display
    if (opponent && gameId) {
      const timer = setTimeout(() => {
        navigate(`/newgame/${gameId}`, { state: { opponentInfo: opponent } });
      }, 2000);
      return () => clearTimeout(timer);
    }

    return () => {
      socket.off('start_game');
    };
  }, [socket, opponent, gameId, navigate]);

  if (!user) {
    return <p style={{ padding: '2rem' }}>Loading…</p>;
  }

  return (
    <>
      <header className="navbar">
        <Link to="/home" className="nav-logo">🎨 ColorGrid</Link>
        <div className="nav-right">
          <span className="coins">💰 {user.coins}</span>
          <div className="profile-dropdown">
            <img
              src={user.profile_picture_url || '/default-avatar.png'}
              alt="Profile"
              className="profile-pic"
            />
            <span className="username">{user.username}</span>
            <div className="dropdown-menu">
              <Link to="/update-profile">Update Profile</Link>
              <button onClick={() => { localStorage.removeItem('token'); navigate('/'); }}>
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="found-container">
        <h1 className="found-title">Match Found!</h1>
        {opponent && (
          <div className="opponent-info">
            <img
              src={opponent.profilePictureUrl || '/default-avatar.png'}
              alt="Opponent Pic"
            />
            <p className="opponent-name">{opponent.username}</p>
          </div>
        )}
        <p className="found-subtitle">
          {opponent ? 'Game is about to start…' : 'Looking for an opponent...'}
        </p>
      </main>
    </>
  );
}
