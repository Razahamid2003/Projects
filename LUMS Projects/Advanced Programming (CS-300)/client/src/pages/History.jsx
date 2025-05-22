import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../utils/api';
import './history.css';

export default function History() {
  const [user, setUser] = useState(null);
  const [games, setGames] = useState([]);
  const navigate = useNavigate();

  // Load current user for header
  useEffect(() => {
    api.get('/auth/me')
      .then(({ data }) => setUser(data))
      .catch(() => {
        localStorage.removeItem('token');
        navigate('/login');
      });
  }, [navigate]);

  // Fetch game history
  useEffect(() => {
    api.get('/history')
      .then(({ data }) => setGames(data))
      .catch(err => {
        console.error('Error fetching history', err);
      });
  }, []);

  if (!user) {
    return <p style={{ padding: '2rem' }}>Loading…</p>;
  }

  return (
    <>
      <header className="navbar">
        <Link to="/home" className="nav-logo">🎨 ColorGrid</Link>
        <div className="nav-right">
          <span className="coins">💰 <span id="coinBalance">{user.coins}</span></span>
          <div className="profile-dropdown">
            <img
              src={user.profile_picture_url || 'https://via.placeholder.com/80'}
              alt="Profile"
              className="profile-pic"
            />
            <span className="username">{user.username}</span>
            <div className="dropdown-menu">
              <Link to="/update-profile">Update Profile</Link>
              <button
                onClick={() => {
                  localStorage.removeItem('token');
                  navigate('/');
                }}
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="history-container">
        <h1 className="history-title">Your Game History</h1>
        <ul className="history-list">
          {games.map(game => (
            <li key={game._id}>
              <Link to={`/history/${game._id}`}>
                Game #{game._id} — {game.opponent.username} — 
                {game.result === 'win' ? 'Won' : game.result === 'loss' ? 'Lost' : 'Draw'}
              </Link>
            </li>
          ))}
        </ul>
      </main>
    </>
  );
}
