import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import api from '../utils/api';
import './history-detail.css';

export default function HistoryDetail() {
  const { gameId } = useParams();
  const [user, setUser] = useState(null);
  const [game, setGame] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    // load current user for header
    api.get('/auth/me')
      .then(({ data }) => setUser(data))
      .catch(() => {
        localStorage.removeItem('token');
        navigate('/login');
      });

    // fetch game snapshot
    api.get(`/history/${gameId}`)
      .then(({ data }) => setGame(data))
      .catch(err => {
        console.error('Error fetching game detail', err);
        if (err.response?.status === 401) navigate('/login');
      });
  }, [gameId, navigate]);

  if (!user || !game) {
    return <p style={{ padding: '2rem' }}>Loading…</p>;
  }

  // determine result text and CSS class
  const { myResult, finalGrid } = game;
  const resultText =
    myResult === 'won' ? 'You Won!' :
    myResult === 'lost' ? 'You Lost' :
    'Draw';
  const resultClass = myResult === 'won'
    ? 'won'
    : myResult === 'lost'
      ? 'lost'
      : 'draw';

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
              <Link to="/">Logout</Link>
            </div>
          </div>
        </div>
      </header>

      <main className="snapshot-container">
        <h1 className="snapshot-title">
          Game #{game._id} Result: <span className={`result ${resultClass}`}>{resultText}</span>
        </h1>

        <div className="grid">
          {finalGrid.map((row, r) =>
            row.map((cellColor, c) => (
              <div
                key={`${r}-${c}`}
                className={`cell ${cellColor || ''}`}
              ></div>
            ))
          )}
        </div>

        <Link to="/history" className="btn btn-secondary">Back to History</Link>
      </main>
    </>
  );
}
