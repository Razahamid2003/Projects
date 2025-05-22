import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../utils/api';
import './home.css';

export default function Home() {
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  // Load current user (coins, username, profilePictureUrl)
  useEffect(() => {
    api.get('/auth/me')
      .then(({ data }) => setUser(data))
      .catch(() => {
        localStorage.removeItem('token');
        navigate('/login');
      });
  }, [navigate]);

  if (!user) {
    return <p style={{ padding: '2rem', color: '#fff' }}>Loading…</p>;
  }

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  return (
    <>
      {/* Navbar */}
      <header className="navbar">
        <Link to="/home" className="nav-logo">🎨 ColorGrid</Link>
        <div className="nav-right">
          <span className="coins">
            💰 <span id="coinBalance">{user.coins}</span>
          </span>
          <div className="profile-dropdown">
            <img
              src={user.profilePictureUrl}
              alt="Profile"
              className="profile-pic"
            />
            <span className="username">{user.username}</span>
            <div className="dropdown-menu">
              <Link to="/update-profile">Update Profile</Link>
              <button onClick={handleLogout} style={{ background: 'none', border: 'none', padding: 0, color: '#fff', font: 'inherit', cursor: 'pointer' }}>
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="home-container">
        <h1 className="home-title">Main Dashboard</h1>
        <div className="home-buttons">
          <Link to="/newgame/waiting" className="btn btn-primary">Play</Link>
          <Link to="/leaderboard"     className="btn btn-secondary">Leaderboard</Link>
          <Link to="/history"         className="btn btn-secondary">History</Link>
        </div>
      </main>
    </>
  );
}
