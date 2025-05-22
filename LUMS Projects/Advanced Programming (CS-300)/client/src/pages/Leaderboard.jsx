import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../utils/api';
import './leaderboard.css';

export default function Leaderboard() {
  const [user, setUser]       = useState(null);
  const [players, setPlayers] = useState([]);
  const [search, setSearch]   = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    api.get('/auth/me')
      .then(({ data }) => setUser(data))
      .catch(() => {
        localStorage.removeItem('token');
        navigate('/login');
      });
  }, [navigate]);

  useEffect(() => {
    const endpoint = search
      ? `/leaderboard?search=${encodeURIComponent(search)}`
      : '/leaderboard';

    api.get(endpoint)
      .then(({ data }) => setPlayers(data))
      .catch(err => console.error('Error fetching leaderboard', err));
  }, [search]);

  if (!user) return <p style={{ padding: '2rem' }}>Loading…</p>;

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

      <main className="board-container">
        <h1 className="board-title">Leaderboard</h1>
        <form onSubmit={e => e.preventDefault()}>
          <input
            type="text"
            placeholder="Search by username…"
            className="search-box"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </form>

        <table className="board-table">
          <thead>
            <tr>
              <th>Username</th>
              <th>Wins</th>
              <th>Losses</th>
              <th>Draws</th>
              <th>Coins</th>
            </tr>
          </thead>
          <tbody>
            {players.map(p => (
              <tr key={p.username}>
                <td>{p.username}</td>
                <td>{p.wins ?? 0}</td>
                <td>{p.losses ?? 0}</td>
                <td>{p.draws ?? 0}</td>
                <td>{p.coins ?? 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </main>
    </>
  );
}
