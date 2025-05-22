import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../utils/api';
import './update-profile.css';

export default function UpdateProfile() {
  const [user, setUser] = useState(null);
  const [username, setUsername] = useState('');
  const [profilePictureUrl, setProfilePictureUrl] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  // Load current user
  useEffect(() => {
    api.get('/auth/me')
      .then(({ data }) => {
        setUser(data);
        setUsername(data.username);
        setProfilePictureUrl(data.profile_picture_url || '');
      })
      .catch(() => {
        localStorage.removeItem('token');
        navigate('/login');
      });
  }, [navigate]);

  // Submit updated profile
  const handleSubmit = async e => {
    e.preventDefault();
    try {
      const payload = { username, profile_picture_url: profilePictureUrl };
      if (password) payload.password = password;

      // hit users route — adjust if backend puts it elsewhere
      await api.patch('/users/me', payload);
      navigate('/home');
    } catch (err) {
      console.error('Profile update failed:', err);
      alert(err.response?.data?.message || 'Update failed');
    }
  };

  if (!user) {
    return <p style={{ padding: '2rem' }}>Loading…</p>;
  }

  return (
    <>
      <header className="navbar">
        {/* …header… */}
        <Link to="/home" className="nav-logo">🎨 ColorGrid</Link>
        <div className="nav-right">
          <span className="coins">💰 {user.coins}</span>
          {/* profile dropdown … */}
        </div>
      </header>

      <main className="update-container">
        <h1 className="update-title">Update Profile</h1>
        <form className="update-form" onSubmit={handleSubmit}>
          <label htmlFor="username">Username</label>
          <input
            id="username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            required
          />

          <label htmlFor="password">New Password</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
          />

          <label htmlFor="profilePic">Profile Picture URL</label>
          <input
            id="profilePic"
            type="url"
            value={profilePictureUrl}
            onChange={e => setProfilePictureUrl(e.target.value)}
          />

          <button type="submit" className="btn btn-primary">
            Save Changes
          </button>
        </form>
      </main>
    </>
  );
}
