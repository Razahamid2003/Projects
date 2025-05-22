import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './signup.css';

export default function Signup() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [profilePictureUrl, setProfilePictureUrl] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async e => {
    e.preventDefault();
    try {
      const res = await axios.post(
        'http://localhost:8000/api/auth/signup',
        { username, password, profilePictureUrl }
      );
      localStorage.setItem('token', res.data.token);
      navigate('/home');
    } catch (err) {
      setError(err.response?.data?.message || 'Signup failed');
    }
  };

  return (
    <div className="auth-container">
      <h2 className="auth-title">Sign Up</h2>
      <form className="auth-form" onSubmit={handleSubmit}>
        <label>Username</label>
        <input
          type="text" value={username}
          onChange={e => setUsername(e.target.value)}
          required
        />

        <label>Password</label>
        <input
          type="password" value={password}
          onChange={e => setPassword(e.target.value)}
          required
        />

        <label>Profile Picture URL (optional)</label>
        <input
          type="url" value={profilePictureUrl}
          onChange={e => setProfilePictureUrl(e.target.value)}
        />

        <button type="submit" className="btn btn-primary">
          Sign Up
        </button>
      </form>

      <div className="auth-footer">
        <span>Already have an account? </span>
        <Link to="/login">Login</Link>
      </div>

      {error && <p style={{ color: 'salmon', marginTop: '1rem' }}>{error}</p>}
    </div>
  );
}
