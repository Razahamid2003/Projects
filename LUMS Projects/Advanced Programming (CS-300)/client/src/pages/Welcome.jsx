import React from 'react';
import { Link } from 'react-router-dom';
import './welcome.css';

export default function Welcome() {
  return (
    <main className="welcome-container">
      <h1 className="welcome-title">Welcome to ColorGrid</h1>
      <p className="welcome-subtitle">A real‑time, multiplayer grid conquest game.</p>
      <div className="welcome-buttons">
        <Link to="/login" className="btn btn-primary">Login</Link>
        <Link to="/signup" className="btn btn-secondary">Sign Up</Link>
      </div>
    </main>
  );
}
