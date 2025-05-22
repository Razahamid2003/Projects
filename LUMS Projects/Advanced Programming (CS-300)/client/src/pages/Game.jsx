// client/src/pages/Game.jsx

import React, { useContext, useEffect, useState } from 'react';
import { useParams, useNavigate, Link, useLocation } from 'react-router-dom';
import api from '../utils/api';
import { SocketContext } from '../contexts/SocketContext';
import './gameplay.css';

export default function Game() {
  const { gameId }       = useParams();
  const socket           = useContext(SocketContext);
  const navigate         = useNavigate();
  const { state }        = useLocation();
  const opponentInfo     = state?.opponentInfo;

  const [user, setUser] = useState(null);
  const [grid, setGrid] = useState(
    Array.from({ length: 5 }, () => Array(5).fill(null))
  );
  const [yourColor, setYourColor] = useState(null);
  const [yourTurn, setYourTurn]   = useState(false);
  const [result, setResult]       = useState(null);

  // Load current user for header & player info
  useEffect(() => {
    api.get('/auth/me')
      .then(({ data }) => setUser(data))
      .catch(() => {
        localStorage.removeItem('token');
        navigate('/login');
      });
  }, [navigate]);

  // Join socket room once authenticated
  useEffect(() => {
    if (!user) return; 
    socket.emit('join_game', { gameId });

    const onGameStart = ({ playerColor, opponentColor, firstTurn, opponentProfileUrl }) => {
      setYourColor(playerColor);
      setYourTurn(firstTurn === playerColor);
      // setOpponentInfo(prev => ({ ...prev, profilePictureUrl: opponentProfileUrl }));
    };
    socket.on('game_start', onGameStart);

    const onMove = ({ row, col, color, nextTurn }) => {
      setGrid(g => {
        const copy = g.map(r => [...r]);
        copy[row][col] = color;
        return copy;
      });
      setYourTurn(nextTurn === yourColor);
    };
    socket.on('move_made', onMove);

    const onEnd = ({ result: gameResult }) => {
      setResult(gameResult);
    };
    socket.on('game_end', onEnd);

    return () => {
      socket.off('game_start', onGameStart);
      socket.off('move_made', onMove);
      socket.off('game_end', onEnd);
    };
  }, [socket, gameId, user, yourColor]);

  const handleCellClick = (r, c) => {
    if (!yourTurn || grid[r][c] || result) return;
    socket.emit('make_move', { gameId, row: r, col: c, color: yourColor });
  };

  const handleForfeit = () => {
    socket.emit('forfeit_game', { gameId });
    navigate('/home');
  };

  const handlePlayAgain = () => {
    navigate('/newgame/waiting');
  };

  if (!user) {
    return <p style={{ padding: '2rem', color: '#fff' }}>Loading…</p>;
  }

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
              src={user.profilePictureUrl || 'https://via.placeholder.com/80'}
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

      {/* Game UI */}
      <main className="game-container">
        {/* Players Header */}
        <div className="players-header">
          <div className="player">
            <img
              src={user.profilePictureUrl || 'https://via.placeholder.com/100'}
              alt="You"
              style={{ width: '100px', height: '100px' }}
            />
            <span>{user.username}</span>
          </div>
          <span className="vs">VS</span>
          <div className="player">
            <img
              src={opponentInfo?.profilePictureUrl || 'https://via.placeholder.com/100'}
              alt="Opponent"
              style={{ width: '100px', height: '100px' }}
            />
            <span>{opponentInfo?.username || 'Opponent'}</span>
          </div>
        </div>

        {/* Grid */}
        <div className="grid">
          {grid.map((row, r) =>
            row.map((cellColor, c) => (
              <div
                key={`${r}-${c}`}
                className={`cell ${cellColor || ''}`}
                onClick={() => handleCellClick(r, c)}
              />
            ))
          )}
        </div>

        {/* Status and Controls */}
        <div className="status-area">
          <p id="status">
            Status: <span>
              {result
                ? result === 'draw'
                  ? 'Draw'
                  : result === 'won'
                    ? 'You Won!'
                    : 'You Lost'
                : yourTurn
                  ? 'Your Turn'
                  : "Opponent's Turn"
              }
            </span>
          </p>

          <button
            id="forfeitBtn"
            className="btn btn-secondary"
            onClick={handleForfeit}
            hidden={!!result}
          >
            Forfeit
          </button>

          <button
            id="playAgainBtn"
            className={`btn btn-primary${result ? '' : ' hidden'}`}
            onClick={handlePlayAgain}
          >
            Play Again
          </button>
        </div>
      </main>
    </>
  );
}
