import React, { useContext, useEffect, useState } from 'react';
import { SocketContext } from '../contexts/SocketContext';  // ← fix relative path
import { useNavigate } from 'react-router-dom';
import './waiting.css';

export default function Waiting() {
  const socket   = useContext(SocketContext);
  const navigate = useNavigate();
  const [opponent, setOpponent] = useState(null);

  useEffect(() => {
    // Emit find_match once mount
    socket.emit('find_match');

    socket.on('start_game', ({ gameId, opponentInfo }) => {
      setOpponent(opponentInfo);
      // Pass opponent info along to MatchFound
      navigate('/newgame/found', { state: { gameId, opponentInfo } });
    });

    return () => {
      socket.off('start_game');
      socket.emit('cancel_match');
    };
  }, [socket, navigate]);

  return (
    <div className="waiting-container">
      <h1 className="waiting-title">
        {opponent ? 'Matched!' : 'Waiting for Opponent...'}
      </h1>

      {opponent && (
        <div className="waiting-subtitle">
          <img
            src={opponent.profilePictureUrl || '/default-avatar.png'}
            alt="Opponent"
            className="profile-pic"
            style={{ width: 80, height: 80, borderRadius: '50%', marginBottom: '1rem' }}
          />
          Opponent: {opponent.username}
        </div>
      )}

      {!opponent && (
        <button
          className="btn btn-secondary"
          onClick={() => {
            socket.emit('cancel_match');
            navigate('/home');
          }}
        >
          Cancel
        </button>
      )}
    </div>
  );
}
