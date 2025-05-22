import { config } from 'dotenv';
config({ path: './config.env' });

import http from 'http';
import mongoose from 'mongoose';
import { Server } from 'socket.io';
import jwt from 'jsonwebtoken';

import { app } from './app.js';
import User from './models/User.js';
import Game from './models/Game.js';
import { maxAreaOfIsland } from './utils/maxAreaOfIsland.js';

const JWT_SECRET = process.env.JWT_SECRET || 'supersecret';

mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log('🗄️  Connected to MongoDB'))
  .catch(err => console.error('❌ MongoDB connection error:', err));

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: '*', methods: ['GET','POST'], credentials: false },
});

// Authenticate socket & attach userId/username
io.use(async (socket, next) => {
  try {
    const token = socket.handshake.auth.token;
    const payload = jwt.verify(token, JWT_SECRET);
    socket.userId = payload.id;
    const user = await User.findById(socket.userId);
    if (!user) throw new Error('User not found');
    socket.username = user.username;
    next();
  } catch {
    next(new Error('Authentication error'));
  }
});

export const waiting = [];
export const games   = {};

io.on('connection', socket => {
  console.log(`USER CONNECTED: ${socket.userId} (${socket.username})`);

  // Remove any stale socket for the same user
  for (const [_, s] of io.of('/').sockets) {
    if (s.userId === socket.userId && s.id !== socket.id) {
      console.log(`Dropping stale socket ${s.id} for user ${s.userId}`);
      s.disconnect(true);
      const idx = waiting.indexOf(s);
      if (idx !== -1) waiting.splice(idx, 1);
    }
  }

  // Matchmaking (no coin‐balance gate)
  socket.on('find_match', () => {
    // Ensure this socket is not already in waiting
    for (let i = waiting.length - 1; i >= 0; i--) {
      if (waiting[i].userId === socket.userId) waiting.splice(i, 1);
    }
    waiting.push(socket);

    if (waiting.length >= 2) {
      const [s1, s2] = waiting.splice(0, 2);
      const gameId = `${s1.userId}-${s2.userId}`;
      const g = {
        p1: s1.userId,
        p2: s2.userId,
        player1Color: 'red',
        player2Color: 'blue',
        firstTurn: 'red',
        grid: Array(5).fill(null).map(() => Array(5).fill(null))
      };
      games[gameId] = g;

      s1.join(gameId);
      s2.join(gameId);

      s1.emit('start_game', {
        gameId,
        opponentInfo: { id: s2.userId, username: s2.username }
      });
      s2.emit('start_game', {
        gameId,
        opponentInfo: { id: s1.userId, username: s1.username }
      });
    }
  });

  // Join existing game
  socket.on('join_game', ({ gameId }) => {
    const g = games[gameId];
    if (!g) return;
    socket.join(gameId);
    const isP1 = socket.userId === g.p1;
    socket.emit('game_start', {
      playerColor:   isP1 ? g.player1Color : g.player2Color,
      opponentColor: isP1 ? g.player2Color : g.player1Color,
      firstTurn:     g.firstTurn,
    });
  });

  // Handle moves
  socket.on('make_move', async ({ gameId, row, col, color }) => {
    const g = games[gameId];
    if (!g) return console.warn(`make_move: no game for ID ${gameId}`);

    // Broadcast move
    const nextTurn = color === g.player1Color
      ? g.player2Color
      : g.player1Color;
    io.to(gameId).emit('move_made', { row, col, color, nextTurn });

    // Update grid
    g.grid[row][col] = color;

    // If not full, wait for more moves
    if (!g.grid.every(r => r.every(c => c !== null))) return;

    // Determine result
    const area1 = maxAreaOfIsland(g.grid.map(r => r.map(c => c===g.player1Color?1:0)));
    const area2 = maxAreaOfIsland(g.grid.map(r => r.map(c => c===g.player2Color?1:0)));
    let result, winnerId;
    if (area1 > area2)      { result = 'won';  winnerId = g.p1; }
    else if (area2 > area1) { result = 'lost'; winnerId = g.p2; }
    else                    { result = 'draw'; winnerId = null; }

    // Save game record
    try {
      await Game.create({
        player1_id:    g.p1,
        player2_id:    g.p2,
        player1_color: g.player1Color,
        player2_color: g.player2Color,
        final_grid:    g.grid,
        result,
        winner_id:     winnerId
      });
    } catch (err) {
      console.error('Error saving game:', err);
    }

    // Update coins & stats (clamp loser to >=0)
    if (result === 'draw') {
      try {
        const [p1, p2] = await Promise.all([
          User.findById(g.p1),
          User.findById(g.p2)
        ]);
        p1.draws += 1;
        p2.draws += 1;
        await Promise.all([p1.save(), p2.save()]);
      } catch (err) {
        console.error('Error recording draws:', err);
      }
    } else {
      const loserId = winnerId === g.p1 ? g.p2 : g.p1;
      try {
        const [winner, loser] = await Promise.all([
          User.findById(winnerId),
          User.findById(loserId)
        ]);
        // Winner gains 200
        winner.coins += 200;
        // Loser loses 200, but not below 0
        loser.coins = Math.max(0, loser.coins - 200);
        // Stats
        winner.wins   += 1;
        loser.losses  += 1;
        await Promise.all([winner.save(), loser.save()]);
      } catch (err) {
        console.error('Error updating coins & stats:', err);
      }
    }

    io.to(gameId).emit('game_end', { result });
    delete games[gameId];
  });

  // Handle forfeit
  socket.on('forfeit_game', async ({ gameId }) => {
    const g = games[gameId];
    if (!g) return;
    const loserId  = socket.userId;
    const winnerId = loserId === g.p1 ? g.p2 : g.p1;

    // Save forfeit
    try {
      await Game.create({
        player1_id:    g.p1,
        player2_id:    g.p2,
        player1_color: g.player1Color,
        player2_color: g.player2Color,
        final_grid:    g.grid,
        result:        'lost',
        winner_id:     winnerId,
      });
    } catch (err) {
      console.error('Error saving forfeited game:', err);
    }

    // Update coins & stats
    try {
      const [winner, loser] = await Promise.all([
        User.findById(winnerId),
        User.findById(loserId),
      ]);
      winner.coins += 200;
      loser.coins = Math.max(0, loser.coins - 200);
      winner.wins   += 1;
      loser.losses  += 1;
      await Promise.all([winner.save(), loser.save()]);
    } catch (err) {
      console.error('Error updating coins on forfeit:', err);
    }

    socket.emit('game_end', { result: 'lost' });
    socket.to(gameId).emit('game_end', { result: 'won' });
    delete games[gameId];
  });

  // Cancel before matched
  socket.on('cancel_match', () => {
    const idx = waiting.indexOf(socket);
    if (idx !== -1) waiting.splice(idx, 1);
  });

  // Cleanup on disconnect
  socket.on('disconnect', () => {
    const idx = waiting.indexOf(socket);
    if (idx !== -1) waiting.splice(idx, 1);
    for (const id in games) {
      if (games[id].p1 === socket.userId || games[id].p2 === socket.userId) {
        delete games[id];
      }
    }
  });
});

const PORT = process.env.PORT || 8000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
