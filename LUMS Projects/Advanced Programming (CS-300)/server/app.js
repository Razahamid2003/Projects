// server/app.js

import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import dotenv from 'dotenv';

// Route handlers
import authRoutes        from './routes/auth.js';
import historyRoutes     from './routes/gameHistory.js';
import leaderboardRoutes from './routes/leaderboard.js';
import matchRoutes       from './routes/match.js';
import userRoutes        from './routes/userRoutes.js';

dotenv.config({ path: './config.env' });

export const app = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// 1) Public auth endpoints
app.use('/api/auth', authRoutes);

// 2) Protected resource endpoints
app.use('/api/history',     historyRoutes);
app.use('/api/leaderboard', leaderboardRoutes);
app.use('/api/match',       matchRoutes);
app.use('/api/users',       userRoutes);