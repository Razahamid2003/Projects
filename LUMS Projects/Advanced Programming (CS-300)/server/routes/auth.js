import express from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import User from '../models/User.js';
import { authenticateToken } from '../middleware/auth.js';

// Use your env var, or default to a hard‑coded secret
const SECRET = process.env.JWT_SECRET || 'supersecret';

const router = express.Router();

// POST /api/auth/signup
router.post('/signup', async (req, res) => {
  try {
    const { username, password, profilePictureUrl } = req.body;
    const hash = await bcrypt.hash(password, 10);
    const user = await User.create({
      username,
      passwordHash: hash,
      profile_picture_url: profilePictureUrl,
      coins: 1000,
      wins: 0,
      losses: 0,
      draws: 0,
    });
    const token = jwt.sign({ id: user._id }, SECRET);
    res.json({ token });
  } catch (err) {
    console.error('Signup error', err);
    res.status(500).json({ message: 'Server error' });
  }
});

// POST /api/auth/login
router.post('/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    const user = await User.findOne({ username });
    if (!user) return res.status(401).json({ message: 'Invalid credentials' });

    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok) return res.status(401).json({ message: 'Invalid credentials' });

    const token = jwt.sign({ id: user._id }, SECRET);
    res.json({ token });
  } catch (err) {
    console.error('Login error', err);
    res.status(500).json({ message: 'Server error' });
  }
});

// GET /api/auth/me
router.get('/me', authenticateToken, async (req, res) => {
  try {
    const u = await User.findById(req.user.id)
      .select('username profile_picture_url coins wins losses draws');
    res.json(u);
  } catch (err) {
    console.error('/me error', err);
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;
