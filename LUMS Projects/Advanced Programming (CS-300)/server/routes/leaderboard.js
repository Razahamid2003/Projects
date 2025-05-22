import express from 'express';
import User from '../models/User.js';
import { authenticateToken } from '../middleware/auth.js';

const router = express.Router();
router.use(authenticateToken);

// GET /api/leaderboard
router.get('/', async (req, res) => {
  try {
    const { search } = req.query;
    const filter = search
      ? { username: new RegExp(search, 'i') }
      : {};
    const users = await User.find(filter)
      .sort({ coins: -1 })
      .limit(50)
      .select('username wins losses draws coins');

    res.json(users);
  } catch (err) {
    console.error('Leaderboard error', err);
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;
