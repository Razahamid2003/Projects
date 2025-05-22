import express from 'express';
import bcrypt from 'bcryptjs';
import User from '../models/User.js';
import { authenticateToken } from '../middleware/auth.js';

const router = express.Router();
router.use(authenticateToken);

// PATCH /api/users/me
router.patch('/me', async (req, res) => {
  try {
    const updates = {};
    if (req.body.username) updates.username = req.body.username;
    if (req.body.profile_picture_url !== undefined) {
      updates.profile_picture_url = req.body.profile_picture_url;
    }
    if (req.body.password) {
      updates.passwordHash = await bcrypt.hash(req.body.password, 10);
    }

    const user = await User.findByIdAndUpdate(
      req.user.id,
      updates,
      { new: true }
    ).select('username profile_picture_url coins wins losses draws');

    res.json(user);
  } catch (err) {
    console.error('Update profile error', err);
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;
