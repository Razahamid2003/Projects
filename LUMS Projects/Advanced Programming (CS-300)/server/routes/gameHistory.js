import express from 'express';
import Game from '../models/Game.js';
import { authenticateToken } from '../middleware/auth.js';

const router = express.Router();
router.use(authenticateToken);

// GET /api/history
router.get('/', async (req, res) => {
  try {
    const userId = req.user.id;
    const games = await Game.find({
      $or: [{ player1_id: userId }, { player2_id: userId }]
    })
    .populate('player1_id player2_id', 'username')
    .sort({ createdAt: -1 });

    const list = games.map(g => {
      const isP1 = g.player1_id._id.toString() === userId;
      const opponent = isP1 ? g.player2_id : g.player1_id;
      const outcome =
        g.winner_id == null                             ? 'draw' :
        g.winner_id.toString() === userId               ? 'win'  :
                                                          'loss';
      return {
        _id: g._id,
        opponent: { username: opponent.username, id: opponent._id },
        result: outcome
      };
    });

    res.json(list);
  } catch (err) {
    console.error('History list error', err);
    res.status(500).json({ message: 'Server error' });
  }
});

// GET /api/history/:gameId
router.get('/:gameId', async (req, res) => {
  try {
    const { gameId } = req.params;
    const userId = req.user.id;
    const g = await Game.findById(gameId)
      .populate('player1_id player2_id', 'username');

    if (!g) return res.status(404).json({ message: 'Game not found' });
    if (![g.player1_id._id.toString(), g.player2_id._id.toString()].includes(userId)) {
      return res.status(403).json({ message: 'Forbidden' });
    }

    const myResult =
      g.winner_id == null                              ? 'draw' :
      g.winner_id.toString() === userId                ? 'won'  :
                                                         'lost';

    res.json({
      _id: g._id,
      myResult,
      finalGrid: g.final_grid
    });
  } catch (err) {
    console.error('History detail error', err);
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;
