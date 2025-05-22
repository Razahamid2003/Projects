import express from 'express';
import { authenticateToken } from '../middleware/auth.js';
import { waiting } from '../server.js';

const router = express.Router();

router.post('/cancel-match', authenticateToken, (req, res) => {
  for (let i = waiting.length - 1; i >= 0; i--) {
    if (waiting[i].userId === req.userId) {
      waiting.splice(i, 1);
      break;
    }
  }
  res.sendStatus(200);
});

export default router;
