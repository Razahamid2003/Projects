import jwt from 'jsonwebtoken';
import dotenv from 'dotenv';
dotenv.config();

export function authenticateToken(req, res, next) {
  const authHeader = req.headers.authorization || '';
  const token = authHeader.replace(/^Bearer\s+/, '');
  if (!token) {
    return res.status(401).json({ message: 'Missing auth token' });
  }
  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET || 'supersecret');
    req.user = { id: payload.id };
    next();
  } catch {
    return res.status(401).json({ message: 'Invalid or expired token' });
  }
}
