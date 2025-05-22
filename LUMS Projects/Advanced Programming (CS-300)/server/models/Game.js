import mongoose from 'mongoose';

const gameSchema = new mongoose.Schema({
  player1_id: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  player2_id: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  player1_color: { type: String, required: true },
  player2_color: { type: String, required: true },
  final_grid: { type: [[String]], required: true },
  result: { type: String, enum: ['won','lost','draw'], required: true },
  winner_id: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }
}, { timestamps: true });

export default mongoose.model('Game', gameSchema);
