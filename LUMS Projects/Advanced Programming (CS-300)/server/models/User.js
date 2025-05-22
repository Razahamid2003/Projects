import mongoose from 'mongoose';

const userSchema = new mongoose.Schema({
  username:           { type: String, required: true, unique: true },
  passwordHash:       { type: String, required: true },
  profilePictureUrl:  { type: String },
  coins:              { type: Number, default: 1000 },
  wins:               { type: Number, default: 0 },       // ← new
  losses:             { type: Number, default: 0 },       // ← new
  draws:              { type: Number, default: 0 }        // ← new
});

export default mongoose.model('User', userSchema);
