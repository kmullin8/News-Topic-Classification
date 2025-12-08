require('dotenv').config();
const express = require('express');
const path = require('path');
const cors = require('cors');
const { MongoClient } = require('mongodb');

const app = express();
app.use(cors());
app.use(express.json());

// Serve static frontend
app.use(express.static(path.join(__dirname, 'public')));

// --- Mongo connection ---
const uri = process.env.MONGODB_URI;
const collectionName = process.env.MONGODB_COLLECTION || 'articles';

let db, collection;

async function connectMongo() {
  if (db && collection) return { db, collection };

  const client = new MongoClient(uri);
  await client.connect();
  db = client.db(); // DB name is in the URI
  collection = db.collection(collectionName);
  console.log('[MongoDB] Connected to', db.databaseName, 'collection', collectionName);
  return { db, collection };
}

// --- GET /api/topics : distinct main_topic values ---
app.get('/api/topics', async (req, res) => {
  try {
    const { collection } = await connectMongo();
    const topics = await collection.distinct('main_topic');
    topics.sort();
    res.json(topics);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to load topics' });
  }
});

// --- GET /api/articles : paginated + filters ---
app.get('/api/articles', async (req, res) => {
  try {
    const { collection } = await connectMongo();

    const limit = Math.min(parseInt(req.query.limit) || 25, 100);
    const skip = parseInt(req.query.skip) || 0;

    const topic = req.query.topic || '';
    const startDate = req.query.startDate || '';
    const endDate = req.query.endDate || '';
    const highConfidence = req.query.highConfidence === 'true';

    const filter = {};

    if (topic && topic !== 'ALL') {
      filter.main_topic = topic;
    }

    // Date range filter on "published"
    if (startDate || endDate) {
      filter.published = {};
      if (startDate) {
        filter.published.$gte = new Date(startDate);
      }
      if (endDate) {
        // add one day to make end inclusive
        const end = new Date(endDate);
        end.setDate(end.getDate() + 1);
        filter.published.$lt = end;
      }
    }

    // Score filter
    if (highConfidence) {
      filter.topic_score = { $gte: 0.8 };
    }

    const cursor = collection
      .find(filter)
      .sort({ published: -1, topic_score: -1 }) // newest first, then score
      .skip(skip)
      .limit(limit);

    const items = await cursor.toArray();
    const total = await collection.countDocuments(filter);

    res.json({
      total,
      skip,
      limit,
      items
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to load articles' });
  }
});

// --- Fallback: send frontend for any other route (optional SPA-ish behavior) ---
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// --- Start server ---
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
