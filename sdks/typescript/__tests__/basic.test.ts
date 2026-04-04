import { MenteDB, MemoryType, EdgeType } from '../npm';

describe('MenteDB TypeScript SDK', () => {
  const TEST_DIR = './test-data';

  afterEach(() => {
    // Clean up test data directory
    const fs = require('fs');
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it('should open a database, store, recall, and close', () => {
    const db = new MenteDB(TEST_DIR);

    const id = db.store({
      content: 'TypeScript SDK integration test',
      memoryType: MemoryType.Episodic,
      embedding: Array.from({ length: 128 }, () => Math.random()),
      tags: ['test', 'sdk'],
    });

    expect(typeof id).toBe('string');
    expect(id.length).toBe(36); // UUID format

    db.close();
  });

  it('should create edges between memories', () => {
    const db = new MenteDB(TEST_DIR);

    const a = db.store({
      content: 'Cause event',
      memoryType: MemoryType.Episodic,
      embedding: Array.from({ length: 128 }, () => Math.random()),
    });

    const b = db.store({
      content: 'Effect event',
      memoryType: MemoryType.Episodic,
      embedding: Array.from({ length: 128 }, () => Math.random()),
    });

    expect(() => db.relate(a, b, EdgeType.Caused, 0.9)).not.toThrow();

    db.close();
  });

  it('should forget a memory', () => {
    const db = new MenteDB(TEST_DIR);

    const id = db.store({
      content: 'Temporary thought',
      memoryType: MemoryType.Semantic,
      embedding: Array.from({ length: 128 }, () => Math.random()),
    });

    expect(() => db.forget(id)).not.toThrow();

    db.close();
  });
});
