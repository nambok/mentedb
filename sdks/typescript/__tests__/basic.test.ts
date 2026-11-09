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

  it('should erase a memory and return a receipt', () => {
    const db = new MenteDB(TEST_DIR);

    const id = db.store({
      content: 'A thought that must leave no trace',
      memoryType: MemoryType.Semantic,
      embedding: Array.from({ length: 128 }, () => Math.random()),
    });

    const receipt = db.erase(id);
    expect(receipt.requested).toBe(id);
    expect(receipt.erased.map((e) => e.id)).toEqual([id]);
    expect(receipt.erased[0].content_sha256).toHaveLength(64);
    expect(receipt.verification?.residual_found).toBe(false);
    expect(JSON.stringify(receipt)).not.toContain('leave no trace');
    expect(() => db.erase(id)).toThrow();

    db.close();
  });

  it('should record provenance, filter by session, and erase a session', () => {
    const db = new MenteDB(TEST_DIR);
    // Distinct, non degenerate vectors: the erase scan looks for the erased
    // vector's bytes, which long zero runs in one hot vectors can fake.
    const vec = (seed: number) => Array.from({ length: 128 }, (_, i) => Math.sin((i + 1) * (seed + 1.37)));
    const near = vec(0);
    const far = vec(1);

    const inSession = db.store({
      content: 'Written during the planning session',
      memoryType: MemoryType.Semantic,
      embedding: far,
      sessionId: 'plan-1',
      source: 'chat',
      sourceRef: 'msg-9',
    });
    const outside = db.store({
      content: 'Written somewhere else entirely',
      memoryType: MemoryType.Semantic,
      embedding: near,
    });

    const memory = db.getMemory(inSession);
    expect(memory.provenance?.sessionId).toBe('plan-1');
    expect(memory.provenance?.source).toBe('chat');
    expect(memory.provenance?.sourceRef).toBe('msg-9');
    expect(db.getMemory(outside).provenance?.source).toBe('api');
    expect(db.memoryIdsInSession('plan-1')).toEqual([inSession]);

    const hits = db.search(near, 5, { sessionId: 'plan-1' });
    expect(hits.map((h) => h.id)).toEqual([inSession]);

    const receipt = db.eraseSession('plan-1');
    expect(receipt.session_id).toBe('plan-1');
    expect(receipt.requested).toEqual([inSession]);
    expect(receipt.verification?.residual_found).toBe(false);
    expect(db.memoryIdsInSession('plan-1')).toEqual([]);
    expect(() => db.getMemory(inSession)).toThrow();
    expect(db.getMemory(outside).content).toBe('Written somewhere else entirely');

    db.close();
  });
});
