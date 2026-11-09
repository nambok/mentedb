// A Claude tool loop whose memory tool files live in MenteDB.
//
// Start a server (`mentedb-server --data-dir ./data`), then:
//
//   npm install @anthropic-ai/sdk mentedb
//   MENTEDB_URL=http://localhost:6677 npx tsx memory_tool_agent.ts "Remember that I deploy on Fridays."
//
// MENTEDB_TOKEN is the server JWT (or an mdb_ key with MENTEDB_URL=https://api.mentedb.com).
// MENTEDB_AGENT_ID picks whose /memories tree to use when the token is not bound to an agent.

import Anthropic from '@anthropic-ai/sdk';
import { MemoryTool } from 'mentedb';

const memory = new MemoryTool({
  baseUrl: process.env.MENTEDB_URL,
  token: process.env.MENTEDB_TOKEN,
  agentId: process.env.MENTEDB_AGENT_ID,
});
const client = new Anthropic();

async function main(): Promise<void> {
  const prompt = process.argv[2] ?? 'What do you remember about me?';
  const messages: Anthropic.MessageParam[] = [{ role: 'user', content: prompt }];

  while (true) {
    const response = await client.messages.create({
      model: 'claude-opus-5-5',
      max_tokens: 16000,
      tools: [MemoryTool.TOOL],
      messages,
    });
    messages.push({ role: 'assistant', content: response.content });

    if (response.stop_reason !== 'tool_use') {
      for (const block of response.content) {
        if (block.type === 'text') console.log(block.text);
      }
      if (response.stop_reason === 'refusal') console.error('The request was declined.');
      return;
    }
    // Every memory call in this turn, answered in one user message.
    messages.push({ role: 'user', content: await memory.results(response.content) });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
