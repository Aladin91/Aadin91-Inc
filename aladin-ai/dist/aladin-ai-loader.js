import AladinAI from '../src/index.js';

globalThis.AladinAI = AladinAI;
globalThis.dispatchEvent?.(new CustomEvent('aladinai:ready', { detail: { version: AladinAI.version } }));

export default AladinAI;
export * from '../src/index.js';
