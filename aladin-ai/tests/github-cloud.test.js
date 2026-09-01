import assert from 'node:assert/strict';
import { GitHubCloud, GitHubReleaseAssetResolver, assertNoEmbeddedGitHubToken, GitHubCloudPatterns } from '../src/github-cloud.js';

const cloud=new GitHubCloud({owner:'Aladin91',repo:'Aadin91-Inc',branch:'aladin-ai-engine',basePath:'aladin-ai/cloud'});
assert.equal(cloud.rawUrl('manifest.json'),'https://raw.githubusercontent.com/Aladin91/Aadin91-Inc/aladin-ai-engine/aladin-ai/cloud/manifest.json');
const releases=new GitHubReleaseAssetResolver({owner:'Aladin91',repo:'Aadin91-Inc'});
assert.equal(releases.assetUrl('v0.4.0','aladin-ai.js'),'https://github.com/Aladin91/Aadin91-Inc/releases/download/v0.4.0/aladin-ai.js');
assert.equal(assertNoEmbeddedGitHubToken('safe config'),true);
assert.throws(()=>assertNoEmbeddedGitHubToken('github_pat_abcdefghijklmnopqrstuvwxyz1234567890'),/Do not embed GitHub personal access tokens/);
assert.equal(GitHubCloudPatterns.recommended.appHosting,'GitHub Pages');
console.log('AladinAI GitHub cloud helper tests passed.');
