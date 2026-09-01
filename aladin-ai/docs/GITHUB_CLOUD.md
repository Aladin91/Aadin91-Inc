# Using GitHub as the AladinAI app cloud

AladinAI is designed to stay local-first. GitHub can safely serve as the versioned cloud layer for code, public/static app assets, model packages, configuration, datasets, release bundles and deployment — while mutable private user data stays local by default.

## Recommended architecture

```text
Browser / Personal App
│
├── AladinAI runtime (local browser bundle)
├── IndexedDB / localStorage (mutable personal data)
├── File export/import (portable backups)
│
└── GitHub
    ├── Repository       -> source code + config + small versioned datasets
    ├── GitHub Pages     -> static app hosting
    ├── Releases         -> versioned large-ish bundles/models/assets
    ├── Actions          -> build/test/deploy + trusted automated writes
    └── Artifacts        -> verified CI builds
```

## What GitHub is good for

- Hosting standalone HTML/CSS/JS apps through GitHub Pages.
- Keeping application source and configuration under version control.
- Serving JSON manifests, rule sets, labels, small datasets and model metadata.
- Serving versioned AladinAI bundles and model packages.
- Publishing release assets.
- Running tests/builds with GitHub Actions.
- Creating a reproducible history of every framework/app change.

## What GitHub should NOT be used for directly from a browser

Do not embed a GitHub Personal Access Token in HTML or JavaScript. Any visitor can inspect the page, DevTools, source maps, network requests or browser storage and recover it.

Therefore, a browser-only app should not perform authenticated writes to a GitHub repository using a hard-coded token.

For browser apps, use one of these patterns instead:

1. Store mutable personal data in IndexedDB and periodically export encrypted/JSON backups.
2. Commit versioned shared data from a trusted local development environment.
3. Use GitHub Actions with repository secrets for automated repository writes.
4. If a future app needs multi-user authenticated writes, add a trusted backend/OAuth layer rather than putting credentials in the frontend.

## Loading versioned assets

```js
const cloud = new AladinAI.GitHubCloud({
  owner: 'Aladin91',
  repo: 'Aadin91-Inc',
  branch: 'master',
  basePath: 'aladin-ai/cloud'
});

const manifest = await cloud.loadManifest('manifest.json');
console.log(manifest.version);
```

## Local-first fallback

```js
const store = new AladinAI.LocalFirstRepository({
  namespace: 'my-personal-app',
  cloud
});

const settings = await store.cloudJSON('settings.json', {
  fallbackToLocalKey: 'settings'
});
```

When the cloud fetch succeeds, a local copy can be retained. If the network is unavailable later, the application can use the local copy.

## Releases for framework/model distribution

```js
const releases = new AladinAI.GitHubReleaseAssetResolver({
  owner: 'Aladin91',
  repo: 'Aadin91-Inc'
});

const url = releases.assetUrl('v0.4.0', 'aladin-ai.js');
```

Release assets are useful for immutable versions of:

- `aladin-ai.js`
- local ML model packages
- OCR language packs
- optional ONNX/local vision models
- templates
- application datasets

## Image analysis policy

The v0.4 vision layer does not upload images. Image pixels are processed in memory in the browser. GitHub may be used to distribute reference images or trained model assets, but private user images should remain local unless the app explicitly chooses to upload them.

## Future optional local model runtimes

The deterministic vision layer is intentionally independent from large vision models. Future adapters can add local-only inference through technologies such as WebGPU, WebNN, ONNX Runtime Web or other browser runtimes. Those model files can be downloaded from GitHub Releases once and cached locally.

This preserves the architecture:

```text
GitHub = distribution/versioning
Browser = inference/processing
IndexedDB = private mutable state
Actions = trusted automation
```
