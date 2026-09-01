export class GitHubCloud {
  constructor({owner,repo,branch='master',basePath='',cache='default'}={}){
    if(!owner||!repo)throw new Error('owner and repo are required');Object.assign(this,{owner,repo,branch,basePath:basePath.replace(/^\/+|\/+$/g,''),cache});
  }
  rawUrl(path){const p=[this.basePath,path].filter(Boolean).join('/').replace(/^\/+/, '');return `https://raw.githubusercontent.com/${encodeURIComponent(this.owner)}/${encodeURIComponent(this.repo)}/${encodeURIComponent(this.branch)}/${p.split('/').map(encodeURIComponent).join('/')}`;}
  pagesUrl(path=''){const p=[this.repo,this.basePath,path].filter(Boolean).join('/').replace(/^\/+|\/+$/g,'');return `https://${this.owner.toLowerCase()}.github.io/${p}${path&&path.endsWith('/')?'':' '}`.trim();}
  async fetchText(path,{signal=null,cache=this.cache}={}){const r=await fetch(this.rawUrl(path),{signal,cache});if(!r.ok)throw new Error(`GitHub fetch failed (${r.status}) for ${path}`);return r.text();}
  async fetchJSON(path,opts={}){return JSON.parse(await this.fetchText(path,opts));}
  async fetchBlob(path,{signal=null,cache=this.cache}={}){const r=await fetch(this.rawUrl(path),{signal,cache});if(!r.ok)throw new Error(`GitHub fetch failed (${r.status}) for ${path}`);return r.blob();}
  async loadManifest(path='manifest.json'){const m=await this.fetchJSON(path);if(!m||typeof m!=='object')throw new Error('Invalid manifest');return m;}
}

export class GitHubReleaseAssetResolver {
  constructor({owner,repo}={}){if(!owner||!repo)throw new Error('owner and repo are required');Object.assign(this,{owner,repo});}
  releasePage(tag='latest'){return tag==='latest'?`https://github.com/${this.owner}/${this.repo}/releases/latest`:`https://github.com/${this.owner}/${this.repo}/releases/tag/${tag}`;}
  assetUrl(tag,name){if(!tag||tag==='latest')throw new Error('A concrete release tag is required for direct asset URLs');return `https://github.com/${this.owner}/${this.repo}/releases/download/${encodeURIComponent(tag)}/${encodeURIComponent(name)}`;}
}

export class LocalFirstRepository {
  constructor({namespace='aladinai-app',cloud=null}={}){this.namespace=namespace;this.cloud=cloud;}
  _key(key){return `${this.namespace}:${key}`;}
  saveJSON(key,value){if(typeof localStorage==='undefined')throw new Error('localStorage unavailable');localStorage.setItem(this._key(key),JSON.stringify({savedAt:new Date().toISOString(),value}));return value;}
  loadJSON(key,fallback=null){if(typeof localStorage==='undefined')return fallback;const raw=localStorage.getItem(this._key(key));if(!raw)return fallback;try{return JSON.parse(raw).value;}catch{return fallback;}}
  remove(key){if(typeof localStorage!=='undefined')localStorage.removeItem(this._key(key));}
  async cloudJSON(path,{fallbackToLocalKey=null}={}){try{if(!this.cloud)throw new Error('No GitHubCloud configured');const value=await this.cloud.fetchJSON(path);if(fallbackToLocalKey)this.saveJSON(fallbackToLocalKey,value);return value;}catch(err){if(fallbackToLocalKey){const v=this.loadJSON(fallbackToLocalKey,null);if(v!==null)return v;}throw err;}}
}

export function assertNoEmbeddedGitHubToken(value){
  const text=String(value??'');if(/gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}/.test(text))throw new Error('Do not embed GitHub personal access tokens in browser code. Use GitHub Actions/secrets or a trusted local/server-side process for writes.');return true;
}

export const GitHubCloudPatterns={
  recommended:{appHosting:'GitHub Pages',publicAssets:'repository/raw files or release assets',models:'versioned release assets or repository files',mutablePersonalData:'IndexedDB/local export by default',automatedWrites:'GitHub Actions with repository secrets'},
  warning:'A public browser app cannot safely keep a GitHub write token secret. Treat GitHub as deployment/versioned asset storage unless writes are performed by Actions or another trusted environment.'
};
