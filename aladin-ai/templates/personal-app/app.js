const output=document.getElementById('out');
const log=value=>{output.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);};

document.getElementById('aiStatus').innerHTML=`<span class="ok">ready v${AladinAI.version}</span>`;
const local=new AladinAI.LocalFirstRepository({namespace:'aladinai-personal-app'});

saveBtn.addEventListener('click',()=>{
  const checkpoint={savedAt:new Date().toISOString(),engineVersion:AladinAI.version,note:'Personal app checkpoint'};
  local.saveJSON('checkpoint',checkpoint);log(checkpoint);
});

cloudBtn.addEventListener('click',async()=>{
  cloudStatus.textContent='checking…';
  try{
    const config=await fetch('config.json').then(r=>r.json());
    const cloud=new AladinAI.GitHubCloud(config.github);
    const manifest=await cloud.loadManifest(config.manifestPath);
    cloudStatus.innerHTML=`<span class="ok">${manifest.version}</span>`;log(manifest);
  }catch(err){cloudStatus.textContent='offline/unavailable';log({message:err.message,localCheckpoint:local.loadJSON('checkpoint',null)});}
});

if('serviceWorker'in navigator)navigator.serviceWorker.register('sw.js').catch(()=>{});
