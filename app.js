import { firebaseConfig } from './firebase-config.js';

const $ = (id) => document.getElementById(id);
const storeKey = 'spin_club_players_v1';
let players = [];
let firebaseReady = false;
let db = null;

function uid(){ return crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2); }
function toast(msg){ const t=$('toast'); t.textContent=msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),2200); }
function saveLocal(){ localStorage.setItem(storeKey, JSON.stringify(players)); }
function loadLocal(){ try{ players = JSON.parse(localStorage.getItem(storeKey)) || []; } catch { players = []; } }
function hasFirebaseConfig(){ return firebaseConfig && firebaseConfig.apiKey && !firebaseConfig.apiKey.includes('PASTE_'); }
function setSync(text){ const el=document.querySelector('.brand small'); if(el) el.textContent=text; }

async function bootFirebase(){
  if(!hasFirebaseConfig()){ setSync('Local Mode'); return; }
  try{
    const appMod = await import('https://www.gstatic.com/firebasejs/10.12.5/firebase-app.js');
    const fs = await import('https://www.gstatic.com/firebasejs/10.12.5/firebase-firestore.js');
    const app = appMod.initializeApp(firebaseConfig);
    db = fs.getFirestore(app);
    window.fs = fs;
    firebaseReady = true;
    setSync('Firebase Live');
    fs.onSnapshot(collectionRef(), snap => {
      players = snap.docs.map(d => ({ id:d.id, ...d.data() }));
      saveLocal();
      render();
    });
  }catch(err){ console.error(err); setSync('Local Fallback'); }
}
function collectionRef(){ return window.fs.collection(db, 'clubs', 'spin-club', 'players'); }
async function upsertPlayer(player){
  if(firebaseReady){
    const fs = window.fs;
    await fs.setDoc(fs.doc(collectionRef(), player.id), player, { merge:true });
  }else{
    const idx = players.findIndex(p => p.id === player.id);
    if(idx >= 0) players[idx] = player; else players.push(player);
    saveLocal(); render();
  }
}
async function deletePlayer(id){
  if(!confirm('¿Eliminar jugador?')) return;
  if(firebaseReady){ await window.fs.deleteDoc(window.fs.doc(collectionRef(), id)); }
  else { players = players.filter(p => p.id !== id); saveLocal(); render(); }
  toast('Jugador eliminado');
}
function resetForm(){
  $('playerForm').reset();
  $('playerId').value = '';
  $('rating').value = 1000;
  $('formTitle').textContent = 'Nuevo jugador';
}
function editPlayer(id){
  const p = players.find(x => x.id === id); if(!p) return;
  $('playerId').value = p.id;
  $('name').value = p.name || '';
  $('birthYear').value = p.birthYear || '';
  $('level').value = p.level || 'Beginner';
  $('rating').value = p.rating || 1000;
  $('status').value = p.status || 'Active';
  $('notes').value = p.notes || '';
  $('formTitle').textContent = 'Editar jugador';
  location.hash = '#registro';
}
function csvExport(){
  const rows = ['name,birthYear,level,rating,status,notes', ...players.map(p => [p.name,p.birthYear,p.level,p.rating,p.status,p.notes].map(v => '"' + String(v ?? '').replaceAll('"','""') + '"').join(','))];
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([rows.join('\n')], { type:'text/csv' }));
  a.download = 'spin-club-players.csv';
  a.click();
}
function render(){ renderTable(); renderDashboard(); }
function renderDashboard(){
  const total = players.length;
  const active = players.filter(p => p.status === 'Active').length;
  const avg = total ? Math.round(players.reduce((s,p)=>s + Number(p.rating || 0),0) / total) : 0;
  const top = [...players].sort((a,b)=>Number(b.rating||0)-Number(a.rating||0))[0];
  $('totalPlayers').textContent = total;
  $('activePlayers').textContent = active;
  $('avgRating').textContent = avg;
  $('topPlayer').textContent = top ? top.name : '—';
  $('heroPlayers').textContent = active;
  $('heroAvg').textContent = avg;
}
function renderTable(){
  const q = $('searchInput').value.toLowerCase();
  const status = $('filterStatus').value;
  const filtered = [...players]
    .filter(p => (!q || (p.name || '').toLowerCase().includes(q)))
    .filter(p => status === 'All' || p.status === status)
    .sort((a,b)=>Number(b.rating||0)-Number(a.rating||0));
  $('playersTable').innerHTML = filtered.map(p => `
    <tr>
      <td><strong>${esc(p.name)}</strong><br><small>${esc(p.notes || '')}</small></td>
      <td>${esc(p.birthYear)}</td>
      <td>${esc(p.level)}</td>
      <td>${Number(p.rating || 0)}</td>
      <td><span class="pill">${esc(p.status || 'Active')}</span></td>
      <td><div class="actions-cell"><button class="icon-btn" onclick="editPlayer('${p.id}')">Editar</button><button class="icon-btn danger" onclick="deletePlayer('${p.id}')">Eliminar</button></div></td>
    </tr>`).join('');
}
function esc(v){ return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

$('playerForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const id = $('playerId').value || uid();
  const player = {
    id,
    name: $('name').value.trim(),
    birthYear: $('birthYear').value.trim(),
    level: $('level').value,
    rating: Number($('rating').value || 1000),
    status: $('status').value,
    notes: $('notes').value.trim(),
    updatedAt: new Date().toISOString()
  };
  if(!player.name || !player.birthYear){ toast('Falta nombre o año'); return; }
  await upsertPlayer(player);
  resetForm();
  toast('Jugador guardado');
});
$('resetForm').onclick = resetForm;
$('searchInput').oninput = renderTable;
$('filterStatus').onchange = renderTable;
$('exportCsv').onclick = csvExport;
window.editPlayer = editPlayer;
window.deletePlayer = deletePlayer;

window.addEventListener('DOMContentLoaded', async () => {
  loadLocal();
  render();
  await bootFirebase();
});
