// Optional archive authentication UI; no dependency on analysis calculations.
const host=document.querySelector('#archive-auth');
if(host){
 host.innerHTML=`<form id="archive-login" class="auth-form" hidden><label>Логин <input name="username" autocomplete="username" maxlength="256" required></label><label>Пароль <input name="password" type="password" autocomplete="current-password" maxlength="1024" required></label><button type="submit" class="ghost">Войти</button></form><button id="archive-logout" class="ghost" type="button" hidden>Выйти</button><p id="auth-message" aria-live="polite"></p>`;
 const form=host.querySelector('form'),logout=host.querySelector('#archive-logout'),message=host.querySelector('#auth-message');
 let state=null,csrf='',busy=false,statusRequest=0,statusTask=null;
 function apply(next){
  csrf=next.csrf||'';
  host.hidden=!next.enabled;
  form.hidden=!next.enabled||next.authenticated||next.configured===false;
  logout.hidden=!next.enabled||!next.authenticated;
  message.textContent=next.message||(next.authenticated?'Вход выполнен. Архив доступен.':'Для просмотра сохранённых анализов войдите в систему.');
  const signature=`${next.enabled}:${next.authenticated}`;
  if(state!==signature){state=signature;document.dispatchEvent(new CustomEvent('archive-auth-changed',{detail:next}))}
 }
 function updateStatus(){
  if(busy)return Promise.resolve();
  if(statusTask)return statusTask;
  statusTask=(async()=>{
  const request=++statusRequest;
  try{const response=await fetch('/auth/status',{cache:'no-store'});if(!response.ok)throw new Error('Не удалось проверить авторизацию');const next=await response.json();if(request===statusRequest)apply(next)}
  catch(e){if(request===statusRequest){apply({enabled:true,authenticated:false,configured:false,message:e.message})}}
  })().finally(()=>{statusTask=null});
  return statusTask;
 }
 async function submit(url,body){
  const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf},body:JSON.stringify(body)});
  const data=await response.json();if(!response.ok)throw new Error(data.detail||'Не удалось выполнить запрос');return data;
 }
 form.addEventListener('submit',async event=>{
  event.preventDefault();if(busy)return;if(statusTask)await statusTask;if(busy)return;busy=true;++statusRequest;
  const button=form.querySelector('button'),username=form.elements.username.value,password=form.elements.password.value;
  form.elements.password.value='';button.disabled=true;
  try{const data=await submit('/auth/login',{username,password});apply({enabled:true,configured:true,...data})}
  catch(e){message.textContent=e.message}finally{busy=false;button.disabled=false}
 });
 logout.addEventListener('click',async()=>{
  if(busy)return;if(statusTask)await statusTask;if(busy)return;busy=true;++statusRequest;logout.disabled=true;
  try{await submit('/auth/logout',{});apply({enabled:true,authenticated:false,configured:true});busy=false;await updateStatus()}
  catch(e){busy=false;await updateStatus();message.textContent=e.message}finally{busy=false;logout.disabled=false}
 });
 document.addEventListener('archive-auth-required',updateStatus);
 window.addEventListener('focus',updateStatus);
 window.addEventListener('pageshow',updateStatus);
 setInterval(()=>{if(!document.hidden)updateStatus()},30000);
 updateStatus();
}
