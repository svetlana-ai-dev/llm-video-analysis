// Run with Node and Playwright available through NODE_PATH.
// Uses only this project's fixtures, with every HTTP request intercepted locally.
const {chromium} = require('playwright');
const fs = require('fs');
const path = require('path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');

(async()=>{
 const directories = fs.readdirSync(path.join(root,'results'),{withFileTypes:true}).filter(x=>x.isDirectory()&&!x.name.startsWith('.'));
 const fixtures = directories.map(x=>path.join(root,'results',x.name,'result.json')).filter(x=>fs.existsSync(x)).map(x=>JSON.parse(fs.readFileSync(x,'utf8')));
 assert(fixtures.length,'Existing result fixture required');
 const first = structuredClone(fixtures.find(x=>x.llm.analysis)||fixtures[0]);
 first.saved_result_path='fixture/first';
 // Ensure every LLM section is exercised, even when historical API was unavailable.
 first.llm={status:'completed',model:'fixture',analysis:{summary:'Резюме',facial_analysis:'Мимика',gaze_analysis:'Взгляд',head_movement_analysis:'Голова',speech_analysis:'Речь',cross_modal_analysis:'Сопоставление',timeline_insights:['Временной вывод'],observed_facts:['Факт'],interpretation:['Вывод'],limitations:['Ограничение']}};
 const second=structuredClone(first);second.filename='second.mp4';second.saved_result_path='fixture/second';second.video.blink_count+=3;
 const media=fs.readdirSync(root).filter(x=>x.endsWith('.mp4')).map(x=>({name:x,size:fs.statSync(path.join(root,x)).size})).sort((a,b)=>a.size-b.size)[0];
 assert(media,'Existing MP4 required for playback check');
 const videoBytes=fs.readFileSync(path.join(root,media.name));
 const records=[{id:'fixture-first',filename:first.filename,video_available:true,video_url:'/api/archive/fixture-first/video'}, {id:'fixture-second',filename:second.filename,video_available:false,video_url:null,warning:'Исходное видео не сохранено. Доступен полный сохранённый анализ.'}];
 const browser=await chromium.launch({headless:true,...(process.env.UI_BROWSER_CHANNEL?{channel:process.env.UI_BROWSER_CHANNEL}:{})});
 try{
 const page=await browser.newPage();const errors=[];let analysisCalls=0,archiveOnly=false,failStorage=false;
 const authEnabled=process.env.UI_AUTH_ENABLED==='true';let authenticated=false;
 page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/*',async route=>{
  const request=route.request(),url=new URL(request.url());
  assert.equal(url.origin,'http://archive.test','External request forbidden');
  const p=url.pathname;
  if(p==='/')return route.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(root,'app/templates/index.html'),'utf8')});
  if(p==='/static/app.js'||p==='/static/styles.css')return route.fulfill({contentType:p.endsWith('.js')?'text/javascript':'text/css',body:fs.readFileSync(path.join(root,'app',p.slice(1)),'utf8')});
  if(p==='/auth/assets/auth.js'||p==='/auth/assets/auth.css')return route.fulfill({contentType:p.endsWith('.js')?'text/javascript':'text/css',body:fs.readFileSync(path.join(root,'app/auth/static',path.basename(p)),'utf8')});
  if(p==='/auth/status')return route.fulfill({json:{enabled:authEnabled,configured:true,authenticated,csrf:'test-csrf'}});
  if(p==='/auth/login'){const body=request.postDataJSON();assert.equal(body.username,'admin');assert.equal(body.password,'Demo password 2026!');authenticated=true;return route.fulfill({json:{authenticated:true,csrf:'test-csrf'},headers:{'Set-Cookie':'video_archive_session=fixture-session; HttpOnly; SameSite=Strict; Path=/'}})}
  if(p==='/auth/logout'){authenticated=false;return route.fulfill({json:{authenticated:false},headers:{'Set-Cookie':'video_archive_session=; Max-Age=0; Path=/'}})}
  if(p==='/health')return route.fulfill({json:{capabilities:{llm:false},models:{}}});
  if(p==='/api/analyze'){assert(!archiveOnly,'Archive must not call analyze');analysisCalls++;const result=structuredClone(analysisCalls===1?first:second);if(failStorage)result.saved_result_path=null;return route.fulfill({json:result,headers:{'X-Archive-Status':failStorage?'failed':'saved'}})}
  if((p==='/api/archive'||p.startsWith('/api/archive/'))&&authEnabled){
   if(!authenticated||!(request.headers().cookie||'').includes('video_archive_session=fixture-session'))return route.fulfill({status:401,json:{detail:'Войдите в систему'}});
  }
  if(p==='/api/archive')return route.fulfill({json:records});
  const match=p.match(/^\/api\/archive\/(fixture-first|fixture-second)$/);
  if(match){const index=match[1]==='fixture-first'?0:1;return route.fulfill({json:{metadata:records[index],result:index===0?first:second}})}
  if(p==='/api/archive/fixture-first/video')return route.fulfill({contentType:'video/mp4',body:videoBytes});
  throw new Error('Unexpected request: '+p);
 });
 const upload=async()=>{await page.locator('#file').setInputFiles({name:media.name,mimeType:'video/mp4',buffer:videoBytes});await page.locator('#analyze').click();};
 const snapshot=()=>page.locator('#results-list > .result-card').evaluateAll(cards=>cards.map(card=>Array.from(card.querySelectorAll('[data-role]')).filter(x=>!['result-video','number'].includes(x.dataset.role)).map(x=>[x.dataset.role,x.textContent])));
 const comparison=()=>page.locator('#comparison').evaluate(x=>x.innerHTML);
 await page.goto('http://archive.test/');
 if(authEnabled){
  await page.locator('#archive-login').waitFor({state:'visible'});
  assert.equal(await page.locator('#archive-list button').count(),0);
  await page.locator('#archive-login [name=username]').fill('admin');await page.locator('#archive-login [name=password]').fill('Demo password 2026!');await page.locator('#archive-login button').click();
  await page.locator('#archive-logout').waitFor({state:'visible'});
 }else assert(await page.locator('#archive-auth').evaluate(x=>x.hidden));
 await upload();await page.waitForFunction(()=>window.videoAnalyses.length===1);
 await upload();await page.waitForFunction(()=>window.videoAnalyses.length===2);
 const fresh=await snapshot(),freshComparison=await comparison();
 const callsBefore=analysisCalls;archiveOnly=true;await page.reload();
 await page.locator('#archive-list button').nth(0).click();await page.waitForFunction(()=>window.videoAnalyses.length===1);
 await page.locator('#archive-list button').nth(1).click();await page.waitForFunction(()=>window.videoAnalyses.length===2);
 assert.deepEqual(await snapshot(),fresh,'Every screen role must match fresh analysis');
 assert.equal(await comparison(),freshComparison,'Comparison table and graph markup must match');
 assert(await page.locator('.result-card').nth(1).locator('[role=alert]').innerText());
 assert(await page.locator('.result-card').nth(1).locator('.result-video').evaluate(x=>x.classList.contains('hidden')));
 await page.locator('#archive-list button').nth(0).click();assert.equal(await page.locator('.result-card').count(),2,'Reopening must not duplicate');
 const playback=await page.locator('[data-role=result-video]').first().evaluate(async v=>{
  if(v.readyState<1)await new Promise((resolve,reject)=>{v.addEventListener('loadedmetadata',resolve,{once:true});v.addEventListener('error',()=>reject(new Error('Video cannot be decoded')),{once:true});setTimeout(()=>reject(new Error('Metadata timeout')),10000)});
  await v.play();await new Promise(resolve=>setTimeout(resolve,250));v.pause();return {duration:v.duration,time:v.currentTime};
 });
 assert(playback.duration>0&&playback.time>0,'Saved source must play');
 assert.equal(analysisCalls,callsBefore,'Archive opened without analysis requests');
 // A new result and an archived result must share the existing comparison.
 archiveOnly=false;await page.reload();await page.locator('#archive-list button').nth(0).click();await page.waitForFunction(()=>window.videoAnalyses.length===1);
 await upload();await page.waitForFunction(()=>window.videoAnalyses.length===2);assert.equal(await comparison(),freshComparison);
 // Storage failure keeps all analytics and the browser source available.
 failStorage=true;await upload();await page.waitForFunction(()=>window.videoAnalyses.length===3);
 assert((await page.locator('.result-card').nth(2).locator('[role=alert]').innerText()).includes('архив не сохранён'));
 assert(await page.locator('.result-card').nth(2).locator('[data-role=result-video]').getAttribute('src'));
 if(authEnabled){
  await page.locator('#archive-logout').click();await page.locator('#archive-login').waitFor({state:'visible'});
  await page.waitForFunction(()=>window.videoAnalyses.length===2);
  assert.equal(await page.locator('#archive-list button').count(),0);
  assert.equal(await page.locator('.result-card').count(),2,'New results must survive logout');
  assert((await page.locator('.result-card').nth(1).locator('[role=alert]').innerText()).includes('архив не сохранён'));
 }
 assert.equal(errors.length,0,errors.join('\n'));
 console.log(JSON.stringify({passed:true,authEnabled,screenRoles:fresh[0].length,comparisonMetrics:await page.locator('#comparison-table tbody tr').count(),playback,checks:['fresh/archive parity','all LLM sections','comparison graphs','legacy warning','no duplicates','no archive analysis requests','mixed comparison','storage failure retains result',...(authEnabled?['inline login','protected archive','logout retains new results']:['disabled auth hides login'])]}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
