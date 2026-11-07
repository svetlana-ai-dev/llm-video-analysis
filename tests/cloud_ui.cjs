// Isolated form-handler tests: no browser, network, media or AI calls.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../app/static/app.js'), 'utf8');
const handler = source.slice(source.indexOf('$("#cloud-form").addEventListener'), source.indexOf('\nfunction renderResult'));

(async () => {
  const nodes = Object.fromEntries(['cloud-form', 'cloud-url', 'cloud-add', 'cloud-status'].map(id => [id, {disabled:false, value:''}]));
  let submit, requests = [], rendered = [], comparisons = 0;
  nodes['cloud-form'].addEventListener = (event, callback) => { assert.equal(event, 'submit'); submit = callback; };
  const context = {
    $: selector => nodes[selector.slice(1)], analyses: [], crypto: {randomUUID: () => 'fixture'},
    renderResult: entry => rendered.push(entry), renderComparison: () => comparisons++,
    fetch: async (url, options) => { requests.push([url,options]); return context.reply; },
  };
  vm.runInNewContext(handler, context);
  const event = {preventDefault(){}};
  nodes['cloud-url'].value = ' https://example.com/meeting.mp4 ';
  context.reply = {ok:false,json:async()=>({detail:'Видео недоступно (HTTP 404).'})};
  await submit(event);
  assert.match(nodes['cloud-status'].textContent,/HTTP 404/);
  assert.match(nodes['cloud-status'].textContent,/файл с компьютера/);
  assert.equal(nodes['cloud-add'].disabled,false);
  assert.equal(nodes['cloud-url'].disabled,false);
  assert.equal(rendered.length,0);

  context.reply = {ok:true,json:async()=>({filename:'meeting.mp4',saved_result_path:'fixture/result'})};
  await submit(event);
  assert.equal(rendered.length,1);
  assert.equal(context.analyses.length,1);
  assert.equal(comparisons,1);
  assert.equal(requests[1][0],'/api/analyze-url');
  assert.equal(JSON.parse(requests[1][1].body).url,'https://example.com/meeting.mp4');
  assert.match(nodes['cloud-status'].textContent,/сохранены в архиве/);

  context.reply = {ok:true,json:async()=>({filename:'meeting.mp4',saved_result_path:null})};
  await submit(event);
  assert.match(nodes['cloud-status'].textContent,/архив не сохранён/);
  context.reply = {ok:false,json:async()=>{throw Error('bad json')}};
  await submit(event);
  assert.match(nodes['cloud-status'].textContent,/некорректный ответ/);
  context.fetch = async()=>{throw Error('Network unavailable')};
  await submit(event);
  assert.match(nodes['cloud-status'].textContent,/Network unavailable/);
  assert.equal(nodes['cloud-add'].disabled,false);
  assert.equal(nodes['cloud-url'].disabled,false);
  const before=requests.length;
  nodes['cloud-add'].disabled=true;
  await submit(event);
  assert.equal(requests.length,before);
  console.log('Cloud UI: 6 scenarios passed (error recovery, success, storage failure, invalid JSON, network failure, duplicate submit).');
})().catch(error=>{console.error(error);process.exitCode=1});
