
let token = null;
const $ = id => document.getElementById(id);

$("file").addEventListener("change", e => { if(e.target.files[0]) upload(e.target.files[0]); });

const drop = $("drop");
["dragenter","dragover"].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault(); drop.style.borderColor="#6a6be5"}));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault(); drop.style.borderColor=""}));
drop.addEventListener("drop", e => { if(e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });

async function upload(file){
  setStatus("Uploading…");
  const fd = new FormData(); fd.append("file", file);
  const r = await fetch("/api/upload",{method:"POST",body:fd}); const d = await r.json();
  if(!r.ok){alert(d.error);setStatus("Error");return}
  token=d.token;
  $("fileInfo").classList.remove("hidden");
  $("fileInfo").innerHTML = `<b>${d.filename}</b><br>Rows: ${d.rows.toLocaleString()} · Columns: ${d.columns}<br>Missing values: ${d.missing_values}`;
  $("settingsDisabled").classList.add("hidden"); $("settings").classList.remove("hidden");
  const sel=$("columns"); sel.innerHTML="";
  d.column_names.forEach(c=>{const o=document.createElement("option");o.value=c;o.textContent=c;o.selected=true;sel.appendChild(o)});
  if(d.rows>5000){$("largeWarn").classList.remove("hidden");$("largeWarn").textContent=`Large dataset detected (${d.rows.toLocaleString()} rows). Spectral clustering can become expensive, so this demo will use a representative sample of up to 5,000 rows.`}
  else $("largeWarn").classList.add("hidden");
  setStatus("Dataset ready");
}

function setStatus(s){$("status").textContent=s}

async function runClustering(){
  if(!token)return;
  setStatus("Computing…");
  const cols=[...$("columns").selectedOptions].map(x=>x.value);
  const body={token,columns:cols,clusters:+$("clusters").value,affinity:$("affinity").value,gamma:+$("gamma").value,neighbors:+$("neighbors").value,sample_n:5000};
  const r=await fetch("/api/cluster",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const d=await r.json();
  if(!r.ok){alert(d.error);setStatus("Error");return}
  $("dashboard").classList.remove("hidden");
  const m=d.metrics;
  $("metrics").innerHTML=[
    ["Clusters",d.clusters],["Rows used",d.rows_used.toLocaleString()],
    ["Silhouette",m.silhouette==null?"—":m.silhouette.toFixed(3)],
    ["Features",d.features_used]
  ].map(x=>`<div class="metric"><div class="eyebrow">${x[0]}</div><div class="v">${x[1]}</div></div>`).join("");
  $("insights").innerHTML=d.insights.map(x=>`<li>${x}</li>`).join("");
  Plotly.newPlot("scatter",JSON.parse(d.scatter),{responsive:true});
  Plotly.newPlot("distribution",JSON.parse(d.distribution),{responsive:true});
  Plotly.newPlot("eigenvalues",JSON.parse(d.eigenvalues),{responsive:true});
  Plotly.newPlot("heatmap",JSON.parse(d.heatmap),{responsive:true});
  $("comparisonTable").innerHTML=`<table><thead><tr><th>Algorithm</th><th>Clusters</th><th>Silhouette</th></tr></thead><tbody>`+
    d.comparison.map(x=>`<tr><td>${x.algorithm}</td><td>${x.clusters}</td><td>${x.silhouette==null?"—":x.silhouette.toFixed(3)}</td></tr>`).join("")+"</tbody></table>";
  $("download").href=d.download;
  setStatus("Analysis complete");
  $("dashboard").scrollIntoView({behavior:"smooth"});
}

async function loadSample(type){
  // Generate a local sample CSV and upload it through the same API.
  let csv="";
  if(type==="iris"){
    csv="sepal_length,sepal_width,petal_length,petal_width\n5.1,3.5,1.4,0.2\n4.9,3.0,1.4,0.2\n6.2,3.4,5.4,2.3\n5.9,3.0,5.1,1.8\n6.0,2.2,4.0,1.0\n5.5,2.4,3.8,1.1\n4.6,3.1,1.5,0.2\n6.7,3.1,4.7,1.5\n5.0,3.4,1.5,0.2\n6.5,3.0,5.2,2.0";
  }else{
    csv="x,y\n1.0,0.2\n0.8,0.6\n0.3,0.9\n-0.2,0.8\n-0.7,0.4\n-1.0,0.0\n-0.7,-0.5\n-0.1,-0.8\n0.5,-0.7\n0.9,-0.3\n-0.2,0.1\n-0.4,0.5";
  }
  const blob=new Blob([csv],{type:"text/csv"});
  const file=new File([blob],type==="iris"?"iris_sample.csv":"moons_sample.csv",{type:"text/csv"});
  await upload(file);
}

$("themeBtn").onclick=()=>document.body.classList.toggle("light");
