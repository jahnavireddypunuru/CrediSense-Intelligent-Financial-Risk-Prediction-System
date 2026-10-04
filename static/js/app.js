// theme toggle (persisted)
(function(){
  try{var t=localStorage.getItem('cs-theme');if(t)document.documentElement.dataset.theme=t;}catch(e){}
  document.addEventListener('click',function(e){
    var b=e.target.closest('[data-theme-toggle]');if(!b)return;
    var d=document.documentElement,n=d.dataset.theme==='dark'?'light':'dark';
    d.dataset.theme=n;try{localStorage.setItem('cs-theme',n);}catch(e){}
    document.dispatchEvent(new Event('themechange'));
  });
  document.addEventListener('click',function(e){
    var b=e.target.closest('[data-pw]');if(!b)return;
    var i=document.getElementById(b.dataset.pw);i.type=i.type==='password'?'text':'password';
    b.textContent=i.type==='password'?'Show':'Hide';
  });
})();
function cssVar(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
