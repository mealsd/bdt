!function(){var o,r=window.location,l=window.document,c=l.currentScript,s=c.getAttribute("data-api")||new URL(c.src).origin+"/api/event",u=c.getAttribute("data-domain");function d(t,e,n){e&&console.warn("Ignoring Event: "+e),n&&n.callback&&n.callback(),"pageview"===t&&(o=!0)}var w,p,f=r.href,v={},h=-1,g=!1,e=!1;function n(){var t=l.body||{},e=l.documentElement||{};return Math.max(t.scrollHeight||0,t.offsetHeight||0,t.clientHeight||0,e.scrollHeight||0,e.offsetHeight||0,e.clientHeight||0)}function i(){var t=l.body||{},e=l.documentElement||{},n=window.innerHeight||e.clientHeight||0,e=window.scrollY||e.scrollTop||t.scrollTop||0;return a<=n?a:e+n}var a=n(),b=i();function m(){var t=w?p+(Date.now()-w):p;e||o||!(h<b||3e3<=t)||(h=b,setTimeout(function(){e=!1},300),t={n:"engagement",sd:Math.round(b/a*100),d:u,u:f,p:v,e:t},w=null,p=0,S(s,t))}function y(t,e){var n="pageview"===t;if(/^localhost$|^127(\.[0-9]+){0,2}\.[0-9]+$|^\[::1?\]$/.test(r.hostname)||"file:"===r.protocol)return d(t,"localhost",e);if((window._phantom||window.__nightmare||window.navigator.webdriver||window.Cypress)&&!window.__plausible)return d(t,null,e);try{if("true"===window.localStorage.plausible_ignore)return d(t,"localStorage flag",e)}catch(t){}var i={},t=(i.n=t,i.u=r.href,i.d=u,i.r=l.referrer||null,e&&e.meta&&(i.m=JSON.stringify(e.meta)),e&&e.props&&(i.p=e.props),c.getAttributeNames().filter(function(t){return"event-"===t.substring(0,6)})),a=i.p||{};t.forEach(function(t){var e=t.replace("event-",""),t=c.getAttribute(t);a[e]=a[e]||t}),i.p=a,n&&(o=!1,f=i.u,v=i.p,h=-1,p=0,w=Date.now(),g||(l.addEventListener("visibilitychange",function(){"hidden"===l.visibilityState?(p+=Date.now()-w,w=null,m()):w=Date.now()}),g=!0)),S(s,i,e)}function S(t,e,n){window.fetch&&fetch(t,{method:"POST",headers:{"Content-Type":"text/plain"},keepalive:!0,body:JSON.stringify(e)}).then(function(t){n&&n.callback&&n.callback({status:t.status})})}window.addEventListener("load",function(){a=n();var t=0,e=setInterval(function(){a=n(),15==++t&&clearInterval(e)},200)}),l.addEventListener("scroll",function(){a=n();var t=i();b<t&&(b=t)});var t=window.plausible&&window.plausible.q||[];window.plausible=y;for(var E,H=0;H<t.length;H++)y.apply(this,t[H]);function L(t){t&&E===r.pathname||(t&&g&&(m(),a=n(),b=i()),E=r.pathname,y("pageview"))}function _(){L(!0)}var k,T=window.history;T.pushState&&(k=T.pushState,T.pushState=function(){k.apply(this,arguments),_()},window.addEventListener("popstate",_)),"prerender"===l.visibilityState?l.addEventListener("visibilitychange",function(){E||"visible"!==l.visibilityState||L()}):L(),window.addEventListener("pageshow",function(t){t.persisted&&L()})}();