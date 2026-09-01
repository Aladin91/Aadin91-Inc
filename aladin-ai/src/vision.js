import { MathX } from './aladin-ai.js';
import { SeededRandom, Validation } from './advanced.js';

export function validateImage(image,name='image'){
  if(!image||!Number.isInteger(image.width)||!Number.isInteger(image.height)||image.width<=0||image.height<=0)throw new Error(`${name} must include positive integer width/height`);
  const data=image.data;
  if(!data||typeof data.length!=='number')throw new Error(`${name}.data is required`);
  const px=image.width*image.height;
  if(data.length!==px&&data.length!==px*4)throw new Error(`${name}.data must be grayscale (w*h) or RGBA (w*h*4)`);
  return image;
}

export function toGrayscale(image){
  validateImage(image); const {width,height,data}=image; const n=width*height;
  if(data.length===n)return {width,height,data:Float64Array.from(data)};
  const out=new Float64Array(n);
  for(let i=0,p=0;i<data.length;i+=4,p++)out[p]=0.2126*data[i]+0.7152*data[i+1]+0.0722*data[i+2];
  return {width,height,data:out};
}

export function imageFromCanvas(canvas){
  if(!canvas?.getContext)throw new Error('A canvas element is required');
  const ctx=canvas.getContext('2d',{willReadFrequently:true});
  const d=ctx.getImageData(0,0,canvas.width,canvas.height);
  return {width:d.width,height:d.height,data:Uint8ClampedArray.from(d.data)};
}

export async function imageFromSource(source,{maxWidth=null,maxHeight=null}={}){
  if(typeof document==='undefined')throw new Error('imageFromSource requires a browser environment');
  let bitmap;
  if(typeof createImageBitmap==='function'&&(source instanceof Blob||source instanceof ImageData||source instanceof HTMLImageElement||source instanceof HTMLCanvasElement)) bitmap=await createImageBitmap(source);
  else if(source instanceof HTMLImageElement) bitmap=source;
  else throw new Error('Unsupported image source');
  let w=bitmap.width,h=bitmap.height,scale=1;
  if(maxWidth&&w>maxWidth)scale=Math.min(scale,maxWidth/w);
  if(maxHeight&&h>maxHeight)scale=Math.min(scale,maxHeight/h);
  w=Math.max(1,Math.round(w*scale));h=Math.max(1,Math.round(h*scale));
  const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;canvas.getContext('2d').drawImage(bitmap,0,0,w,h);
  if(bitmap!==source&&bitmap.close)bitmap.close();
  return imageFromCanvas(canvas);
}

export function resizeNearest(image,newWidth,newHeight){
  const g=toGrayscale(image); const out=new Float64Array(newWidth*newHeight);
  for(let y=0;y<newHeight;y++)for(let x=0;x<newWidth;x++){
    const sx=Math.min(g.width-1,Math.floor(x*g.width/newWidth)),sy=Math.min(g.height-1,Math.floor(y*g.height/newHeight));
    out[y*newWidth+x]=g.data[sy*g.width+sx];
  }
  return {width:newWidth,height:newHeight,data:out};
}

export function histogram(image,{bins=256,normalize=true}={}){
  const g=toGrayscale(image),out=Array(bins).fill(0);
  for(const v of g.data){const b=Math.max(0,Math.min(bins-1,Math.floor((v/256)*bins)));out[b]++;}
  if(normalize){const n=g.data.length||1;return out.map(v=>v/n);}return out;
}

export function convolve(image,kernel,{normalize=false,clamp=true}={}){
  const g=toGrayscale(image); if(!Array.isArray(kernel)||!kernel.length||!Array.isArray(kernel[0]))throw new Error('kernel must be a 2D array');
  const kh=kernel.length,kw=kernel[0].length;if(kh%2===0||kw%2===0)throw new Error('kernel dimensions must be odd');
  const oy=Math.floor(kh/2),ox=Math.floor(kw/2),out=new Float64Array(g.data.length); let ksum=kernel.flat().reduce((a,b)=>a+b,0);if(!ksum)ksum=1;
  for(let y=0;y<g.height;y++)for(let x=0;x<g.width;x++){
    let s=0;for(let ky=0;ky<kh;ky++)for(let kx=0;kx<kw;kx++){const yy=Math.max(0,Math.min(g.height-1,y+ky-oy)),xx=Math.max(0,Math.min(g.width-1,x+kx-ox));s+=g.data[yy*g.width+xx]*kernel[ky][kx];}
    if(normalize)s/=ksum;if(clamp)s=Math.max(0,Math.min(255,s));out[y*g.width+x]=s;
  }
  return {width:g.width,height:g.height,data:out};
}

export const Kernels={
  blur3:[[1,1,1],[1,1,1],[1,1,1]],
  gaussian3:[[1,2,1],[2,4,2],[1,2,1]],
  sharpen:[[0,-1,0],[-1,5,-1],[0,-1,0]],
  laplacian:[[0,1,0],[1,-4,1],[0,1,0]],
  sobelX:[[-1,0,1],[-2,0,2],[-1,0,1]],
  sobelY:[[-1,-2,-1],[0,0,0],[1,2,1]]
};

export function sobelEdges(image){
  const g=toGrayscale(image),gx=convolve(g,Kernels.sobelX,{clamp:false}),gy=convolve(g,Kernels.sobelY,{clamp:false}),out=new Float64Array(g.data.length);
  for(let i=0;i<out.length;i++)out[i]=Math.min(255,Math.hypot(gx.data[i],gy.data[i]));
  return {width:g.width,height:g.height,data:out};
}

export function otsuThreshold(image){
  const g=toGrayscale(image),h=histogram(g,{normalize:false}); const total=g.data.length; let sum=0;for(let i=0;i<256;i++)sum+=i*h[i];
  let sumB=0,wB=0,best=0,maxVar=-1;
  for(let t=0;t<256;t++){wB+=h[t];if(!wB)continue;const wF=total-wB;if(!wF)break;sumB+=t*h[t];const mB=sumB/wB,mF=(sum-sumB)/wF,v=wB*wF*(mB-mF)**2;if(v>maxVar){maxVar=v;best=t;}}
  return best;
}

export function threshold(image,value=null,{invert=false}={}){
  const g=toGrayscale(image),t=value??otsuThreshold(g),out=new Uint8Array(g.data.length);
  for(let i=0;i<out.length;i++){const on=g.data[i]>=t;out[i]=(invert?!on:on)?1:0;}
  return {width:g.width,height:g.height,data:out,threshold:t};
}

export function connectedComponents(binary,{connectivity=8,minArea=1}={}){
  validateImage(binary,'binary'); const {width,height,data}=binary; const seen=new Uint8Array(width*height),components=[];
  const dirs=connectivity===4?[[1,0],[-1,0],[0,1],[0,-1]]:[[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];
  for(let i=0;i<data.length;i++){
    if(seen[i]||!data[i])continue;const q=[i];seen[i]=1;let qi=0,area=0,minX=Infinity,minY=Infinity,maxX=-1,maxY=-1,sumX=0,sumY=0;
    while(qi<q.length){const p=q[qi++],x=p%width,y=Math.floor(p/width);area++;sumX+=x;sumY+=y;minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
      for(const[dX,dY]of dirs){const xx=x+dX,yy=y+dY;if(xx<0||yy<0||xx>=width||yy>=height)continue;const ni=yy*width+xx;if(!seen[ni]&&data[ni]){seen[ni]=1;q.push(ni);}}
    }
    if(area>=minArea)components.push({area,bbox:{x:minX,y:minY,width:maxX-minX+1,height:maxY-minY+1},centroid:{x:sumX/area,y:sumY/area}});
  }
  return components.sort((a,b)=>b.area-a.area);
}

export function averageHash(image,{size=8}={}){
  const r=resizeNearest(image,size,size),m=MathX.mean([...r.data]);return [...r.data].map(v=>v>=m?'1':'0').join('');
}

export function differenceHash(image,{size=8}={}){
  const r=resizeNearest(image,size+1,size),bits=[];for(let y=0;y<size;y++)for(let x=0;x<size;x++)bits.push(r.data[y*(size+1)+x]>r.data[y*(size+1)+x+1]?'1':'0');return bits.join('');
}

export function hammingDistance(a,b){if(a.length!==b.length)throw new Error('hash lengths must match');let d=0;for(let i=0;i<a.length;i++)if(a[i]!==b[i])d++;return d;}

export function perceptualSimilarity(a,b,{method='dhash',size=8}={}){
  const ha=method==='ahash'?averageHash(a,{size}):differenceHash(a,{size}),hb=method==='ahash'?averageHash(b,{size}):differenceHash(b,{size});return 1-hammingDistance(ha,hb)/ha.length;
}

export function templateMatch(image,template,{step=1}={}){
  const img=toGrayscale(image),tpl=toGrayscale(template);if(tpl.width>img.width||tpl.height>img.height)throw new Error('template must not exceed image dimensions');
  const tMean=MathX.mean([...tpl.data]);let tVar=0;for(const v of tpl.data)tVar+=(v-tMean)**2;tVar=Math.sqrt(tVar)||1;let best={score:-Infinity,x:0,y:0};
  for(let y=0;y<=img.height-tpl.height;y+=step)for(let x=0;x<=img.width-tpl.width;x+=step){let mean=0,n=tpl.width*tpl.height;for(let ty=0;ty<tpl.height;ty++)for(let tx=0;tx<tpl.width;tx++)mean+=img.data[(y+ty)*img.width+x+tx];mean/=n;let num=0,den=0;for(let ty=0;ty<tpl.height;ty++)for(let tx=0;tx<tpl.width;tx++){const a=img.data[(y+ty)*img.width+x+tx]-mean,b=tpl.data[ty*tpl.width+tx]-tMean;num+=a*b;den+=a*a;}const score=num/((Math.sqrt(den)||1)*tVar);if(score>best.score)best={score,x,y};}
  return best;
}

export function colorKMeans(image,{clusters=4,maxIterations=30,seed=42}={}){
  validateImage(image);if(image.data.length!==image.width*image.height*4)throw new Error('colorKMeans requires RGBA input');
  const X=[];for(let i=0;i<image.data.length;i+=4)X.push([image.data[i],image.data[i+1],image.data[i+2]]);Validation.matrix(X);const rng=new SeededRandom(seed);const centroids=[];while(centroids.length<clusters)centroids.push([...X[rng.int(X.length)]]);let labels=[];
  for(let it=0;it<maxIterations;it++){labels=X.map(r=>centroids.map((c,i)=>[i,MathX.euclidean(r,c)]).sort((a,b)=>a[1]-b[1])[0][0]);let moved=0;for(let k=0;k<clusters;k++){const rows=X.filter((_,i)=>labels[i]===k);if(!rows.length)continue;const next=[0,1,2].map(j=>MathX.mean(rows.map(r=>r[j])));moved+=MathX.euclidean(next,centroids[k]);centroids[k]=next;}if(moved<1e-6)break;}
  return {labels,centroids,width:image.width,height:image.height};
}

export function imageDescriptor(image,{histogramBins=32,edgeBins=16}={}){
  const g=toGrayscale(image),h=histogram(g,{bins:histogramBins}),e=sobelEdges(g),eh=histogram(e,{bins:edgeBins});
  const edgeDensity=[...e.data].filter(v=>v>64).length/Math.max(1,e.data.length);const mean=MathX.mean([...g.data]),std=MathX.std([...g.data]);
  return {vector:[...h,...eh,mean/255,std/255,edgeDensity],stats:{mean,std,edgeDensity},hash:differenceHash(g)};
}

export function compareDescriptors(a,b){
  if(!a?.vector||!b?.vector)throw new Error('descriptors required');return {cosine:MathX.cosine(a.vector,b.vector),hashSimilarity:1-hammingDistance(a.hash,b.hash)/a.hash.length};
}
