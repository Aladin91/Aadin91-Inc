import { MathX } from './aladin-ai.js';
import { validateImage, toGrayscale, resizeNearest, sobelEdges } from './vision.js';

function gradients(image){
  const g=toGrayscale(image),gx=new Float64Array(g.data.length),gy=new Float64Array(g.data.length),mag=new Float64Array(g.data.length),ang=new Float64Array(g.data.length);
  const at=(x,y)=>g.data[Math.max(0,Math.min(g.height-1,y))*g.width+Math.max(0,Math.min(g.width-1,x))];
  for(let y=0;y<g.height;y++)for(let x=0;x<g.width;x++){
    const i=y*g.width+x,dx=at(x+1,y)-at(x-1,y),dy=at(x,y+1)-at(x,y-1);gx[i]=dx;gy[i]=dy;mag[i]=Math.hypot(dx,dy);let a=Math.atan2(dy,dx)*180/Math.PI;if(a<0)a+=180;if(a>=180)a-=180;ang[i]=a;
  }
  return {width:g.width,height:g.height,gx,gy,magnitude:mag,angle:ang};
}

export function hogDescriptor(image,{width=64,height=64,cellSize=8,bins=9,blockSize=2}={}){
  if(width%cellSize||height%cellSize)throw new Error('width and height must be divisible by cellSize');
  const r=resizeNearest(image,width,height),gr=gradients(r),cellsX=width/cellSize,cellsY=height/cellSize;
  const hist=Array.from({length:cellsY},()=>Array.from({length:cellsX},()=>Array(bins).fill(0)));
  const binWidth=180/bins;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=y*width+x,cx=Math.floor(x/cellSize),cy=Math.floor(y/cellSize),a=gr.angle[i]/binWidth,b0=Math.floor(a)%bins,b1=(b0+1)%bins,w1=a-Math.floor(a),w0=1-w1,m=gr.magnitude[i];hist[cy][cx][b0]+=m*w0;hist[cy][cx][b1]+=m*w1;
  }
  const vector=[];
  for(let by=0;by<=cellsY-blockSize;by++)for(let bx=0;bx<=cellsX-blockSize;bx++){
    const block=[];for(let y=0;y<blockSize;y++)for(let x=0;x<blockSize;x++)block.push(...hist[by+y][bx+x]);
    const norm=Math.sqrt(block.reduce((s,v)=>s+v*v,0)+1e-8);for(const v of block)vector.push(v/norm);
  }
  return {vector,width,height,cellSize,bins,blockSize};
}

function binaryOp(binary,radius,mode){
  validateImage(binary,'binary');if(binary.data.length!==binary.width*binary.height)throw new Error('binary image required');if(radius<1)return {width:binary.width,height:binary.height,data:Uint8Array.from(binary.data)};
  const out=new Uint8Array(binary.data.length),{width,height,data}=binary;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    let value=mode==='erode'?1:0,done=false;
    for(let dy=-radius;dy<=radius&&!done;dy++)for(let dx=-radius;dx<=radius;dx++){
      const xx=x+dx,yy=y+dy,v=(xx>=0&&yy>=0&&xx<width&&yy<height)?(data[yy*width+xx]?1:0):0;
      if(mode==='erode'&&!v){value=0;done=true;break;}if(mode==='dilate'&&v){value=1;done=true;break;}
    }
    out[y*width+x]=value;
  }
  return {width,height,data:out};
}

export function erode(binary,{radius=1}={}){return binaryOp(binary,radius,'erode');}
export function dilate(binary,{radius=1}={}){return binaryOp(binary,radius,'dilate');}
export function openMorphology(binary,{radius=1}={}){return dilate(erode(binary,{radius}),{radius});}
export function closeMorphology(binary,{radius=1}={}){return erode(dilate(binary,{radius}),{radius});}

export function structuralSimilarity(a,b,{resize=true}={}){
  let A=toGrayscale(a),B=toGrayscale(b);if(A.width!==B.width||A.height!==B.height){if(!resize)throw new Error('images must share dimensions');B=resizeNearest(B,A.width,A.height);}
  const av=[...A.data],bv=[...B.data],ma=MathX.mean(av),mb=MathX.mean(bv),va=MathX.variance(av),vb=MathX.variance(bv);let cov=0;for(let i=0;i<av.length;i++)cov+=(av[i]-ma)*(bv[i]-mb);cov/=Math.max(1,av.length);
  const c1=(0.01*255)**2,c2=(0.03*255)**2;return ((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2));
}

export function edgeDensity(image,{threshold=64}={}){const e=sobelEdges(image);return [...e.data].filter(v=>v>=threshold).length/Math.max(1,e.data.length);}

export function imageMoments(image,{binaryThreshold=null}={}){
  const g=toGrayscale(image);let m00=0,m10=0,m01=0,m20=0,m02=0,m11=0;
  for(let y=0;y<g.height;y++)for(let x=0;x<g.width;x++){const raw=g.data[y*g.width+x],w=binaryThreshold===null?raw:(raw>=binaryThreshold?1:0);m00+=w;m10+=x*w;m01+=y*w;m20+=x*x*w;m02+=y*y*w;m11+=x*y*w;}
  if(!m00)return {m00:0,centroid:{x:0,y:0},central:{mu20:0,mu02:0,mu11:0}};const cx=m10/m00,cy=m01/m00;return {m00,m10,m01,m20,m02,m11,centroid:{x:cx,y:cy},central:{mu20:m20-cx*m10,mu02:m02-cy*m01,mu11:m11-cx*m01}};
}
