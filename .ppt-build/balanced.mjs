import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='C:/Users/Manan Kulshrestha/Downloads/AI_project';
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/presentation/Skin_Lesion_Project_Review_Illustrated.pptx'));
const choices={1:'research-workstation.png',2:'literature-review.png',3:'data-split.png',6:'feature-pipeline.png',8:'training-compute.png',13:'transformer-concept.png',18:'model-comparison.png',20:'federated-concept.png',24:'evaluation.png'};
for(const [index,file] of Object.entries(choices)){
 let s=p.slides.items[Number(index)];
 let body=s.shapes.items.filter(z=>z.position && z.position.left>=70 && z.position.left<100 && z.position.top>=130 && z.position.top<620 && z.position.width>900).sort((a,b)=>a.position.top-b.position.top);
 let gap=body.length===5?95:body.length===4?110:130;
 body.forEach((z,i)=>{z.position={left:75,top:145+i*gap,width:635,height:Math.min(gap-8,108)};});
 const bytes=await fs.readFile(root+'/presentation/visual-assets/'+file);
 s.images.add({blob:new Uint8Array(bytes),contentType:'image/png',position:{left:745,top:170,width:470,height:390},fit:'contain',alt:'Conceptual illustration supporting the slide topic'});
 s.speakerNotes.append('\nIllustration is AI-generated and conceptual; visual details are not project measurements.');
}
await fs.mkdir(root+'/.ppt-build/balanced',{recursive:true});for(let i=0;i<p.slides.items.length;i++){let b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(root+'/.ppt-build/balanced/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await b.arrayBuffer()));}
await (await PresentationFile.exportPptx(p)).save(root+'/.ppt-build/balanced-draft.pptx');
const {finalizePresentation}=await import('file:///C:/Users/Manan%20Kulshrestha/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations/container_tools/artifact_tool_utils.mjs');
await finalizePresentation({workspaceDir:root,candidatePath:root+'/.ppt-build/balanced-draft.pptx',finalPath:root+'/presentation/Skin_Lesion_Project_Review_Balanced.pptx',pythonExecutable:'C:/Users/Manan Kulshrestha/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:'C:/Users/Manan Kulshrestha/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:'C:/Users/Manan Kulshrestha/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000'],explicitTotalSlideCount:26,fontPolicy:{basis:'design',families:['Aptos']},verifyArtifactToolImport:true,receiptPath:root+'/.ppt-build/balanced-validation.json'});
