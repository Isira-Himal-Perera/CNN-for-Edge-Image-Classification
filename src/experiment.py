"""Real training/metric collection. Run: python run_experiments.py --dataset digits"""
import csv,json,random,time,argparse,platform,traceback,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import get_data,make_loaders,SEED
from .models import EdgeCNN,count_params,estimate_macs,load_pretrained,unfreeze_last_block


def seed_everything(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))


def optimizer_for(name,model,learning_rate=None):
    params=(p for p in model.parameters() if p.requires_grad)
    if name=='adam':return torch.optim.Adam(params,lr=learning_rate or 0.001)
    if name=='sgd':return torch.optim.SGD(params,lr=learning_rate or 0.02)
    if name=='momentum':return torch.optim.SGD(params,lr=learning_rate or 0.02,momentum=0.9)
    raise ValueError(name)


def run_epochs(model,loaders,epochs,optimizer,device='cpu',history=None):
    model.to(device)
    loss_fn=nn.CrossEntropyLoss()
    if history is None:history=[]
    best_val=min([h['val_loss'] for h in history],default=float('inf'))
    best_state=copy.deepcopy(model.state_dict())
    for ep in range(epochs):
        record={'epoch':len(history)+1}
        start=time.perf_counter()
        for phase in ['train','val']:
            model.train(phase=='train')
            # Frozen pretrained BatchNorm statistics should not drift on tiny datasets.
            if hasattr(model, '_pretrained_backbone'):
                model.features.eval()
            loss_sum=correct=total=0
            for xs,ys in loaders[phase]:
                xs,ys=xs.to(device),ys.to(device)
                if phase=='train':optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase=='train'):
                    logits=model(xs)
                    loss=loss_fn(logits,ys)
                    if phase=='train':loss.backward();optimizer.step()
                loss_sum+=loss.item()*len(ys)
                correct+=(logits.argmax(1)==ys).sum().item()
                total+=len(ys)
            record[f'{phase}_loss']=float(loss_sum/total)
            record[f'{phase}_accuracy']=float(correct/total)
        record['train_and_validation_seconds']=round(time.perf_counter()-start,4)
        history.append(record)
        if record['val_loss'] < best_val:
            best_val=record['val_loss']
            best_state=copy.deepcopy(model.state_dict())
        print(f"epoch {record['epoch']:02d} train-loss={record['train_loss']:.4f} val-loss={record['val_loss']:.4f} val-accuracy={record['val_accuracy']:.3f} time={record['train_and_validation_seconds']:.1f}s",flush=True)
    # Select model with smallest VALIDATION loss; test data are never used for selection.
    model.load_state_dict(best_state)
    return history


def evaluate(model,loader,num_classes,device='cpu'):
    model.eval();true=[];pred=[]
    with torch.no_grad():
        for x,y in loader:
            guesses=model(x.to(device)).argmax(1).cpu().numpy()
            pred.extend(guesses.tolist());true.extend(y.numpy().tolist())
    precision,recall,f1,_=precision_recall_fscore_support(true,pred,labels=list(range(num_classes)),average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(true,pred)),'precision_macro':float(precision),'recall_macro':float(recall),'f1_macro':float(f1),'confusion_matrix':confusion_matrix(true,pred,labels=list(range(num_classes))).tolist(),'test_samples':len(true)}


def plot_result(history,outpath,title):
    plt.figure(figsize=(6.6,3.7));plt.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],label='Training loss')
    plt.plot([x['epoch'] for x in history],[x['val_loss'] for x in history],label='Validation loss')
    plt.xlabel('Epoch');plt.ylabel('Cross entropy loss');plt.title(title);plt.legend();plt.grid(alpha=.18)
    plt.tight_layout();plt.savefig(outpath,dpi=180);plt.close()


def plot_confusion(cm,classes,outpath,title):
    arr=np.asarray(cm)
    fig,ax=plt.subplots(figsize=(6.1,5.3));im=ax.imshow(arr,cmap='Blues')
    ax.set_xticks(np.arange(len(classes)),labels=classes,rotation=45,ha='right',fontsize=8)
    ax.set_yticks(np.arange(len(classes)),labels=classes,fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i,str(arr[i,j]),ha='center',va='center',fontsize=8,color='white' if arr[i,j]>arr.max()*.5 else 'black')
    ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.8)
    fig.tight_layout();fig.savefig(outpath,dpi=180);plt.close(fig)


def dump_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2),encoding='utf8')


def describe_run(model,hist,metrics,output_dir,stem,classes,image_size,device):
    torch.save(model.cpu().state_dict(),output_dir/'models'/f'{stem}.pth')
    filebytes=(output_dir/'models'/f'{stem}.pth').stat().st_size
    rec={'name':stem,'params_total':count_params(model),'params_trainable':count_params(model,True),
         'size_kb':round(filebytes/1024,2),'size_mb':round(filebytes/1048576,3),
         'average_epoch_seconds':round(float(np.mean([h['train_and_validation_seconds'] for h in hist])),3),
         'estimated_macs':estimate_macs(model,image_size),
         'epochs':len(hist),'best_validation_epoch':int(min(hist,key=lambda h:h['val_loss'])['epoch']),'image_size':image_size,'metrics':metrics,'history':hist}
    dump_json(output_dir/f'{stem}.json',rec)
    plot_result(hist,output_dir/'plots'/f'{stem}_loss.png',stem.replace('_',' ').title())
    plot_confusion(metrics['confusion_matrix'],classes,output_dir/'plots'/f'{stem}_confusion.png',stem.replace('_',' ').title())
    return rec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['digits','beans'],default='digits')
    p.add_argument('--output',default='results')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--sota',action='store_true',help='Download ImageNet weights and fine-tune two SOTA networks')
    p.add_argument('--only-sota',action='store_true',help='Skip the four custom-model runs, if previously complete')
    p.add_argument('--sota-head-epochs',type=int,default=7)
    p.add_argument('--sota-finetune-epochs',type=int,default=3)
    args=p.parse_args()
    output_dir=Path(args.output);(output_dir/'models').mkdir(parents=True,exist_ok=True);(output_dir/'plots').mkdir(parents=True,exist_ok=True)
    seed_everything();torch.backends.mkldnn.enabled=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('device',device,'torch',torch.__version__,flush=True)
    splits,classes,metadata=get_data(args.dataset)
    image_size=splits['train'][0].shape[-1]
    loaders=make_loaders(splits)
    dump_json(output_dir/'dataset_metadata.json',{**metadata,'classes':classes,'split_sizes':{k:len(v[1]) for k,v in splits.items()},'hardware':{'device':device,'cpu':platform.processor(),'system':platform.platform()},'seed':SEED})
    all_results={}
    if not args.only_sota:
        for stem,lightweight,optim in [('model_a_adam',False,'adam'),('model_b_adam',True,'adam'),('model_b_sgd',True,'sgd'),('model_b_momentum',True,'momentum')]:
            print('RUNNING:',stem,flush=True);seed_everything()
            m=EdgeCNN(len(classes),lightweight)
            loaders_for_run=make_loaders(splits,seed=SEED)  # identical per-epoch shuffled batches for fair optimizer comparison
            hist=run_epochs(m,loaders_for_run,args.epochs,optimizer_for(optim,m),device)
            metrics=evaluate(m,loaders['test'],len(classes),device)
            all_results[stem]=describe_run(m,hist,metrics,output_dir,stem,classes,image_size,device)
            dump_json(output_dir/'results_summary.json',all_results)
    if args.sota:
        for name in ['mobilenet_v2','squeezenet1_1']:
            print('FINE TUNING IMAGE-NET PRETRAINED:',name,flush=True)
            seed_everything()
            try:
                m=load_pretrained(name,len(classes))
                hist=run_epochs(m,loaders,args.sota_head_epochs,optimizer_for('adam',m,0.001),device)
                unfreeze_last_block(m,name)
                hist=run_epochs(m,loaders,args.sota_finetune_epochs,optimizer_for('adam',m,0.0001),device,hist)
                metrics=evaluate(m,loaders['test'],len(classes),device)
                all_results[name]=describe_run(m,hist,metrics,output_dir,name,classes,image_size,device)
                dump_json(output_dir/'results_summary.json',all_results)
            except Exception as exc:
                # No invented accuracy! Logs show which weights failed to download.
                info={'model':name,'status':'not_completed','reason':str(exc),'traceback':traceback.format_exc()[-1600:]}
                dump_json(output_dir/f'{name}_FAILED.json',info)
                print('PRETRAINED RUN INCOMPLETE:',str(exc)[:250],flush=True)
    print('Finished. Real experiments:',list(all_results),flush=True)

if __name__=='__main__':main()
