"""Parameter counts and MAC estimates for the pretrained architectures."""
import json
from pathlib import Path
from torchvision import models
from torch import nn
from .models import count_params, estimate_macs

def main(output_dir='results',num_classes=None,input_size=None):
    metadata_path=Path(output_dir)/'dataset_metadata.json'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        num_classes=num_classes or len(metadata['classes'])
        input_size=input_size or int(metadata['input_pixels'].split('x')[0])
    num_classes=num_classes or 10
    input_size=input_size or 32
    m1=models.mobilenet_v2(weights=None,num_classes=num_classes)
    m2=models.squeezenet1_1(weights=None,num_classes=num_classes)
    info={}
    for name,m in [('mobilenet_v2',m1),('squeezenet1_1',m2)]:
        info[name]={'params_total':count_params(m),
                    'float32_weights_mb_approx':round(sum(p.numel()*p.element_size() for p in m.parameters())/1048576,3),
                    'estimated_macs':estimate_macs(m,input_size),
                    'status':'ARCHITECTURE ONLY: Random initialization used exclusively for counting; not trained.'}
    Path(output_dir).mkdir(exist_ok=True)
    (Path(output_dir)/'architecture_stats.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
