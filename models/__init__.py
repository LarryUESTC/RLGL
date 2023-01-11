from models.Self.E2_SGRL import E2_SGRL
from models.Self.SGRL import SGRL
from models.Self.SUGRL import SUGRL
from models.Self.GMI import GMI
from models.Self.MVGRL import MVGRL
from models.Rein.RLGRL import RLGL
from models.Rein.RLG import RLG
from models.Rein.RLGDQN import RLGDQN
from models.Rein.GDPNet import GDPNet
from models.Rein.GDPNet2 import GDPNet2
from models.Rein.offlineRLG import offlineRLG
from models.NodeClas.Semi_GCN import GCN as Semi_GCN
from models.NodeClas.Semi_RGCN import RGCN as Semi_RGCN
from models.NodeClas.Semi_GCNMIXUP import GCNMIXUP
from models.NodeClas.Semi_SELFCONS import SELFCONS
from models.NodeClas.Sup_GCN import Sup_GCN
from models.NodeClas.Semi_CR import GCNCR
from models.VisionGraph.PyramidViG import PyramidViG as IMGCLS_VIG
from models.Brain.Self_Brain import SELFBRAIN
from models.Brain.Self_Brain_MLP import SELFBRAINMLP

method_dict = {
'E2_SGRL':E2_SGRL,
'SEMI_GCN':Semi_GCN,
'SEMI_RGCN':Semi_RGCN,
'SEMI_GCNCR': GCNCR,
'SEMI_SELFCONS':SELFCONS,
'SUP_GCN':Sup_GCN,
'GCNMIXUP': GCNMIXUP,
'RLGL':RLGL,
'OFFLINERLG':offlineRLG,
'SGRL':SGRL,
'RLG':RLG,
'RLGDQN':RLGDQN,
'SUGRL':SUGRL,
'GMI':GMI,
'MVGRL':MVGRL,
'IMGCLS_VIG':IMGCLS_VIG,
'GDP':GDPNet,
'GDP2':GDPNet2,
'SELFBRAIN':SELFBRAIN,
'SELFBRAINMLP':SELFBRAINMLP,
}


def getmodel(name):
    return method_dict[name.upper()]
