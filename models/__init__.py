from models.Self.E2_SGRL import E2_SGRL
from models.Self.SGRL import SGRL
from models.Self.SUGRL import SUGRL
from models.Self.GMI import GMI
from models.Self.MVGRL import MVGRL
from models.Rein.RLGRL import RLGL
from models.Rein.RLG import RLG
from models.Rein.GDPNet import GDPNet
from models.NodeClas.Semi_GCN import GCN as Semi_GCN
from models.NodeClas.Semi_RGCN import RGCN as Semi_RGCN
from models.NodeClas.Semi_GCNMIXUP import GCNMIXUP
from models.NodeClas.Semi_SELFCONS import SELFCONS
from models.NodeClas.Sup_GCN import Sup_GCN
from models.VisionGraph.PyramidViG import PyramidViG as IMGCLS_VIG


method_dict = {
'E2_SGRL':E2_SGRL,
'SEMI_GCN':Semi_GCN,
'SEMI_RGCN':Semi_RGCN,
'SEMI_SELFCONS':SELFCONS,
'SUP_GCN':Sup_GCN,
'GCNMIXUP': GCNMIXUP,
'RLGL':RLGL,
'SGRL':SGRL,
'RLG':RLG,
'SUGRL':SUGRL,
'GMI':GMI,
'MVGRL':MVGRL,
'IMGCLS_VIG':IMGCLS_VIG,
'GDP':GDPNet,
}


def getmodel(name):
    return method_dict[name.upper()]
